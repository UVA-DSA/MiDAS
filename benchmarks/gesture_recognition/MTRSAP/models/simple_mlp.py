import torch
import torch.nn as nn
import torch.nn.functional as F

# Option 1: Simple MLP with Temporal Pooling (Best for small datasets)
class TemporalPoolingMLP(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_classes=7, dropout=0.5, use_batchnorm=False):
        super().__init__()
        self.feature_dim = feature_dim

        self.use_batchnorm = use_batchnorm
        # MLP layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        # Average pooling over temporal dimension
        x = x.mean(dim=1)  # (batch_size, feature_dim)
        
        # MLP
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


# Option 2: LSTM-based classifier (Better for temporal patterns)
class LSTMClassifier(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=256, num_layers=2, 
                 num_classes=7, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Classification
        out = self.fc(last_hidden)
        return out


# Option 3: Temporal Convolutional Network (Good balance)
class TemporalConvNet(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_classes=7, dropout=0.5):
        super().__init__()
        
        # 1D convolutions over time
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        # Transpose for Conv1d: (batch_size, feature_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch_size, hidden_dim // 2)
        
        x = self.fc(x)
        return x


# Training example
def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
    
    return best_val_acc

def create_model(model_type='mlp', device='cuda', args=None):
    kwargs = {
        'feature_dim': 2048,
        'hidden_dim': 512,
        'num_classes': args.num_classes}
    
    if model_type == 'mlp':
        model = TemporalPoolingMLP(**kwargs)
        model = model.to(device)
        return model
    elif model_type == 'lstm':
        model = LSTMClassifier(**kwargs)
        model = model.to(device)
        return model
    elif model_type == 'tcn':
        model = TemporalConvNet(**kwargs)
        model = model.to(device)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")



# Usage example
if __name__ == '__main__':
    # Initialize model (choose one)
    model = TemporalPoolingMLP(feature_dim=2048, hidden_dim=512, num_classes=7)
    # model = LSTMClassifier(feature_dim=2048, hidden_dim=256, num_classes=7)
    # model = TemporalConvNet(feature_dim=2048, hidden_dim=512, num_classes=7)
    
    # Test with dummy data
    batch_size = 4
    seq_len = 30  # Variable length clips
    feature_dim = 2048  # ResNet features
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (batch_size, num_classes)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")