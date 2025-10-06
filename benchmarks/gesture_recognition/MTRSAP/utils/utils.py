#import from models folder transtcn
from models.mmtransformer import ModularMultimodalTransformer
from models.transtcn import TransformerModel, MultimodalFusion
from models.tcn import TCNClassifier, MultiBranchTCNClassifier
from models.mstcn import MS_TCN2
import torch
from datautils.ems import *
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from functools import partial
import torch.nn.functional as F

from datautils.midas import MultimodalGestureDataset



fusion = MultimodalFusion()

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, num_classes, class_counts):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.class_counts = torch.Tensor(class_counts)
        self.weights = (1 - beta) / (1 - beta ** self.class_counts)
        self.weights = self.weights / self.weights.sum()  # Normalize weights

    def forward(self, logits, labels):
        weights = self.weights.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, labels, weight=weights)
        return loss

def init_model(args, device):
    model = TransformerModel(args)
    model.to(device)

    num_classes = len(args.dataloader_params["keysteps"])
    class_counts = args.dataloader_params["train_class_stats"]
    val_class_counts = args.dataloader_params["val_class_stats"]

    print("Training class counts: ", class_counts)
    print("Validation class counts: ", val_class_counts)
    # update class_counts with missing classes from keysteps with 0 count
    for key in args.dataloader_params["keysteps"].keys():
        if key not in class_counts.keys():
            class_counts[key] = 0

    # reorganize class_counts to match the order of the keysteps
    class_counts = {key: class_counts[key] for key in args.dataloader_params["keysteps"].keys()}

    # convert dictionary values to list
    class_counts = [class_counts[key] for key in class_counts.keys()]
    class_counts = torch.Tensor([max(1, count) for count in class_counts])

    print("Class counts: ", class_counts, len(class_counts))
           
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_params["lr"], weight_decay=args.learning_params["weight_decay"])
    # criterion = nn.CrossEntropyLoss()
    
    # Class balanced loss
    criterion = ClassBalancedLoss(beta=0.99, num_classes=num_classes, class_counts=class_counts)
        
    return model, optimizer, criterion


def preprocess(x, modality, backbone, device, task='classification'):
    global fusion
    # print("-*" * 10, "Preprocessing", "*" * 10, "=" * 10)
    # check the shape of the input tensor
    feature = None
    label = x['keystep_id']
    # print(f"\nSubject ID: {x['subject_id']}, Trial ID: {x['trial_id']}, Start Frame: {x['start_frame']}, End Frame: {x['end_frame']}, Start Time: {x['start_t']}, End Time: {x['end_t']}")

    if task == 'segmentation':
        majority_label, _ = torch.mode(label, dim=1)  # [batch_size], mode returns (values, indices)
        label = majority_label


    if('video' in modality):
        feature = None
        x = x['frames']
        # extract resnet50 features
        x = x.to(device)
        x = backbone.extract_resnet(x)
        feature = x

    elif ( 'audio' in modality and  'resnet_ego' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet = resnet.to(device)

        smartwatch = x['smartwatch'].float()
        smartwatch = smartwatch.to(device)
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        audio = x['audio']
        audio = audio.to(device)
        # print("Raw Audio shape: ", audio.shape)
        audio_feature = backbone.extract_wav2vec_features(audio, multimodal=True) # for wav2vec features
        # print("Resnet feature shape: ", resnet.shape)
        # print("Audio feature shape: ", audio_feature.shape)
        # print("Smartwatch feature shape: ", smartwatch.shape)
        # print("Resnet feature shape: ", resnet.shape)

        fusion = fusion.to(device)
        fused = fusion(audio_feature, resnet, smartwatch)  # [B, T_common, D_total]


        feature = fused.float()

        # feature = torch.cat((resnet, audio_feature, smartwatch), dim=-1).float()

    elif ( 'audio' in modality and  'resnet_ego' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet = resnet.to(device)

        audio = x['audio']
        audio = audio.to(device)
        # print("Raw Audio shape: ", audio.shape)
        # audio_feature = backbone.extract_wav2vec_features(audio, multimodal=True) # for wav2vec features
        # print("Resnet feature shape: ", resnet.shape)
        audio_feature = backbone.extract_mel_spectrogram(audio, multimodal=True) # for mel spectrogram features

        feature = torch.cat((resnet, audio_feature), dim=1).float()

    elif ( 'flow' in modality and  'rgb' in modality and  'smartwatch' in modality):

        # I3D features are already extracted
        flow = x['flow'].float()
        rgb = x['rgb'].float()
        smartwatch = x['smartwatch'].float()

        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        # concatenate all features
        feature = torch.cat((flow, rgb, smartwatch), dim=-1).float()
        
    elif ( 'flow' in modality and  'rgb' in modality):

        # I3D features are already extracted
        flow = x['flow'].float()
        rgb = x['rgb'].float()
        feature = torch.cat((flow, rgb), dim=-1).float()

    elif ('resnet_ego' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, smartwatch), dim=-1).float()

    elif ('resnet_ego' in modality and 'resnet_exo' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet_exo = x['resnet_exo'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, resnet_exo, smartwatch), dim=-1).float()


    elif ('resnet_ego' in modality and 'resnet_exo' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet_exo = x['resnet_exo'].float()
        feature = torch.cat((resnet, resnet_exo), dim=-1).float()


    elif ('resnet_ego' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_ego'].float()

    elif ('resnet_exo' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_exo'].float()

    elif ('clip_ego' in modality):
        # resnet50 features are already extracted
        feature = x['clip_ego'].float()
        # print("Clip ego feature shape: ", feature.shape)
    elif ('clip_exo' in modality):
        # resnet50 features are already extracted
        feature = x['clip_exo'].float()
        # print("Clip exo feature shape: ", feature.shape)

    elif ('clip_ego' in modality and 'clip_exo' in modality):
        # resnet50 features are already extracted
        feature = torch.cat((x['clip_ego'].float(), x['clip_exo'].float()), dim=-1)
        # print("Clip ego and exo feature shape: ", feature.shape)

    elif ('rgb' in modality):
        # I3D features are already extracted
        feature = x['rgb'].float()

    elif ('flow' in modality):
        # I3D features are already extracted
        feature = x['flow'].float()

    # elif ('audio' in modality):
    #     # Audio features are already extracted

    #     # Example batch of audio clips (batch, samples, channels)
    #     audio_clips = x['audio']  # Assume shape [batch, samples, channels]
    #     audio_clips = audio_clips.to(device)
    #     feature = backbone.extract_mel_spectrogram(audio_clips)

    elif ('smartwatch' in modality):
        # Audio features are already extracted
        smartwatch = x['smartwatch'].float()
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        feature = smartwatch

    elif ('audio' in modality): # uncomment this if you want to use wav2vec features
        audio_clips = x['audio']  # Assume shape [batch, samples, channels]
        audio_clips = audio_clips.to(device)
        # feature = backbone.extract_wav2vec_features(audio_clips)
        feature = backbone.extract_mel_spectrogram(audio_clips)

        # print("Wav2Vec feature shape: ", feature.shape)

    feature_size = feature.shape[-1]
    # print("Feature shape: ", feature.shape, "\n")

    if(feature is not None):
        feature = feature.to(device)
        label = label.to(device)

    return feature, feature_size, label


# add wandb logging
def train_one_epoch(model, train_loader, criterion, optimizer, device, logger, modality, task='classification'):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:
            # print("Batch: ", i)
            print("=" * 10, "-" * 10, "=" * 10)
            input,feature_size, label = preprocess(batch, modality, model, device, task=task)

            # get more info about input
            keystep_label = batch['keystep_label'] if task == 'segmentation' else batch['keystep_label'][0]
            keystep_id = batch['keystep_id'] if task == 'segmentation' else batch['keystep_id'][0]
            start_frame = batch['start_frame'] if task == 'segmentation' else batch['start_frame'][0]
            end_frame = batch['end_frame'] if task == 'segmentation' else batch['end_frame'][0]
            start_t = batch['start_t'] if task == 'segmentation' else batch['start_t'][0]
            end_t = batch['end_t'] if task == 'segmentation' else batch['end_t'][0]
            subject_id = batch['subject_id'] if task == 'segmentation' else batch['subject_id']
            trial_id = batch['trial_id'] if task == 'segmentation' else batch['trial_id']
            window_start_frame = batch['window_start_frame'] if task == 'segmentation' else torch.tensor(-1)
            window_end_frame = batch['window_end_frame'] if task == 'segmentation' else torch.tensor(-1)

            if task == 'segmentation':
                print(f"Subject ID: {subject_id[0][0]}, Trial ID: {trial_id[0][0]}, Start Frame: {start_frame[0][0]}, End Frame: {end_frame[0][0]}, Start Time: {start_t[0][0]}, End Time: {end_t[0][0]}")
                print(f"Keystep Label: {keystep_label[0][0]}, Keystep ID: {keystep_id[0][0]}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")

            else:
                print(f"Subject ID: {subject_id}, Trial ID: {trial_id}, Start Frame: {start_frame}, End Frame: {end_frame}, Start Time: {start_t}, End Time: {end_t}")
                print(f"Keystep Label: {keystep_label}, Keystep ID: {keystep_id}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")


                    # ←—— ADDED CHECK ———→
            # if the time-dimension is zero, skip this batch
            # (inputs.shape == [B, T, F] or [B, C, T] depending on your preprocess)
            if input.size(1) == 0 or (task== 'segmentation' and input.size(1) != 150 and "audio" not in modality): 
                print(f"Skipping batch {i}: feature sequence : {input.shape}")
                continue

            if torch.isnan(input).any():
                print(f"⚠️ Skipping batch {i} because NaN")
                continue

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 1 == 0:
                print("\n")
                print("*" * 10, "=" * 10, "*" * 10)
                print(f"Pred: {torch.argmax(output, dim=1)} GT: {label}")
                logger.log({"train_loss": loss.item()})
                print(f"Batch: {i}, Loss: {loss.item()}")
                print("*" * 10, "=" * 10, "*" * 10)
                print("\n")
            # break
        
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            # print(f"Batch data: {batch}")
            continue

    return total_loss / len(train_loader)


# validate the model 
def validate(model, val_loader, criterion, device, logger, modality, task='classification'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                input,feature_size, label = preprocess(batch, modality, model, device, task=task)

                # check if the time-dimension is zero, skip this batch
                if input.size(1) == 0:
                    print(f"Skipping batch {i}: empty feature sequence : {input.shape}")
                    continue

                if torch.isnan(input).any():
                    print(f"⚠️ Skipping batch {i} because NaN")
                    continue

                output = model(input)
                loss = criterion(output, label)
                total_loss += loss.item()
                if i % 100 == 0:
                    logger.log({"val_loss": loss.item()})
            # break
            
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                print(f"Batch data: {batch}")
                continue
            
    return total_loss / len(val_loader)


# test the model
def test_model(model, test_loader, criterion, device, logger, epoch, results_dir, modality, task='classification'):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                input,feature_size, label = preprocess(batch, modality, model, device, task=task)
                print("=" * 10, "-" * 10, "=" * 10)
                print(f"[TEST] Batch: {i}")

                # check if the time-dimension is zero, skip this batch
                if input.size(1) == 0:
                    print(f"Skipping batch {i}: empty feature sequence : {input.shape}")
                    continue

                # get more info about input
                keystep_label = batch['keystep_label'] if task == 'segmentation' else batch['keystep_label'][0]
                keystep_id = batch['keystep_id'] if task == 'segmentation' else batch['keystep_id'][0]
                start_frame = batch['start_frame'] if task == 'segmentation' else batch['start_frame'][0]
                end_frame = batch['end_frame'] if task == 'segmentation' else batch['end_frame'][0]
                start_t = batch['start_t'] if task == 'segmentation' else batch['start_t'][0]
                end_t = batch['end_t'] if task == 'segmentation' else batch['end_t'][0]
                subject_id = batch['subject_id'] if task == 'segmentation' else batch['subject_id']
                trial_id = batch['trial_id'] if task == 'segmentation' else batch['trial_id']
                window_start_frame = batch['window_start_frame'] if task == 'segmentation' else torch.tensor(-1)
                window_end_frame = batch['window_end_frame'] if task == 'segmentation' else torch.tensor(-1)

                if task == 'segmentation':
                    print(f"Subject ID: {subject_id[0][0]}, Trial ID: {trial_id[0][0]}, Start Frame: {start_frame[0][0]}, End Frame: {end_frame[0][0]}, Start Time: {start_t[0][0]}, End Time: {end_t[0][0]}")
                    print(f"Keystep Label: {keystep_label[0][0]}, Keystep ID: {keystep_id[0][0]}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")
                else:
                    print(f"Subject ID: {subject_id}, Trial ID: {trial_id}, Start Frame: {start_frame}, End Frame: {end_frame}, Start Time: {start_t}, End Time: {end_t}")
                    print(f"Keystep Label: {keystep_label}, Keystep ID: {keystep_id}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")  
                
                if torch.isnan(input).any():
                    print(f"⚠️ Skipping batch {i} because NaN")
                    continue
                
                output = model(input)
                pred = torch.argmax(output, dim=1)
                print(f"Model Pred: {pred.item()}")

                gt.append(label.item())
                preds.append(pred.item())

                preds_detail.append({
                    "keystep_label": keystep_label,
                    "keystep_id": keystep_id.tolist(),
                    "start_frame": start_frame.tolist(),
                    "end_frame": end_frame.tolist(),
                    "start_t": start_t.tolist(),
                    "end_t": end_t.tolist(),
                    "window_start_frame": window_start_frame.item(),
                    "window_end_frame": window_end_frame.item(),
                    "subject_id": subject_id[0],
                    "trial_id": trial_id[0],
                    "pred_keystep_id": pred.item(),
                    "all_preds": output.tolist()
                })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)
    
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])

    # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["keystep_label", "keystep_id", "start_frame", "end_frame", "start_t", "end_t","window_start_frame","window_end_frame", "subject_id", "trial_id", "pred_keystep_id","all_preds"])
        for pred in preds_detail:
            writer.writerow([pred["keystep_label"], pred["keystep_id"], pred["start_frame"], pred["end_frame"], pred["start_t"], pred["end_t"], pred["window_start_frame"],pred["window_end_frame"], pred["subject_id"], pred["trial_id"], pred["pred_keystep_id"], pred["all_preds"]])
    return results



# return train,val,test dataloaders using the VideoDataset class
def get_dataloaders(args):
    train_dataset = VideoDataset(base_path=args.dataloader_params["base_path"], fold=args.dataloader_params["fold"], skip_frames=25, transform=tfs, clip_length_in_frames=args.dataloader_params["observation_window"], train=True)
    test_dataset = VideoDataset(base_path=args.dataloader_params["base_path"], fold=args.dataloader_params["fold"], skip_frames=25, transform=tfs, clip_length_in_frames=args.dataloader_params["observation_window"], train=False)

    split_indices_path = f'{args.dataloader_params["base_path"]}/val_test_split_indices_fold_0{args.dataloader_params["fold"]}.npz'

    if os.path.exists(split_indices_path):
        # Load pre-existing indices
        split_data = np.load(split_indices_path)
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']
    else:
        # Create new split and save the indices
        total_size = len(test_dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        val_size = int(0.5 * total_size)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        # Save the indices for later use
        np.savez(split_indices_path, val_indices=val_indices, test_indices=test_indices)
    
        # Subset datasets based on indices
    val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)


    # Create DataLoaders for training and validation subsets
    train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

    print("train dataset size: ", len(train_dataset))
    print("val dataset size: ", len(val_dataset))
    print("test dataset size: ", len(test_dataset))

    return train_loader, val_loader, test_loader



# ''' ***** EGOEXOEMS DATASET ***** '''


# # add wandb logging
# def eee_train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
#     model.train()
#     total_loss = 0
#     for i, batch in enumerate(train_loader):

#         i3d_rgb_features = batch['rgb']
#         i3d_flow_features = batch['flow']

#         # move to device
#         i3d_rgb_features = i3d_rgb_features.to(device)
#         i3d_flow_features = i3d_flow_features.to(device)

#         # get labels
#         labels = batch['keystep_id']
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         output = model(i3d_rgb_features)


#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         if i % 1 == 0:
#             print("\n ***** ")
#             print(batch['frames'].shape, batch['audio'].shape, batch['flow'].shape, batch['rgb'].shape, batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'],batch['start_t'], batch['end_t'],  batch['subject_id'], batch['trial_id'])
#             print(f"Pred: {torch.argmax(output, dim=1)} GT: {labels}")
#             logger.log({"train_loss": loss.item()})
#             print(f"Batch: {i}, Loss: {loss.item()}")
#             print(" ***** \n")

#     return total_loss / len(train_loader)


# # validate the model 
# def eee_validate(model, val_loader, criterion, device, logger):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):

#             i3d_rgb_features = batch['rgb']
#             i3d_flow_features = batch['flow']

#             # move to device
#             i3d_rgb_features = i3d_rgb_features.to(device)
#             i3d_flow_features = i3d_flow_features.to(device)

#             # get labels
#             labels = batch['keystep_id']
#             labels = labels.to(device)

#             output = model(i3d_rgb_features)

#             loss = criterion(output, labels)
#             total_loss += loss.item()
#             if i % 1 == 0:
#                 logger.log({"val_loss": loss.item()})

#     return total_loss / len(val_loader)


# # test the model
# def eee_test_model(model, test_loader, criterion, device, logger, epoch, results_dir):
#     model.eval()
#     total_loss = 0


#     accuracy = 0.0
#     gt = []
#     preds = []
    

#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):

#             i3d_rgb_features = batch['rgb']
#             i3d_flow_features = batch['flow']

#             # move to device
#             i3d_rgb_features = i3d_rgb_features.to(device)
#             i3d_flow_features = i3d_flow_features.to(device)

#             # get labels
#             labels = batch['keystep_id']
#             labels = labels.to(device)

#             output = model(i3d_rgb_features)
#             pred = torch.argmax(output, dim=1)
#             gt.append(labels.item())
#             preds.append(pred.item())
    
#     # Calculate metrics
#     accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
#     precision = precision_score(gt, preds, average='macro')
#     recall = recall_score(gt, preds, average='macro')
#     f1 = f1_score(gt, preds, average='macro')

#     # Log metrics to wandb
#     logger.log({
#         "test_accuracy": accuracy,
#         "test_precision": precision,
#         "test_recall": recall,
#         "test_f1": f1,
#         "epoch": epoch
#     })
    
#     # Save metrics to CSV
#     metrics_path = f'{results_dir}/metrics.csv'
#     with open(metrics_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not os.path.isfile(metrics_path):
#             writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
#         writer.writerow([epoch, accuracy, precision, recall, f1])
    
#     return accuracy




# # return train,val,test dataloaders using the EgoExoEMSDataset class
# def eee_get_dataloaders(args):
    
#     if(args.dataloader_params["task"] == 'classification'):
#         print("*" * 10, "=" * 10, "*" * 10)
#         print("Loading dataloader for Classification task")

#         train_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         val_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         test_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])


#         train_class_stats = train_dataset._get_class_stats()
#         print("Train class stats: ", train_class_stats)
#         # print number of keys in the dictionary
#         print("Train Number of classes: ", len(train_class_stats.keys()))

#         val_class_stats = val_dataset._get_class_stats()
#         print("val class stats: ", val_class_stats)
#         # print number of keys in the dictionary
#         print("Val Number of classes: ", len(val_class_stats.keys()))

#         # Create DataLoaders for training and validation subsets
#         train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
#         val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

#         print("train dataset size: ", len(train_dataset))
#         print("val dataset size: ", len(val_dataset))
#         print("test dataset size: ", len(test_dataset))
    
#     elif (args.dataloader_params["task"] == 'segmentation'):
#         print("*" * 10, "=" * 10, "*" * 10)
#         print("Loading dataloader for Segmentation task")
        
#         train_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         val_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         test_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         train_class_stats = train_dataset._get_class_stats()
#         print("Train class stats: ", train_class_stats)
#         # print number of keys in the dictionary
#         print("Train Number of classes: ", len(train_class_stats.keys()))

#         val_class_stats = val_dataset._get_class_stats()
#         print("val class stats: ", val_class_stats)
#         # print number of keys in the dictionary
#         print("Val Number of classes: ", len(val_class_stats.keys()))

#         test_class_stats = test_dataset._get_class_stats()
#         print("test class stats: ", test_class_stats)
#         # print number of keys in the dictionary
#         print("Test Number of classes: ", len(test_class_stats.keys()))

        
#         # Use a partial function or lambda to pass the frames_per_clip argument
#         collate_fn_with_args = partial(window_collate_fn, frames_per_clip=args.dataloader_params["observation_window"])

#         # Create DataLoaders for training and validation subsets
#         train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True, collate_fn=collate_fn_with_args)
#         test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)
#         val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)

#         print("train dataset size: ", len(train_dataset))
#         print("val dataset size: ", len(val_dataset))
#         print("test dataset size: ", len(test_dataset))

#     return train_loader, val_loader, test_loader, train_class_stats, val_class_stats




# return train,val,test dataloaders using the MIDAS dataset class
def MIDAS_get_dataloaders(args):
    
    print("*" * 10, "=" * 10, "*" * 10)
    print("Loading dataloader for Classification task")

    train_dataset = MultimodalGestureDataset(
        base_path=args.dataloader_params["base_path"],
        csv_paths=args.dataloader_params["train_trials"],
        clip_len=args.dataloader_params["observation_window"],
        step=args.dataloader_params["step"],
        include_modalities=args.dataloader_params["modalities"],
        sample_rate=args.dataloader_params["sample_rate"],

        # Allowlist patterns (fnmatch)
        modality_selections= args.dataloader_params['selections'],

        # modality_exclude={
        #     "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
        # },

        normalize=True,
    )

    val_dataset = MultimodalGestureDataset(
        base_path=args.dataloader_params["base_path"],
        csv_paths=args.dataloader_params["val_trials"],
        clip_len=args.dataloader_params["observation_window"],
        step=args.dataloader_params["step"],
        include_modalities=args.dataloader_params["modalities"],
        sample_rate=args.dataloader_params["sample_rate"],


        modality_selections= args.dataloader_params['selections'],

        # modality_exclude={
        #     "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
        # },
        normalize=True,
    )

    test_dataset = MultimodalGestureDataset(
        base_path=args.dataloader_params["base_path"],
        csv_paths=args.dataloader_params["test_trials"],
        clip_len=args.dataloader_params["observation_window"],
        step=args.dataloader_params["step"],
        include_modalities=args.dataloader_params["modalities"],
        sample_rate=args.dataloader_params["sample_rate"],

        modality_selections= args.dataloader_params['selections'],

        # modality_exclude={
        #     "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
        # },
        normalize=True,
    )

    train_class_stats = train_dataset._get_class_stats()
    print("Train class stats: ", train_class_stats)

    val_class_stats = val_dataset._get_class_stats()
    print("Val class stats: ", val_class_stats)

    test_class_stats = test_dataset._get_class_stats()
    print("Test class stats: ", test_class_stats)

    train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True,
                        collate_fn=MultimodalGestureDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False,
                        collate_fn=MultimodalGestureDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False,
                        collate_fn=MultimodalGestureDataset.collate_fn)
    
    return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats

def print_one_batch(loader):
    batch = next(iter(loader))

    # print more details about batch content
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: dtype={v.dtype}, shape={tuple(v.shape)}, "
                f"min={v.min().item():.3f}, max={v.max().item():.3f}")
        elif k == "gesture_code":
            # show a few sample codes for sanity
            print(f"{k}: {len(v)} items -> {v[:5]}{' ...' if len(v) > 5 else ''}")
        elif isinstance(v, list):
            print(f"{k}: list of {len(v)} items, first 3: {v[:3]}")
        else:
            print(f"{k}: type={type(v)}, value={v}")



# MMT Model initialization

def initialize_mmt_model(args, device):
    print("Initializing MMT model...")
    print("MMT Config: ", args.mmtransformercfg)
    model = ModularMultimodalTransformer(args.mmtransformercfg)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def build_inputs(batch, active_modalities):
    return {m: batch[m] for m in active_modalities if m in batch}


def train_mmt_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:

            inputs = build_inputs(batch, args.dataloader_params["modalities"])
            inputs = {k: v.to(device) for k,v in inputs.items()}

            logits = model(inputs)

            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)



def validate_mmt(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                inputs = build_inputs(batch, args.dataloader_params["modalities"])
                inputs = {k: v.to(device) for k,v in inputs.items()}

                logits = model(inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)

def test_mmt_model(model, test_loader, criterion, device, logger, epoch, results_dir, args):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # forward
                inputs = build_inputs(batch, args.dataloader_params["modalities"])
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                logits = model(inputs)                            # [B, C]
                labels = batch["label"].to(device, non_blocking=True)  # [B]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # predictions
                pred = torch.argmax(logits, dim=1)               # [B]

                # accumulate scalar lists
                gt.extend(batch["label"].cpu().tolist())         # extend with B items
                preds.extend(pred.cpu().tolist())                # extend with B items

                # optional: probs if you need them
                probs = torch.softmax(logits, dim=1).detach().cpu().tolist()

                # detailed per-sample records
                B = pred.shape[0]
                trial_ids      = batch.get("trial_id",      [None]*B)   # might be list[str] or tensor
                subject_ids    = batch.get("subject_id",    [None]*B)
                gesture_codes  = batch.get("gesture_code",  [None]*B)   # usually list[str]

                for i in range(B):
                    preds_detail.append({
                        "trial_id":      trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item(),
                        "subject_id":    subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item(),
                        "gesture_code":  gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
                        "pred_label":    int(pred[i].cpu().item()),
                        "logits":        logits[i].detach().cpu().tolist(),
                        "probs":         probs[i],
                    })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)
    
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])

    # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_id", "subject_id", "gesture_code", "pred_label","probs"])
        for pred in preds_detail:
            writer.writerow([pred["trial_id"], pred["subject_id"], pred["gesture_code"], pred["pred_label"], pred["probs"]])
    return results

def mmt_preprocess(batch, args, backbone, device):
    """
    Build [B, T_max, F_total] by concatenating features from the modalities
    listed in args.dataloader_params["modalities"].

    Rules for image data:
      - If "images_feat" is listed in modalities AND present in batch, use it directly.
      - Else if "images" is listed AND present, run backbone.extract_resnet on frames to get per-frame features.
      - "images" frames themselves are NOT concatenated to model inputs; they are for viz.
    """
    if not hasattr(args, "dataloader_params") or "modalities" not in args.dataloader_params:
        raise ValueError("args.dataloader_params['modalities'] not found.")

    wanted_mods = list(args.dataloader_params["modalities"])

    # Expand the effective list with image logic:
    # Prefer precomputed features if requested/present; otherwise fall back to frames->backbone.
    effective_mods = []
    for m in wanted_mods:
        if m == "images_feat":
            if "images_feat" in batch:
                effective_mods.append("images_feat")
            # if images_feat was requested but not present, we do NOT silently
            # substitute frames; keep behavior strict and explicit.
        elif m == "images":
            if "images_feat" in batch:
                # If user asked for "images" but precomputed features are already available,
                # prefer features for the model (frames are for viz). Add only once.
                if "images_feat" not in effective_mods:
                    effective_mods.append("images_feat")
            elif "images" in batch:
                # We will compute features from frames using the provided backbone.
                effective_mods.append("images")  # means: derive features from frames below
        else:
            if m in batch:
                effective_mods.append(m)

    if not effective_mods:
        raise ValueError(
            "No active modalities found in batch matching args.dataloader_params['modalities'] "
            "(nothing to concatenate)."
        )

    mod_tensors = []  # list of (name, tensor[B,T,F])
    T_max = 0
    B_ref = None

    for m in effective_mods:
        if m == "images_feat":
            x = batch["images_feat"]  # [B, T, F_img]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                raise ValueError(f"'images_feat' must be [B,T,F], got {type(x)} with shape {getattr(x,'shape',None)}")
            B, T, F = x.shape
            if B_ref is None:
                B_ref = B
            elif B != B_ref:
                raise ValueError(f"Batch size mismatch across modalities: expected {B_ref}, got {B} for 'images_feat'")

            # Respect images_mask if present (zero out invalid timesteps)
            img_mask = batch.get("images_mask", None)  # [B,T] bool
            if img_mask is not None:
                if not isinstance(img_mask, torch.Tensor) or img_mask.shape[:2] != (B, T):
                    raise ValueError(f"images_mask must be [B,T] bool, got {type(img_mask)} with shape {getattr(img_mask,'shape',None)}")
                mask = img_mask.to(device=device, dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
                x = x.to(device, non_blocking=True) * mask
            else:
                x = x.to(device, non_blocking=True)

            mod_tensors.append((m, x))
            T_max = max(T_max, T)
            continue

        if m == "images":
            # Derive features from frames using the provided backbone
            imgs = batch["images"]  # [B, T, C, H, W], float in [0,1]
            if not isinstance(imgs, torch.Tensor) or imgs.dim() != 5:
                raise ValueError(f"'images' must be [B,T,C,H,W], got {type(imgs)} with shape {getattr(imgs,'shape',None)}")

            B, T, C, H, W = imgs.shape
            if B_ref is None:
                B_ref = B
            elif B != B_ref:
                raise ValueError(f"Batch size mismatch across modalities: expected {B_ref}, got {B} for 'images'")

            imgs = imgs.to(device, non_blocking=True)

            # Backbone is user-provided; expected to output per-frame features.
            # We flatten B,T -> (B*T) for feature extraction, then reshape back.
            imgs_btchw = imgs.view(B * T, C, H, W)

            with torch.no_grad():
                feats = backbone.extract_resnet(imgs_btchw)  # expected [B*T, D] or [B*T, D, 1, 1]

            if feats.dim() == 4:
                # pool to [N, D]
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
            elif feats.dim() == 2:
                pass
            else:
                feats = feats.view(feats.size(0), -1)

            feats = feats.view(B, T, -1)  # [B, T, Fimg]

            # Respect images_mask if present (zero out invalid timesteps)
            img_mask = batch.get("images_mask", None)  # [B,T] bool
            if img_mask is not None:
                if not isinstance(img_mask, torch.Tensor) or img_mask.shape != (B, T):
                    raise ValueError(f"images_mask must be [B,T] bool, got {type(img_mask)} with shape {getattr(img_mask,'shape',None)}")
                mask = img_mask.to(device=device, dtype=feats.dtype).unsqueeze(-1)  # [B,T,1]
                feats = feats * mask

            mod_tensors.append(("images_feat", feats))  # store as images_feat going forward
            T_max = max(T_max, T)
            continue

        # ---- Non-image modalities: expected [B, T, F] ----
        x = batch[m]
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError(f"Expected [B,T,F] for modality '{m}', got {type(x)} with shape {getattr(x,'shape',None)}")

        B, T, F = x.shape
        if B_ref is None:
            B_ref = B
        elif B != B_ref:
            raise ValueError(f"Batch size mismatch across modalities: got {B_ref} and {B} for '{m}'")

        T_max = max(T_max, T)
        mod_tensors.append((m, x.to(device, non_blocking=True)))

    # Right-pad each [B,T,F] to T_max along time, then concat on feature dim
    padded = []
    for _, x in mod_tensors:
        B, T, F = x.shape
        if T < T_max:
            pad = x.new_zeros((B, T_max - T, F))
            x = torch.cat([x, pad], dim=1)
        padded.append(x)

    X = torch.cat(padded, dim=-1)  # [B, T_max, sum(F)]
    return X



def get_feature_dim(loader, args, model, device):
    batch = next(iter(loader))
    preprocessed_inputs = mmt_preprocess(batch, args, model, device)
    return preprocessed_inputs.size(-1)


def train_transtcn_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:

            preprocessed_inputs = mmt_preprocess(batch, args, model, device)  # [B, T, F_total]
            logits = model(preprocessed_inputs)

            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)


def validate_transtcn(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                logits = model(preprocessed_inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)

def test_transtcn_model(model, test_loader, criterion, device, logger, epoch, results_dir, args):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # forward
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                logits = model(preprocessed_inputs)                            # [B, C]
                labels = batch["label"].to(device, non_blocking=True)  # [B]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # predictions
                pred = torch.argmax(logits, dim=1)               # [B]

                # accumulate scalar lists
                gt.extend(batch["label"].cpu().tolist())         # extend with B items
                preds.extend(pred.cpu().tolist())                # extend with B items

                # optional: probs if you need them
                probs = torch.softmax(logits, dim=1).detach().cpu().tolist()

                # detailed per-sample records
                B = pred.shape[0]
                trial_ids      = batch.get("trial_id",      [None]*B)   # might be list[str] or tensor
                subject_ids    = batch.get("subject_id",    [None]*B)
                gesture_codes  = batch.get("gesture_code",  [None]*B)   # usually list[str]

                for i in range(B):
                    preds_detail.append({
                        "trial_id":      trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item(),
                        "subject_id":    subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item(),
                        "gesture_code":  gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
                        "pred_label":    int(pred[i].cpu().item()),
                        "logits":        logits[i].detach().cpu().tolist(),
                        "probs":         probs[i],
                    })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)

    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])  

        # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_id", "subject_id", "gesture_code", "pred_label","probs"])
        for pred in preds_detail:
            writer.writerow([pred["trial_id"], pred["subject_id"], pred["gesture_code"], pred["pred_label"], pred["probs"]])
    return results



def initialize_tcn_model(args, input_dim, device, num_classes):
    print("Initializing TCN model...")
    print("TCN Config: input_dim=", input_dim, ", num_classes=", num_classes)

    model = TCNClassifier(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def initialize_multi_tcn_model(args, device, feature_dim):
    print("Initializing Multi-TCN model...")
    print("Multi-TCN Config: ", args.multi_tcncfg)
    model = MultiBranchTCNClassifier(modality_input_dims=args.tc, num_classes=args.num_classes, **args.multi_tcncfg)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion



def train_TCN_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:

            preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
            logits = model(preprocessed_inputs)

            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)


def validate_TCN(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                logits = model(preprocessed_inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)

def test_TCN_model(model, test_loader, criterion, device, logger, epoch, results_dir, args):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # forward
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                logits = model(preprocessed_inputs)                            # [B, C]
                labels = batch["label"].to(device, non_blocking=True)  # [B]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # predictions
                pred = torch.argmax(logits, dim=1)               # [B]

                # accumulate scalar lists
                gt.extend(batch["label"].cpu().tolist())         # extend with B items
                preds.extend(pred.cpu().tolist())                # extend with B items

                # optional: probs if you need them
                probs = torch.softmax(logits, dim=1).detach().cpu().tolist()

                # detailed per-sample records
                B = pred.shape[0]
                trial_ids      = batch.get("trial_id",      [None]*B)   # might be list[str] or tensor
                subject_ids    = batch.get("subject_id",    [None]*B)
                gesture_codes  = batch.get("gesture_code",  [None]*B)   # usually list[str]

                for i in range(B):
                    preds_detail.append({
                        "trial_id":      trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item(),
                        "subject_id":    subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item(),
                        "gesture_code":  gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
                        "pred_label":    int(pred[i].cpu().item()),
                        "logits":        logits[i].detach().cpu().tolist(),
                        "probs":         probs[i],
                    })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)

    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])  

        # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_id", "subject_id", "gesture_code", "pred_label","probs"])
        for pred in preds_detail:
            writer.writerow([pred["trial_id"], pred["subject_id"], pred["gesture_code"], pred["pred_label"], pred["probs"]])
    return results


# =========================
# MS-TCN++ utilities
# =========================

IGNORE_INDEX = -100  # make sure your dataloader uses this for pad frames if you have variable-length sequences

def initialize_mstcn_model(args, input_dim, device, num_classes):
    print("Initializing MS-TCN++ model...")
    print("MSTCN++ Config: input_dim=", input_dim, ", num_classes=", num_classes)

    model = MS_TCN2(
        num_layers_PG=args.mstcn_model_params['num_layers_PG'],
        num_layers_R=args.mstcn_model_params['num_layers_R'],
        num_R=args.mstcn_model_params['num_R'],
        num_f_maps=args.mstcn_model_params['num_f_maps'],
        dim=input_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_params['lr'],
        weight_decay=args.learning_params['weight_decay']
    )

    # Loss fns + weights
    loss_fns = {
        "ce": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
        "mse": nn.MSELoss(reduction="none"),
        # args.learning_params is a dict in your code – use .get(...)
        "lambda_smooth": args.learning_params.get("lambda_smooth", 0.15),
        "num_classes": num_classes
    }

    return model, optimizer, loss_fns


@torch.no_grad()
def _masked_accuracy_from_logits(logits_BCT, target_BT, mask_B1T):
    """
    logits_BCT: (B, C, T)
    target_BT:  (B, T) long
    mask_B1T:   (B, 1, T) float {0,1}
    """
    # valid positions exclude IGNORE_INDEX and respect mask
    valid = (mask_B1T.squeeze(1) > 0.5) & (target_BT != IGNORE_INDEX)
    pred = logits_BCT.argmax(dim=1)  # (B, T)
    correct = (pred.eq(target_BT) & valid).sum().item()
    total = valid.sum().item()
    return (correct / total) if total > 0 else 0.0


def _as_stage_list(predictions):
    """
    Normalize model output into a list of (B,C,T).
    MS-TCN++ may return a list of tensors or a stacked tensor (S,B,C,T).
    """
    if isinstance(predictions, (list, tuple)):
        return list(predictions)
    # assume (S,B,C,T)
    return [predictions[s] for s in range(predictions.shape[0])]


def mstcn_compute_loss(predictions, target_BT, mask_B1T, loss_fns):
    """
    predictions: list[(B,C,T)] or (S,B,C,T)
    target_BT:   (B,T) long with class indices; pads should be IGNORE_INDEX
    mask_B1T:    (B,1,T) float {0,1} (1 for valid frames)
    """
    stages = _as_stage_list(predictions)
    ce = loss_fns["ce"]
    mse = loss_fns["mse"]
    lam = float(loss_fns.get("lambda_smooth", 0.15))

    total_loss = 0.0
    for p in stages:  # p: (B,C,T)
        # -------- Cross-Entropy per stage --------
        total_loss += ce(p, target_BT)  # CE expects (N,C,*) + target (N,*)

        # -------- Temporal smoothing on log-probs --------
        T = p.size(-1)
        if lam > 0.0 and T > 1:
            logp_t   = F.log_softmax(p[:, :, 1:],   dim=1)          # (B,C,T-1)
            logp_tm1 = F.log_softmax(p.detach()[:, :, :-1], dim=1)  # (B,C,T-1), stop-grad
            per_elem = mse(logp_t, logp_tm1)                        # (B,C,T-1)
            per_elem = torch.clamp(per_elem, min=0.0, max=16.0)
            m = mask_B1T[:, :, 1:]                                  # (B,1,T-1)
            smoothed = (per_elem * m).mean()
            total_loss += lam * smoothed

    return total_loss


def train_mstcn_one_epoch(model, train_loader, loss_fns, optimizer, device, logger, args):
    """
    Expects each batch dict to have:
      - 'label' : (B,) or (B,T) long
      - optionally 'mask': (B,1,T) float {0,1}; if missing, we create all-ones
    And your mmt_preprocess returns: (B, T, F_total). We convert to (B, F_total, T).
    """
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches = 0

    for i, batch in enumerate(train_loader):
        try:
            # ----- Inputs -----
            x_BTF = mmt_preprocess(batch, args, None, device)  # (B, T, F_total)
            if x_BTF.dim() != 3:
                raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
            x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B, C=F_total, T)

            # ----- Targets -----
            labels = batch['label'].to(device)                 # (B,) or (B,T)
            B, C, T = x_BCT.shape
            
            # skip batches with time length < 30
            if T < 30:
                continue

            if labels.dim() == 1:
                # Expand single label per sequence across time to match MS-TCN++ interface
                labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
            elif labels.dim() == 2:
                if labels.size(1) != T:
                    raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
            else:
                raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

            # ----- Mask (optional) -----
            if 'mask' in batch:
                mask = batch['mask'].to(device)                # (B,1,T) preferred
                # be lenient: accept (B,T) and upgrade
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                if mask.size(-1) != T:
                    raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
            else:
                mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)
            
            # print(f"Batch {i}: x_BCT shape={x_BCT.shape}, labels shape={labels.shape}, mask shape={mask.shape}")
            # print(f"Labels: {labels}")
            # ----- Forward / Backward -----
            optimizer.zero_grad()
            preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
            loss = mstcn_compute_loss(preds, labels, mask, loss_fns)
            loss.backward()
            optimizer.step()

            # ----- Metrics from final stage -----
            stages = _as_stage_list(preds)
            final_logits = stages[-1]                           # (B,C,T)
            acc = _masked_accuracy_from_logits(final_logits, labels, mask)

            running_loss += float(loss.item())
            running_acc  += float(acc)
            n_batches += 1

            if i % 10 == 0:
                print(f"Batch {i}: loss={loss.item():.4f}, acc={acc:.4f}")

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    if n_batches == 0:
        return 0.0

    epoch_loss = running_loss / n_batches
    epoch_acc  = running_acc  / n_batches
    if logger is not None:
        try:
            logger.info(f"Train epoch: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        except Exception:
            pass
    return epoch_loss
def validate_mstcn(model, val_loader, loss_fns, device, logger, args):
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                # ----- Inputs -----
                x_BTF = mmt_preprocess(batch, args, None, device)  # (B, T, F_total)
                if x_BTF.dim() != 3:
                    raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
                x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B, C=F_total, T)

                # ----- Targets -----
                labels = batch['label'].to(device)                 # (B,) or (B,T)
                B, C, T = x_BCT.shape

                # skip batches with time length < 30
                if T < 30:
                    continue

                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
                elif labels.dim() == 2:
                    if labels.size(1) != T:
                        raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
                else:
                    raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

                # ----- Mask (optional) -----
                if 'mask' in batch:
                    mask = batch['mask'].to(device)                # (B,1,T) preferred
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1)
                    if mask.size(-1) != T:
                        raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
                else:
                    mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)

                # ----- Forward & loss -----
                preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
                loss  = mstcn_compute_loss(preds, labels, mask, loss_fns)

                # ----- Metrics from final stage -----
                stages = _as_stage_list(preds)
                final_logits = stages[-1]                           # (B,C,T)
                acc = _masked_accuracy_from_logits(final_logits, labels, mask)

                running_loss += float(loss.item())
                running_acc  += float(acc)
                n_batches    += 1

                if i % 10 == 0:
                    print(f"[VAL] Batch {i}: loss={loss.item():.4f}, acc={acc:.4f}")

            except Exception as e:
                print(f"[VAL] Error in batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    if n_batches == 0:
        if logger is not None:
            try:
                logger.warning("Validation had zero successful batches.")
            except Exception:
                pass
        return {"loss": 0.0, "acc": 0.0}

    val_loss = running_loss / n_batches
    val_acc  = running_acc  / n_batches

    if logger is not None:
        try:
            logger.info(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")
        except Exception:
            pass

    return {"loss": val_loss, "acc": val_acc}



import os
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def test_MSTCN_model(model, test_loader, loss_fns, device, logger, epoch, results_dir, args):
    """
    Frame-wise test loop for MS-TCN++.
    - Computes average loss over batches using mstcn_compute_loss
    - Aggregates frame-level predictions vs. labels (mask + IGNORE_INDEX respected)
    - Reports macro precision/recall/F1 on valid frames
    - Saves metrics.csv (appends) and per-sample predictions as npy files

    Expects each batch to contain:
      - 'label': (B,) or (B,T) long (if (B,), we expand to (B,T))
      - optional 'mask': (B,1,T) float in {0,1}; if absent, uses all-ones
      - optional metadata: 'trial_id', 'subject_id', 'gesture_code'
    """
    os.makedirs(results_dir, exist_ok=True)

    model.eval()
    running_loss = 0.0
    n_batches    = 0

    # For global metrics (flatten after masking)
    all_gt_frames   = []
    all_pred_frames = []

    # Optional: keep a compact per-sample artifact (npy) rather than huge CSV rows
    per_sample_records = []  # a small summary per sample

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # ----- Inputs -----
                x_BTF = mmt_preprocess(batch, args, None, device)  # (B,T,F)
                if x_BTF.dim() != 3:
                    raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
                x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B,C=F,T)
                B, C, T = x_BCT.shape

                # skip batches with time length < 30
                if T < 30:
                    continue

                # ----- Targets -----
                labels = batch['label'].to(device)                 # (B,) or (B,T)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
                elif labels.dim() == 2 and labels.size(1) != T:
                    raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
                elif labels.dim() != 2:
                    raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

                # ----- Mask (optional) -----
                if 'mask' in batch:
                    mask = batch['mask'].to(device)                # (B,1,T) preferred
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1)
                    if mask.size(-1) != T:
                        raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
                else:
                    mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)

                # ----- Forward & Loss -----
                preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
                loss  = mstcn_compute_loss(preds, labels, mask, loss_fns)
                running_loss += float(loss.item())
                n_batches    += 1

                # ----- Final-stage logits → predictions -----
                stages = _as_stage_list(preds)
                final_logits = stages[-1]                           # (B,C,T)
                pred_BT = final_logits.argmax(dim=1)               # (B,T)

                # ----- Collect masked frames for global metrics -----
                # valid = mask & label != IGNORE_INDEX
                valid = (mask.squeeze(1) > 0.5) & (labels != IGNORE_INDEX)
                gt_flat   = labels[valid].detach().cpu().numpy()
                pred_flat = pred_BT[valid].detach().cpu().numpy()
                if gt_flat.size > 0:
                    all_gt_frames.append(gt_flat)
                    all_pred_frames.append(pred_flat)

                # ----- Optional: save per-sample predictions/probs -----
                # (safer to store per-sample npy files than giant CSV rows)
                probs_BCT = torch.softmax(final_logits, dim=1).detach().cpu().numpy()  # (B,C,T)
                trial_ids     = batch.get("trial_id",    [None]*B)
                subject_ids   = batch.get("subject_id",  [None]*B)
                gesture_codes = batch.get("gesture_code",[None]*B)

                # Make iterables uniform
                if torch.is_tensor(trial_ids):     trial_ids     = trial_ids.cpu().tolist()
                if torch.is_tensor(subject_ids):   subject_ids   = subject_ids.cpu().tolist()
                if torch.is_tensor(gesture_codes): gesture_codes = gesture_codes.cpu().tolist()

                for b in range(B):
                    # derive a filename stem
                    stem = str(trial_ids[b]) if trial_ids[b] is not None else f"sample_{i}_{b}"
                    # save predictions and (optionally) probabilities
                    np.save(os.path.join(results_dir, f"{stem}_pred.npy"),
                            pred_BT[b].detach().cpu().numpy())
                    # probs can be large; save only if you want them
                    if getattr(args, "save_probs", False):
                        np.save(os.path.join(results_dir, f"{stem}_probs.npy"),
                                probs_BCT[b])

                    # short summary for CSV
                    per_sample_records.append({
                        "trial_id":     trial_ids[b],
                        "subject_id":   subject_ids[b],
                        "gesture_code": gesture_codes[b],
                        "T":            int(T),
                        "valid_frames": int(valid[b].sum().item()),
                    })

                if i % 10 == 0:
                    logger.log({"test_batch_loss": loss.item()})

            except Exception as e:
                print(f"[TEST] Error in batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    # ----- Aggregate metrics -----
    if n_batches == 0:
        results = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "epoch": epoch}
        logger.log(results)
        return results

    avg_loss = running_loss / n_batches

    if len(all_gt_frames) == 0:
        # No valid frames collected
        accuracy = precision = recall = f1 = 0.0
    else:
        gt_all   = np.concatenate(all_gt_frames, axis=0)
        pred_all = np.concatenate(all_pred_frames, axis=0)
        accuracy  = float((gt_all == pred_all).mean())
        precision = float(precision_score(gt_all, pred_all, average='macro', zero_division=0))
        recall    = float(recall_score(gt_all, pred_all, average='macro', zero_division=0))
        f1        = float(f1_score(gt_all, pred_all, average='macro', zero_division=0))

    results = {
        "epoch":     epoch,
        "loss":      avg_loss,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }
    logger.log(results)

    # ----- Save metrics to CSV (append) -----
    metrics_path = os.path.join(results_dir, "metrics.csv")
    fresh_file = not os.path.exists(metrics_path)
    with open(metrics_path, mode='a', newline='') as f:
        w = csv.writer(f)
        if fresh_file:
            w.writerow(["epoch", "loss", "precision", "recall", "f1", "accuracy"])
        w.writerow([epoch, avg_loss, precision, recall, f1, accuracy])

    # ----- Save a compact per-sample summary CSV -----
    # (detailed framewise preds are saved as .npy per sample above)
    samples_path = os.path.join(results_dir, "samples.csv")
    fresh_file = not os.path.exists(samples_path)
    with open(samples_path, mode='a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "subject_id", "gesture_code", "T", "valid_frames"])
        if fresh_file:
            w.writeheader()
        for rec in per_sample_records:
            w.writerow(rec)

    return results
