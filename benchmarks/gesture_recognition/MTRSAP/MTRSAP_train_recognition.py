from utils.utils import *
from scripts.config import DefaultArgsNamespace
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from datetime import datetime

import os
import time

import argparse
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")


torch.manual_seed(0)

if __name__ == "__main__":

    # get cmd line args
    parser = argparse.ArgumentParser(description="Training script for recognition")
    parser.add_argument('--job_id', type=str, help='SLURM job ID')
    
    cmd_args = parser.parse_args()
    
    print(f"Job ID: {cmd_args.job_id}")

    if cmd_args.job_id is None:
        # generate a random job id
        cmd_args.job_id = str(int(time.time()))

    args = DefaultArgsNamespace()

    wandb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="MIDAS Gesture Recognition",
        group="Gesture Recognition",
        mode="disabled",
        name="train",
        notes="",
        config={
        "args": args,
        }
    )

    keysteps = args.dataloader_params['keysteps']
    out_classes = len(keysteps)

    print(f"Keysteps: {keysteps}")

    modality = args.dataloader_params['modalities']
    print(f"Modality: {modality}")

    # train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats = MIDAS_get_dataloaders(args)
    args.dataloader_params['train_class_stats'] = train_class_stats
    args.dataloader_params['val_class_stats'] = val_class_stats

    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    print_one_batch(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_dim = get_feature_dim(train_loader, args, device)
    print(f"Feature dimension: {feature_dim}")

    args.transformer_params['input_dim'] = feature_dim
    args.transformer_params['output_dim'] = out_classes


    model, optimizer, criterion = init_model(args, device)

    scheduler = StepLR(optimizer, step_size=args.learning_params["lr_drop"], gamma=0.1)  # adjust parameters as needed

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f'./results/job_{cmd_args.job_id}'
    chkpoint_dir = f'./checkpoints/job_{cmd_args.job_id}'

    # print(f"Model: {model}")

    # create results directory if not exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create checkpoint directory if not exists
    if not os.path.exists(chkpoint_dir):
        os.makedirs(chkpoint_dir)
    
    min_val_loss = float('inf')

    # # Train the model
    for epoch in range(1, args.learning_params["epochs"] + 1):
        print("*"*10, "="*10, "*"*10)
        print(f"Epoch: {epoch}")
        train_loss = train_transtcn_one_epoch(model, train_loader, criterion, optimizer, device, wandb_logger,args)
        wandb_logger.log({"avg_train_loss": train_loss, "epoch": epoch})
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")

        val_loss = validate_transtcn(model, val_loader, criterion, device, wandb_logger, args)
        # wandb_logger.log({"avg_val_loss": val_loss, "epoch": epoch})
        print(f"Epoch: {epoch}, Val Loss: {val_loss}")

        # # save checkpoints if validation loss is minimum 
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{chkpoint_dir}/val_best_model.pt')


        scheduler.step()

        results = test_transtcn_model(model, test_loader, criterion, device, wandb_logger, epoch, results_dir, args)
        print(f"Results: {results}")
        
        print("*"*10, "="*10, "*"*10)
        
