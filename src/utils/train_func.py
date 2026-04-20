import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from src.utils.loss import get_station_from_grid
from src.utils import utils
import copy
import torch.nn.functional as F

def load_checkpoint(model, checkpoint_path, device):
    """Load the model checkpoint from file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_dict'])
    return model

def to_float(x, device):
    if isinstance(x,list):
        list_x = []
        for x_i in x:
            x_i = x_i.to(device).float()
            list_x.append(x_i)
        x = list_x
    else:
        x = x.to(device).float()
        
    return x

def valid_func(model, valid_dataloader, early_stopping, loss_func, config, device):
    model.eval()
    with torch.no_grad():
        valid_epoch_loss = []
        for data in tqdm(valid_dataloader):
            input_data, lead_time, y_valid = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device)
            h = data.get("h", None)
            if h is not None:
                h = h.to(device)
                # h = h[:, :-1, :, :]
                y_ = model([input_data, lead_time, h])
            else:
                y_ = model([input_data, lead_time])
            
            y_ = get_station_from_grid(y_, y_valid, config) # (batch_size, num_station, 1)
            y_valid = y_valid[:,:,0] # (batch_size, num_station)
            loss = loss_func(y_.squeeze(), y_valid.squeeze())
            valid_epoch_loss.append(loss.item())
            del y_valid, y_, input_data, lead_time
            if h is not None:
                del h
        valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

    return valid_epoch_loss

import gc


def train_func(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, config, device):
    model.to(device)
    checkpoint_path = f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt"
    start_epoch = 0
    best_valid_loss = float('inf')
    best_model_state = None
    results = {'train_losses': [], 'valid_losses': [], 'learning_rates': []}

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}, loading model...")
        model = load_checkpoint(model, checkpoint_path, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and config.LRS.USE_LRS:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float('inf'))
        results = checkpoint.get("results", results)
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_valid_loss:.4f}")

    # Scheduler setup
    scheduler = None
    if config.LRS.USE_LRS:
        if config.LRS.NAME == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.LRS.COSINE_T_MAX,
                eta_min=config.LRS.COSINE_ETA_MIN
            )
        elif config.LRS.NAME == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config.LRS.PLATEAU_MODE,
                factor=config.LRS.PLATEAU_FACTOR,
                patience=config.LRS.PLATEAU_PATIENCE,
                min_lr=config.LRS.PLATEAU_MIN_LR,
                verbose=config.LRS.PLATEAU_VERBOSE
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.LRS.NAME}. Choose 'CosineAnnealingLR' or 'ReduceLROnPlateau'")

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    # Training loop
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        epoch_loss = []
        if not early_stopping.early_stop:
            model.train()
            
            for i, data in enumerate(tqdm(train_dataloader)):
                
                optimizer.zero_grad()
                input_data, lead_time, y_train = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device)
                
                h = data.get("h", None)
                
                if h is not None:
                    h = h.to(device)
                    # print(h.shape)
                    # h = h[:, :-1, :, :]
                    y_ = model([input_data, lead_time, h])
                else:
                    y_ = model([input_data, lead_time])
                y_ = get_station_from_grid(y_, y_train, config)
                
                y_train = y_train[:,:,0]
                
                loss = loss_func(y_.squeeze(), y_train.squeeze())
                loss.backward()
                
                
                optimizer.step()
                
                epoch_loss.append(loss.item())
                # Free temps ASAP
                del y_train, y_, input_data, lead_time
                if h is not None:
                    del h
            
            train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            valid_epoch_loss =  valid_func(model, valid_dataloader, early_stopping, loss_func, config, device)

            # Scheduler step
            if config.LRS.USE_LRS and scheduler is not None:
                if config.LRS.NAME == 'CosineAnnealingLR':
                    scheduler.step()
                elif config.LRS.NAME == 'ReduceLROnPlateau':
                    scheduler.step(valid_epoch_loss)

            early_stopping(valid_epoch_loss, model)
            
            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                best_model_state = {k: v.to(device) for k, v in model.state_dict().items()}
                # Save checkpoint
            torch.save({
                    "model_dict": {k: v.to(device) for k, v in model.state_dict().items()},
                    "best_dict": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                    "results": results
                }, checkpoint_path)    

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{config.TRAIN.EPOCHS} | Train Loss: {train_epoch_loss:.4f} | "
                  f"Valid Loss: {valid_epoch_loss:.4f} | LR: {current_lr:.6f}")
            
            results['train_losses'].append(train_epoch_loss)
            results['valid_losses'].append(valid_epoch_loss)
            results['learning_rates'].append(current_lr)
            
            
            
            if config.WANDB.STATUS:
                wandb.log({
                    "loss/train_loss": train_epoch_loss,
                    "loss/valid_loss": valid_epoch_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    f"{config.LRS.NAME}/epoch": epoch if config.LRS.USE_LRS else "NoScheduler/epoch"
                })
       
        torch.cuda.empty_cache()
        gc.collect()             
            

    return {
        'best_model_state': best_model_state,
        'final_train_loss': train_epoch_loss,
        'best_valid_loss': best_valid_loss,
        'train_losses': results['train_losses'],
        'valid_losses': results['valid_losses'],
        'learning_rates': results['learning_rates'],
        'scheduler_type': config.LRS.NAME if config.LRS.USE_LRS else 'None'
    }
    
