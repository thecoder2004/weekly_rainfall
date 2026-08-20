import argparse
import os
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
from src.model.baseline.unet import UNet
from src.utils.get_option import get_option
from src.model import models
from src.model.baseline import cnn_lstm, conv_lstm
from src.utils import utils, get_scaler, train_func, test_func
from src.utils.loss import *
from src.utils.get_session_name import get_session_name
from src.utils.dataloader import CustomDataset3
def get_device():
    if torch.cuda.is_available():
        print("Device: GPU")
        return torch.device("cuda")
    else:
        print("Device: CPU")
        return torch.device("cpu")

def get_loss_function(config):
    loss_name = config.LOSS.NAME.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mae":
        return nn.L1Loss(reduction='mean')
    elif loss_name == "huberloss":
        return nn.HuberLoss(delta=1.0, reduction='mean')
    elif loss_name == "expweightedloss":
        return ExpMagnitudeWeightedMAELoss(config.LOSS.k)
    elif loss_name == "weightedmse":
        return WeightedMSELoss(weight_func=config.LOSS.WEIGHT_FUNC)
    elif loss_name == "magnitudeweight":
        return MagnitudeWeightedHuberLoss(delta=config.LOSS.DELTA)
    elif loss_name == "msle":
        return MSLELoss()
    elif loss_name == "logmagnitudeweight":
        return LogMagnitudeWeightedHuberLoss(delta = config.LOSS.DELTA, alpha = config.LOSS.ALPHA)
    elif loss_name =="quantile":
        return QuantileLoss(quantile=0.7, alpha=0.4)
    elif loss_name == "weightedthresholdmse":
        return WeightedThresholdMSE(
            high_weight=config.LOSS.HIGH_WEIGHT,
            low_weight=config.LOSS.LOW_WEIGHT,
            threshold=config.LOSS.GROUNDTRUTH_THRESHOLD,
        )
    elif loss_name == "combineweightloss":
        return CombinedWeightedLoss(gamma=2.0, beta=0.01)
    else:
        raise ValueError(f"Invalid loss function name: {config.LOSS.NAME}")

def get_model(config, device):
    name = config.MODEL.NAME.lower()
    model_map = {
        "vifos-conv3d": models.VIFOS_CONV3D,
        "vifos-spatial-temporal-extractor": models.VIFOS_SPATIAL_TEMPORAL_EXTRACTOR,
        "cnn-lstm": cnn_lstm.CNN,
        "unet": UNet,
        "quantitle": UNet,
    }
    if name not in model_map:
        raise ValueError(f"Wrong model name: {config.MODEL.NAME}")
    model = model_map[name](config).to(device)
    return model

def create_checkpoint_dir(path):
    if not os.path.exists(path):
        print(f"Creating directory {path} ...")
        os.makedirs(path)


def init_wandb(config):
    if config.WANDB.STATUS:
        wandb.login(key='960dec1c23ffe487b2ecb98ffc097cf118d94c19')
        wandb.init(
            entity="aiotlab",
            project="SubSeasonalForecasting",
            group=config.WANDB.GROUP_NAME,
            name=config.WANDB.SESSION_NAME,
            config=config,
        )

def main():
    args, config = get_option()
    
    config.WANDB.SESSION_NAME = get_session_name(config)
    device = get_device()
    config.DEVICE = device
    
    utils.seed_everything(config.MODEL.SEED)
    
    loss_func = get_loss_function(config)
        
    # Preprocess data
    print("*************** Get scaler ***************")
    input_scaler, esp_scaler, output_scaler = get_scaler.get_scaler(config)
    
    print("*************** Init dataset ***************")
    

    checkpoint_dir = f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/"
    create_checkpoint_dir(checkpoint_dir)

    early_stopping = utils.EarlyStopping(
        patience=config.EARLY_STOPPING.PATIANCE,
        verbose=True,
        delta=config.EARLY_STOPPING.DELTA,
        path=os.path.join(checkpoint_dir, f"{config.WANDB.SESSION_NAME}.pt")
    )
    
    ## Init model 
    print("*************** Init model ***************")
    model = get_model(config, device)

    
    
    
        
        
    ## Set optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if  config.OPTIMIZER.NAME == "adam":
        optimizer = torch.optim.Adam(
        trainable_params, lr=config.OPTIMIZER.LR, weight_decay=config.OPTIMIZER.L2_COEF
        )
    elif config.OPTIMIZER.NAME == "adamw":
        optimizer = torch.optim.AdamW(
        trainable_params, lr=config.OPTIMIZER.LR, weight_decay=config.OPTIMIZER.L2_COEF
        )
    else:
        raise("Error: Wrong optimizer name")
    # config.WANDB.STATUS = True
    
    # Init wandb session
    init_wandb(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    train_dataset = CustomDataset3(mode='train', config=config, ecmwf_scaler=input_scaler, esp_scaler=esp_scaler, output_scaler= output_scaler, shuffle=True)
    valid_dataset = CustomDataset3(mode='valid', config=config, ecmwf_scaler=input_scaler, esp_scaler=esp_scaler, output_scaler= output_scaler)
    test_dataset = CustomDataset3(mode='test', config=config, ecmwf_scaler=input_scaler, esp_scaler=esp_scaler, output_scaler= output_scaler)
    if config.MODEL.NAME.lower() == "quantitle":
        test_func.test_func_quantile( test_dataset, loss_func, config, esp_scaler, output_scaler, device)
        return
    results = train_func.train_func(model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, config, device)
    
    print(f"Final Train Loss: {results['final_train_loss']:.4f}")
    
    utils.load_model(model, f"saved_checkpoints/{config.WANDB.GROUP_NAME}/checkpoint/{config.WANDB.SESSION_NAME}.pt")
    test_func.test_func(model, test_dataset, loss_func, config, esp_scaler, output_scaler, device)
    

if __name__ == "__main__":
    main()