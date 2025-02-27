#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# multinode training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import contextlib
import os
import glob
import gc

from dataset_classes.lazy_dataset import LazyDataset
from src.setup_model import setup_model
from src.setup_config_variables import setup_config_variables
from src.animate_loader import animate_loader
from src.setup_transforms import setup_transforms

def ddp_setup(local_rank):
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")

def prepare_dataloader(training_dir, test_dir, batch_size, augment_data):
    transform, target_transform = setup_transforms(augment_data)
    train_dataset = LazyDataset(training_dir, transform=transform, target_transform=target_transform)
    test_dataset = LazyDataset(test_dir, transform=transform, target_transform=target_transform)

    # load the data, can implement batching etc. here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=5, sampler=DistributedSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=5, sampler=DistributedSampler(test_dataset))

    return train_loader, test_loader

def warm_start(model, optimizer, local_rank, last_model_dir):
        """
        Load the last model and optimizer state from the last saved model, returns starting epoch and best loss
        """
        loc = f"cuda:{local_rank}"
        last_model_file = glob.glob(f"{last_model_dir}/*")[-1]
        load_dict = torch.load(last_model_file, map_location=loc)
        model.load_state_dict(load_dict["model_state_dict"])
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        epoch = load_dict["epoch"]
        best_test_loss = load_dict["best_test_loss"]
        print(f"Resuming training from epoch {epoch} with best test loss {best_test_loss} from file: {last_model_file}")
        return epoch + 1, best_test_loss


def clean_models(directory):
    '''
    Delete the models in a given directory
    '''
    for model_file in glob.glob(directory + "/*.pth"):
        os.remove(model_file)

def save_model(directory, model, optimizer, epoch, loss, best_test_loss, *, clean_old=False):
    '''
    Saves the model in the given location.
    If clean_old, cleans the old model files in the directory
    '''
    state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'best_test_loss': best_test_loss
    }
    
    if clean_old:
        clean_models(directory)
    
    loss_steering_angle_rmse = 180 * (loss ** 0.5)
    torch.save(state, f"{directory}/e{epoch + 1}_L{loss_steering_angle_rmse:.2f}.pth")


def run_epoch(model, loader, optimizer, scaler, criterion, mode, portion_of_dataset, local_rank, global_rank, model_type):
        epoch_loss = 0
        context = torch.no_grad() if mode == "TEST" else contextlib.nullcontext()

        # report every 25% progress
        total_batches = len(loader)
        training_batches = portion_of_dataset * total_batches
        
        
        progress_thresholds = {int(total_batches * 0.25), int(total_batches * 0.50), int(total_batches * 0.75)}
        with context:
            for i, (combined_images, steering_angle) in enumerate(loader):
                if i > training_batches:
                    break

                if model_type == "Resnet":
                    b, t, c, w, h = combined_images.shape
                    combined_images = combined_images.view(b * t, c, w, h)
                elif model_type != "ResnetRNN":
                    combined_images = combined_images.permute(0, 2, 1, 3, 4)

                if mode != "TEST":
                    optimizer.zero_grad()

                combined_images = combined_images.to(local_rank)
                steering_angle = steering_angle.to(local_rank)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(combined_images).squeeze(1)
                    if model_type == "Resnet":
                        loss = criterion(outputs, steering_angle.view(-1))
                    else:
                        last_angles = steering_angle[:, -1]
                        loss = criterion(outputs, last_angles)

                if mode != "TEST":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                epoch_loss += loss.item()
                
                # Report progress if we have hit one of the thresholds
                if (i + 1) in progress_thresholds:
                    progress = round(100 * (i + 1) / total_batches)
                    print(f"[GPU{global_rank}]{mode}, Epoch progress: {progress}% completed.")
                
                del combined_images, steering_angle, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Average training loss for the epoch
            epoch_loss /= len(loader)
            return epoch_loss

def main():
    config = setup_config_variables()
    ddp_setup(config["LOCAL_RANK"])
    model = setup_model(config["MODEL_TYPE"], config["MODEL_DIMS"], config["PRETRAINED"], config["LOCAL_RANK"])
    model = DDP(model, device_ids=[config["LOCAL_RANK"]])

    starting_epoch = 0
    best_test_loss = np.inf
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])

    if config["CLEAN_OLD_MODELS"]:
        clean_models(config["MODEL_SAVE_DIR"])

    last_model_dir = f"{config['MODEL_SAVE_DIR']}/last"
    best_model_dir = f"{config['MODEL_SAVE_DIR']}/best"

    if config["WARM_START"]:
        starting_epoch, best_test_loss = warm_start(model, optimizer, config["LOCAL_RANK"], last_model_dir)
    
    train_loader, test_loader = prepare_dataloader(training_dir=config["TRAINING_DIR"], test_dir=config["TEST_DIR"], batch_size=config["BATCH_SIZE"], augment_data=config["AUGMENT_DATA"])
    
    animate_loader(train_loader, save_path=config["IMAGES_DIR"])

    print(f"[GPU{config['GLOBAL_RANK']}]Train loader has {len(train_loader)} batches of length {config['WINDOW_SIZE']}, equalling {len(train_loader) * config['BATCH_SIZE'] * config['WINDOW_SIZE']} frames.")
    print(f"[GPU{config['GLOBAL_RANK']}]Test loader has {len(test_loader)} batches of length {config['WINDOW_SIZE']}, equalling {len(test_loader) * config['BATCH_SIZE'] * config['WINDOW_SIZE']} frames.")
    scaler = torch.amp.GradScaler("cuda")
    
    for epoch in range(starting_epoch, config["EPOCHS"]):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        print(f"GPU{config['GLOBAL_RANK']}]Beging training Epoch [{epoch + 1}/{config['EPOCHS']}], training on {100 * config['PORTION_OF_DATASET']}% of the dataset, as set by PORTION_OF_DATASET in config.")
        epoch_train_loss = run_epoch(model=model, loader=train_loader, optimizer=optimizer, scaler=scaler, criterion=criterion, mode="TRAIN", portion_of_dataset=config["PORTION_OF_DATASET"], local_rank=config["LOCAL_RANK"], global_rank=config["GLOBAL_RANK"], model_type=config["MODEL_TYPE"])
        epoch_test_loss = run_epoch(model=model, loader=test_loader, optimizer=optimizer, scaler=scaler, criterion=criterion, mode="TEST", portion_of_dataset=config["PORTION_OF_DATASET"], local_rank=config["LOCAL_RANK"], global_rank=config["GLOBAL_RANK"], model_type=config["MODEL_TYPE"])
        print(f"[GPU{config['GLOBAL_RANK']}]Epoch [{epoch + 1}/{config['EPOCHS']}], Train Loss (RMSE in degs of steering): {180 * (epoch_train_loss ** 0.5): .4f} Test Loss (RMSE in degs of steering): {180 * (epoch_test_loss ** 0.5): .4f}")
        
        save_model(last_model_dir, model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)
        if (epoch_test_loss < best_test_loss):
            best_test_loss = epoch_test_loss
            save_model(best_model_dir, model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)

    destroy_process_group()

if __name__ == "__main__":
    main()