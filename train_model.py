#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

# multinode training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import contextlib
import os
import glob
import gc
from dotenv import load_dotenv

from dataset_classes.lazy_dataset import LazyDataset
from src.setup_model import setup_model

load_dotenv("config.env")
# --- Dataset parameters ---
USE_DVS = int(os.getenv("USE_DVS"))
USE_APS = int(os.getenv("USE_APS"))
MODEL_DIMS = USE_DVS + USE_APS
if MODEL_DIMS == 0: raise Exception("MODEL_DIMS must be non-zero")

BINSIZE_MS = int(os.getenv("BINSIZE_MS"))
HEIGHT = int(os.getenv("HEIGHT"))
WIDTH = int(os.getenv("WIDTH"))
PRECISION = int(os.getenv("PRECISION"))
DTYPE = np.float16 if PRECISION == 16 else np.float32
AUGMENT_DATA = bool(int(os.getenv("AUGMENT_DATA")))

APS_FRAME_NAME = f"aps_frame_{HEIGHT}x{WIDTH}"
DVS_FRAME_NAME = f"dvs_frame_{HEIGHT}x{WIDTH}"

# --- Location parameters ---
DATA_DIR = os.getenv("DATA_DIR")
MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR")
TRAINING_DIR = f"{DATA_DIR}/Training"
TEST_DIR = f"{DATA_DIR}/Test"
IMAGES_DIR = os.getenv("IMAGES_DIR")

# --- Training parameters ---
MODEL_TYPE = os.getenv("MODEL_TYPE")
PRETRAINED = bool(int(os.getenv("PRETRAINED")))
WARM_START= bool(int(os.getenv("WARM_START")))
CLEAN_OLD_MODELS = bool(int(os.getenv("CLEAN_OLD_MODELS")))
if WARM_START and CLEAN_OLD_MODELS: raise Exception("Can't have WARM_START and CLEAN_OLD_MODELS at the same time!")
TRAINING_SPLIT = float(os.getenv("TRAINING_SPLIT"))
EPOCHS = int(os.getenv("EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))

# --- Multi-node training parameters ---
LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
GLOBAL_RANK = int(os.getenv("RANK"))

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def setup_transforms():
    inv_apsdvs_max = 1 / 255
    def normalise_dvsaps(data):
        # shorthand for (data - self.apsdvs_min) / (self.apsdvs_max - self.apsdvs_min) since we know the min is 0
        return data * inv_apsdvs_max

    transform_list = [
        lambda x: normalise_dvsaps(x),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]

    if AUGMENT_DATA:
        augmentations = [
            transforms.RandomApply([transforms.RandomResizedCrop(224, ratio=(0.75, 1.25), scale=(0.64, 1.0))], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
            transforms.RandomApply([transforms.GaussianNoise(mean=0, sigma=0.02, clip=True)], p=0.5)
        ]

        transform_list.extend(augmentations)

    transform = transforms.Compose(transform_list)

    # normalise the steering values from -180 to 180 to -1 to 1
    # Precompute normalization coefficients
    steering_slope = 1 / 180  # (output_max - output_min)/(input_max - input_min) = (1-(-1))/(180-(-180))
    def normalise_steering(data):
        return data * steering_slope

    # used for getting the data out at the end.
    inv_steering_slope = 180
    def denormalise_steering(data):
        return data * inv_steering_slope

    target_transform = transforms.Compose([
        lambda x: normalise_steering(x)
    ])

    return transform, target_transform

def prepare_dataloader():
    transform, target_transform = setup_transforms()
    train_dataset = LazyDataset(TRAINING_DIR, transform=transform, target_transform=target_transform, window_size=WINDOW_SIZE, use_aps=USE_APS, use_dvs=USE_DVS, DVS_FRAME_NAME=DVS_FRAME_NAME, APS_FRAME_NAME=APS_FRAME_NAME, DTYPE=DTYPE)
    test_dataset = LazyDataset(TEST_DIR, transform=transform, target_transform=target_transform, window_size=WINDOW_SIZE, use_aps=USE_APS, use_dvs=USE_DVS, DVS_FRAME_NAME=DVS_FRAME_NAME, APS_FRAME_NAME=APS_FRAME_NAME, DTYPE=DTYPE)

    # load the data, can implement batching etc. here
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=5, sampler=DistributedSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=5, sampler=DistributedSampler(test_dataset))

    return train_loader, test_loader

def warm_start(model, optimizer):
        """
        Load the last model and optimizer state from the last saved model, returns starting epoch and best loss
        """
        loc = f"cuda:{LOCAL_RANK}"
        last_model_file = glob.glob(f"{MODEL_SAVE_DIR}/last/*")[-1]
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


def animate_loader(train_loader, save_path=f"{IMAGES_DIR}/animation.gif"):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import display
    frames = []  # for storing the generated images
    # Create 2 subplots (one for each channel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("APS Images")
    axes[1].set_title("DVS Images")
    
    print(f"Saving an animation of the first 10 batches slowed down 20x to {save_path}")
    # Loop over the first 100 batches
    for i, (combined_images, steering_angles) in enumerate(train_loader):
        if i >= 10:
            break
        b, t, c, w, h = combined_images.shape
        combined_images = combined_images.view(-1, c, w, h)
        # Extract the images for each channel
        aps_images = combined_images[:, 0, :, :].cpu().numpy()  # converting tensor to numpy if needed
        dvs_images = combined_images[:, 1, :, :].cpu().numpy()
        steering_angles = steering_angles.flatten().cpu().numpy()
        # Display images on the subplots (set animated=True for animation)
        for j in range(dvs_images.shape[0]):
            im0 = axes[0].imshow(dvs_images[j], cmap="gray", vmin=0, vmax=1.0, animated=True, origin="lower")
            im1 = axes[1].imshow(aps_images[j], cmap="gray", vmin=0, vmax=1.0, animated=True, origin="lower")                
            # Add steering angle as text on the image
            angle_text = axes[0].text(0.5, -0.1, f"Steering Angle: {180 * steering_angles[j]:.2f}", 
                                    size=12, ha="center", transform=axes[0].transAxes, color="red")
            # Append the artists as the current frame.
            frames.append([im0, im1, angle_text])
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True, repeat_delay=1000)
            
    # Save the animation if a save_path is provided.
    if save_path is not None:
        ani.save(save_path, dpi=80, writer='pillow')

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


def run_epoch(model, loader, optimizer, scaler, criterion, mode):
        epoch_loss = 0
        context = torch.no_grad() if mode == "TEST" else contextlib.nullcontext()

        # report every 25% progress
        total_batches = len(loader)
        progress_thresholds = {int(total_batches * 0.25), int(total_batches * 0.50), int(total_batches * 0.75)}
        with context:
            for i, (combined_images, steering_angle) in enumerate(loader):
                if MODEL_TYPE == "Resnet":
                    b, t, c, w, h = combined_images.shape
                    combined_images = combined_images.view(b * t, c, w, h)
                elif MODEL_TYPE != "ResnetRNN":
                    combined_images = combined_images.permute(0, 2, 1, 3, 4)

                if mode != "TEST":
                    optimizer.zero_grad()

                combined_images = combined_images.to(LOCAL_RANK)
                steering_angle = steering_angle.to(LOCAL_RANK)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(combined_images).squeeze(1)
                    if MODEL_TYPE == "Resnet":
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
                    print(f"[GPU{GLOBAL_RANK}]{mode}, Epoch progress: {progress}% completed.")
                
                del combined_images, steering_angle, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Average training loss for the epoch
            epoch_loss /= len(loader)
            return epoch_loss

def main():
    ddp_setup()
    model = setup_model(MODEL_TYPE, MODEL_DIMS, PRETRAINED, LOCAL_RANK)
    model = DDP(model, device_ids=[LOCAL_RANK])

    starting_epoch = 0
    best_test_loss = np.inf
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if CLEAN_OLD_MODELS:
        clean_models(MODEL_SAVE_DIR)
    if WARM_START:
        starting_epoch, best_test_loss = warm_start(model, optimizer)
    
    train_loader, test_loader = prepare_dataloader()
    
    animate_loader(train_loader)

    print(f"[GPU{GLOBAL_RANK}]Train loader has {len(train_loader)} batches of length {WINDOW_SIZE}, equalling {len(train_loader) * BATCH_SIZE * WINDOW_SIZE} frames.")
    print(f"[GPU{GLOBAL_RANK}]Test loader has {len(test_loader)} batches of length {WINDOW_SIZE}, equalling {len(test_loader) * BATCH_SIZE * WINDOW_SIZE} frames.")
    scaler = torch.amp.GradScaler("cuda")
    
    for epoch in range(starting_epoch, EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        epoch_train_loss = run_epoch(model=model, loader=train_loader, optimizer=optimizer, scaler=scaler, criterion=criterion, mode="TRAIN")
        epoch_test_loss = run_epoch(model=model, loader=test_loader, optimizer=optimizer, scaler=scaler, criterion=criterion, mode="TEST")
        print(f"[GPU{GLOBAL_RANK}]Epoch [{epoch + 1}/{EPOCHS}], Train Loss (RMSE in degs of steering): {180 * (epoch_train_loss ** 0.5): .4f} Test Loss (RMSE in degs of steering): {180 * (epoch_test_loss ** 0.5): .4f}")
        
        save_model(f"{MODEL_SAVE_DIR}/last", model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)
        if (epoch_test_loss < best_test_loss):
            best_test_loss = epoch_test_loss
            save_model(f"{MODEL_SAVE_DIR}/best", model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)

    destroy_process_group()

if __name__ == "__main__":
    main()