import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from torchvision import transforms
import matplotlib.pyplot as plt

import h5py

import gc
import math
import random

# for deleting old best models
import os
import shutil
import glob

from dotenv import load_dotenv

from DatasetClasses.LazyDDD20Dataset import LazyDDD20Dataset
from NetworkModels import ResNetRNN, ResNetRNNStated, ResidualBlock
###########################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}")
random.seed(1)

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

APS_FRAME_NAME = f"aps_frame_{HEIGHT}x{WIDTH}"
DVS_FRAME_NAME = f"dvs_frame_{HEIGHT}x{WIDTH}"

# --- Location parameters ---
DATASET_DIR = os.getenv("OUT_DIR")
MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR")
# --- Preprocessing parameters ---
PERFORM_PRUNING = bool(int(os.getenv("PERFORM_PRUNING")))
# km/h for speed to be considered high enough
SPEED_THRESHOLD = float(os.getenv("SPEED_THRESHOLD"))
# absolute value for degree of steering to be considered high enough
STEERING_THRESHOLD = float(os.getenv("STEERING_THRESHOLD"))
STEERING_DISCARD_CHANCE = float(os.getenv("STEERING_DISCARD_CHANCE"))

# --- Training parameters ---
WARM_START= bool(int(os.getenv("WARM_START")))
CLEAN_OLD_MODELS = bool(int(os.getenv("CLEAN_OLD_MODELS")))
if WARM_START and CLEAN_OLD_MODELS: raise Exception("Can't have WARM_START and CLEAN_OLD_MODELS at the same time!")
USE_STATED= bool(int(os.getenv("USE_STATED")))
TRAINING_SPLIT = float(os.getenv("TRAINING_SPLIT"))
EPOCHS = int(os.getenv("EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))

# Randomly allocate a percentage of the data to the training set
TRAINING_DIR = f"{DATASET_DIR}/Training"
TEST_DIR = f"{DATASET_DIR}/Test"

# Try to create directory, shuffle the files back to the main directory if it already exists
try:
    os.mkdir(TRAINING_DIR)
except FileExistsError:
    for files in glob.glob(TRAINING_DIR + "/*.hdf5"):
        os.rename(files, f"{DATASET_DIR}/{os.path.basename(files)}")
    # delete everything that remains, a.k.a. pruned files
    for file in glob.glob(TRAINING_DIR + "/*"):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)

try:
    os.mkdir(TEST_DIR)
except FileExistsError:
    for files in glob.glob(TEST_DIR + "/*"):
         os.rename(files, f"{DATASET_DIR}/{os.path.basename(files)}")

# Sort by name to make the randomness seedable
data_files = sorted(glob.glob(f"{DATASET_DIR}/*.hdf5"))

# Randomly shuffle the data files
random.shuffle(data_files)

split_idx = int(len(data_files) * TRAINING_SPLIT)
train_files = data_files[:split_idx]
test_files = data_files[split_idx:]

for file in train_files:
    os.rename(file, f"{TRAINING_DIR}/{os.path.basename(file)}")

for file in test_files:
    os.rename(file, f"{TEST_DIR}/{os.path.basename(file)}")

# Only prune training data
if PERFORM_PRUNING:
    PRUNE_SPEED = bool(int(os.getenv("SPEED_PRUNING")))
    PRUNE_STEERING = bool(int(os.getenv("STEERING_PRUNING")))
    
    PRUNED_DATASET_PATH = TRAINING_DIR + "/Pruned-Recordings"
    os.mkdir(PRUNED_DATASET_PATH)
    
    training_files = glob.glob(f"{TRAINING_DIR}/*.hdf5")
    for file in training_files:
        print(f"Pruning file: {file}")
        discarded_batches = []
        with h5py.File(file, "r") as f:
            speeds = np.array(f["vehicle_speed"])
            steering_angles = np.array(f["steering_wheel_angle"])
            file_len = len(speeds)
            for i in range(0, len(speeds), BATCH_SIZE):
                if PRUNE_SPEED:
                    if np.average(speeds[i:i+BATCH_SIZE]) < SPEED_THRESHOLD:
                        discarded_batches.append(i)
                        continue
                if PRUNE_STEERING:
                    if np.average(np.absolute(steering_angles[i:i+BATCH_SIZE])) < STEERING_THRESHOLD and random.random() < STEERING_DISCARD_CHANCE:
                        discarded_batches.append(i)

            indices_to_remove = []
            for idx in discarded_batches:
                indices_to_remove.extend(range(idx, idx + BATCH_SIZE))
            indices_to_keep = np.setdiff1d(np.arange(file_len), indices_to_remove)
            renamed_file = f"{PRUNED_DATASET_PATH}/{os.path.splitext(os.path.basename(file))[0]}_pruned.hdf5"
            with h5py.File(renamed_file, "w") as new_f:
                for dset_name in f.keys():
                    if len(f[dset_name]) != file_len:
                        # print(f"Dataset {dset_name} has a different length than the rest of the data, skipping.")
                        # in practice this is just test_idxs, and train_idxs, which are not needed
                        continue
                    full_data = np.array(f[dset_name])
                    new_f.create_dataset(dset_name, data=full_data[indices_to_keep])
            
            print(f"Discarded {len(discarded_batches)} batches, as their average was below the thresholds, totalling {len(indices_to_remove)} frames.")
            print(f"{len(indices_to_keep)} frames remain, with average speed {np.average(speeds[indices_to_keep])} and average absolute steering {np.average(np.absolute(steering_angles[indices_to_keep]))}") 

    TRAINING_DIR = PRUNED_DATASET_PATH

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

train_dataset = LazyDDD20Dataset(TRAINING_DIR, transform=transform, window_size=1, use_aps=USE_APS, use_dvs=USE_DVS, DVS_FRAME_NAME=DVS_FRAME_NAME, APS_FRAME_NAME=APS_FRAME_NAME, DTYPE=DTYPE)
test_dataset = LazyDDD20Dataset(TEST_DIR, transform=transform, window_size=1, use_aps=USE_APS, use_dvs=USE_DVS, DVS_FRAME_NAME=DVS_FRAME_NAME, APS_FRAME_NAME=APS_FRAME_NAME, DTYPE=DTYPE)

# load the data, can implement batching etc. here
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
print(f"Train loader has {len(train_loader)} batches, equalling {len(train_loader) * BATCH_SIZE} frames.\n-Test loader has {len(test_loader)} batches, equalling {len(test_loader) * BATCH_SIZE} frames.")

def setup_model(model_dims, use_stated):
    if use_stated:
        model = ResNetRNNStated(ResidualBlock, [3, 4, 6, 3], num_channels=model_dims).to(device)
    else:
        model = ResNetRNN(ResidualBlock, [3, 4, 6, 3], num_channels=model_dims).to(device)
    
    return model

def warm_start(model, optimizer):
        """
        Load the last model and optimizer state from the last saved model, returns starting epoch and best loss
        """
        last_model_file = glob.glob(f"{MODEL_SAVE_DIR}/last/*")[-1]
        load_dict = torch.load(last_model_file, map_location=device)
        model.load_state_dict(load_dict["model_state_dict"])
        optimizer.load_state_dict(load_dict["optimizer_state_dict"])
        epoch = load_dict["epoch"]
        best_test_loss = load_dict["best_test_loss"]
        return epoch + 1, best_test_loss

model = setup_model(MODEL_DIMS, USE_STATED)

starting_epoch = 0
best_test_loss = np.inf

#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

if WARM_START:
    starting_epoch, best_test_loss = warm_start(model, optimizer)

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
    
    torch.save(state, f"{directory}/e{epoch + 1}_L{loss:.2f}.pth")

if CLEAN_OLD_MODELS:
  clean_models(MODEL_SAVE_DIR)

train_losses = []
test_losses = []

scaler = torch.amp.GradScaler("cuda")
for epoch in range(starting_epoch, EPOCHS):
    epoch_train_loss = 0
    for combined_images, steering_angle in train_loader:
      optimizer.zero_grad()
      #Move tensors to the configured device
      combined_images = combined_images.to(device)
      steering_angle = steering_angle.to(device)
      #Forward pass
      with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(combined_images)
        # print(f"o {outputs.dtype} og {steering_angle.dtype}")
        loss = criterion(outputs, steering_angle.squeeze(1))
      #Backward and optimize
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      epoch_train_loss += loss.item()
      del combined_images, steering_angle, outputs
      torch.cuda.empty_cache()
      gc.collect()

    # Average training loss for the epoch
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    #Validation
    epoch_test_loss = 0
    with torch.no_grad():
          for combined_images, steering_angle in test_loader:
              combined_images = combined_images.to(device)
              steering_angle = steering_angle.to(device)
              with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(combined_images)
                loss = criterion(outputs, steering_angle.squeeze(1))
              epoch_test_loss += loss.item()
              del combined_images, steering_angle, outputs
          epoch_test_loss /= len(test_loader)
          
          # Save the best
          if (epoch_test_loss < best_test_loss):
              best_test_loss = epoch_test_loss
              save_model(f"{MODEL_SAVE_DIR}/best", model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)

          # Save every 20 epochs
          if not ((epoch + 1) % 20):
              save_model(MODEL_SAVE_DIR, model, optimizer, epoch, epoch_test_loss, best_test_loss)
          # Save the last model
          save_model(f"{MODEL_SAVE_DIR}/last", model, optimizer, epoch, epoch_test_loss, best_test_loss, clean_old=True)

          # Report on the loss
          test_losses.append(epoch_test_loss)
          print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {epoch_train_loss: .4f} Test Loss: {epoch_test_loss: .4f}")

