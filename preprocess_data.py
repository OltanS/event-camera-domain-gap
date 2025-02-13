import numpy as np
import h5py
import math
import random
import csv
import os
import shutil
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config.env")
# --- Dataset parameters --- #
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
BINSIZE_MS = int(os.getenv("BINSIZE_MS"))
BINS_PER_S = int(1000 / BINSIZE_MS)

# --- Location parameters --- #
DATA_DIR = os.getenv("OUT_DIR")
CSV_FILE = os.getenv("CSV_FILE")
PROCESSED_DIR = os.getenv("PROCESSED_DIR")

# --- Preprocessing parameters --- #
PERFORM_PRUNING = bool(int(os.getenv("PERFORM_PRUNING")))
WINDOWED_PRUNING = bool(int(os.getenv("WINDOWED_PRUNING")))
# km/h for speed to be considered high enough
SPEED_THRESHOLD = float(os.getenv("SPEED_THRESHOLD"))
# absolute value for degree of steering to be considered high enough
STEERING_THRESHOLD = float(os.getenv("STEERING_THRESHOLD"))
STEERING_DISCARD_CHANCE = float(os.getenv("STEERING_DISCARD_CHANCE"))
MIN_STEERING = int(os.getenv("MIN_STEERING"))
MAX_STEERING = int(os.getenv("MAX_STEERING"))

# --- Training parameters --- #
TRAINING_SPLIT = float(os.getenv("TRAINING_SPLIT"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE"))

# Prune on a frame-by-frame basis.
if not WINDOWED_PRUNING:
    WINDOW_SIZE = 1

# --- Fix randomness --- #
random.seed(RANDOM_SEED)

TRAINING_DIR = f"{PROCESSED_DIR}/Training"
TEST_DIR = f"{PROCESSED_DIR}/Test"


# ------------ Functions ------------ #

def search_csv(file_path, target):
    '''
    Look for the first row of CSV for target, return its second and third column if found.
    '''
    with open(file_path, 'r') as file:
        "returns start_time, end_time, is_flipped"
        reader = csv.reader(file)
        for row in reader:
            if row[0] == target:
                return int(row[1]), int(row[2]), bool(int(row[3]))
    return None, None, None

def main():
    try:
        os.mkdir(TRAINING_DIR)
    except FileExistsError:
        pass

    try:
        os.mkdir(TEST_DIR)
    except FileExistsError:
        pass

    data_files = sorted(glob.glob(f"{DATA_DIR}/*.hdf5"))
    for i, df in enumerate(data_files):
        try:
            p = Path(df)
            # The issue is, there are unused bits of the file at the beginning and end, that are not chopped off with the previous preprocessing.
            # Thankfully, where we should chop is shown in the relevant CSV file, and we can calculate the indices they correspond to.
            start_s, end_s, is_flipped = search_csv(CSV_FILE, p.name.replace(f"_bin{BINSIZE_MS}ms", ''))
            start_index, end_index = start_s * BINS_PER_S, end_s *BINS_PER_S
            with h5py.File(df, "r") as f:
                train_file_name = f"{TRAINING_DIR}/{p.stem}_train.hdf5"
                test_file_name = f"{TEST_DIR}/{p.stem}_test.hdf5"

                if glob.glob(train_file_name) or glob.glob(test_file_name):
                    print(f"Skipping {p.name} the file has already been preprocessed. If you'd like to do it again, please delete the file.")
                    continue

                print(f"Splitting {p.name} into training and testing subsections. [{i + 1}/{len(data_files)}]")
                print(f"Start(s) - End(s): {start_s} - {end_s} | Start (index) - End (index) {start_index} - {end_index}")
                with h5py.File(train_file_name, "w") as train_f, h5py.File(test_file_name, "w") as test_f:
                    for key, data in f.items():
                        data_len = end_index - start_index
                        split_index = math.ceil(data_len * TRAINING_SPLIT)
                        # if the CSV indicates the video should be flipped, flip video streams upside down.
                        if is_flipped and (key.lower().startswith("aps_frame") or key.lower().startswith("dvs_frame")):
                            print(f"File {p.name} is flipped, flipping video data key: {key.lower()}")
                            # for processing speed, it would be faster to flip the "unflipped" videos, since it doesnt matter which way the video is flipped.
                            # But for viewing purposes, it is better to flip the "flipped" videos.
                            data = np.flipud(data)

                        train_f.create_dataset(key, data=data[start_index:split_index])
                        test_f.create_dataset(key, data=data[split_index:end_index])
            print(f"Done preprocessing {p.name}")
        except OSError as e:
            print(f"Skipping {df}, cannot open file. Reason: {e}.\n This most likely means the file is corrupted!")
    # Perform full suite of preprocessing on training data
    if PERFORM_PRUNING:
        PRUNE_SPEED = bool(int(os.getenv("SPEED_PRUNING")))
        PRUNE_STEERING = bool(int(os.getenv("STEERING_PRUNING")))
        
        PRUNED_DATASET_PATH = TRAINING_DIR + "/Pruned-Recordings"
        os.mkdir(PRUNED_DATASET_PATH)
        
        training_files = glob.glob(f"{TRAINING_DIR}/*.hdf5")
        for i, file in enumerate(training_files):
            p = Path(file)
            print(f"Pruning file: {p.name}. Batch size is {WINDOW_SIZE}. [{i + 1}/{len(training_files)}]")
            discarded_batches = []
            with h5py.File(file, "r") as f:
                speeds = np.array(f["vehicle_speed"])
                steering_angles = np.array(f["steering_wheel_angle"])
                file_len = len(speeds)
                for i in range(0, len(speeds), WINDOW_SIZE):
                    if PRUNE_SPEED:
                        if np.median(speeds[i:i+WINDOW_SIZE]) < SPEED_THRESHOLD:
                            discarded_batches.append(i)
                            continue
                    if PRUNE_STEERING:
                        if np.median(np.absolute(steering_angles[i:i+WINDOW_SIZE])) < STEERING_THRESHOLD and random.random() < STEERING_DISCARD_CHANCE:
                            discarded_batches.append(i)
                            continue
                        
                    # remove extreme values
                    if any(angle > MAX_STEERING or angle < MIN_STEERING for angle in steering_angles[i:i+WINDOW_SIZE]):
                        discarded_batches.append(i)
                        continue   

                indices_to_remove = []
                for idx in discarded_batches:
                    indices_to_remove.extend(range(idx, idx + WINDOW_SIZE))
                indices_to_keep = np.setdiff1d(np.arange(file_len), indices_to_remove)
                renamed_file = f"{PRUNED_DATASET_PATH}/{p.stem}_pruned.hdf5"
                with h5py.File(renamed_file, "w") as new_f:
                    for dset_name in f.keys():
                        if len(f[dset_name]) != file_len:
                            # print(f"Dataset {dset_name} has a different length than the rest of the data, skipping.")
                            # in practice this is just test_idxs, and train_idxs, which are not needed
                            continue
                        full_data = np.array(f[dset_name])
                        new_f.create_dataset(dset_name, data=full_data[indices_to_keep])
                
                print(f"Discarded {len(discarded_batches)} batches, as their median was below the thresholds, totalling {len(indices_to_remove)} frames.")
                print(f"{len(indices_to_keep)} frames remain, with average speed {np.average(speeds[indices_to_keep])} and average absolute steering {np.average(np.absolute(steering_angles[indices_to_keep]))}") 

        for f in glob.glob(f"{TRAINING_DIR}/*.hdf5"):
            os.remove(f)
        for f in glob.glob(f"{PRUNED_DATASET_PATH}/*.hdf5"):
            shutil.move(f, TRAINING_DIR)
        os.rmdir(PRUNED_DATASET_PATH)

        # --- Prune testing data but DO NOT discard for low steering angles ---
        PRUNED_TESTSET_PATH = TEST_DIR + "/Pruned-Recordings"
        os.mkdir(PRUNED_TESTSET_PATH)
        
        testing_files = glob.glob(f"{TEST_DIR}/*.hdf5")
        for i, file in enumerate(testing_files):
            p = Path(file)
            print(f"Pruning test file: {p.name}. Batch size is {WINDOW_SIZE}. [{i + 1}/{len(testing_files)}]")
            discarded_batches = []
            with h5py.File(file, "r") as f:
                speeds = np.array(f["vehicle_speed"])
                steering_angles = np.array(f["steering_wheel_angle"])
                file_len = len(speeds)
                for j in range(0, len(speeds), WINDOW_SIZE):
                    if PRUNE_SPEED:
                        if np.median(speeds[j:j+WINDOW_SIZE]) < SPEED_THRESHOLD:
                            discarded_batches.append(j)
                            continue
                    # Skipping low steering angle check for testing data.
                    if any(angle > MAX_STEERING or angle < MIN_STEERING for angle in steering_angles[j:j+WINDOW_SIZE]):
                        discarded_batches.append(j)
                        continue
                
                indices_to_remove = []
                for idx in discarded_batches:
                    indices_to_remove.extend(range(idx, idx + WINDOW_SIZE))
                indices_to_keep = np.setdiff1d(np.arange(file_len), indices_to_remove)
                renamed_file = f"{PRUNED_TESTSET_PATH}/{p.stem}_pruned.hdf5"
                with h5py.File(renamed_file, "w") as new_f:
                    for dset_name in f.keys():
                        if len(f[dset_name]) != file_len:
                            continue
                        full_data = np.array(f[dset_name])
                        new_f.create_dataset(dset_name, data=full_data[indices_to_keep])
                
                print(f"Discarded {len(discarded_batches)} batches, totalling {len(indices_to_remove)} frames.")
                print(f"{len(indices_to_keep)} frames remain, with average speed {np.average(speeds[indices_to_keep])} and average absolute steering {np.average(np.absolute(steering_angles[indices_to_keep]))}") 

        for f in glob.glob(f"{TEST_DIR}/*.hdf5"):
            os.remove(f)
        for f in glob.glob(f"{PRUNED_TESTSET_PATH}/*.hdf5"):
            shutil.move(f, TEST_DIR)
        os.rmdir(PRUNED_TESTSET_PATH)

if __name__ == "__main__":
    main()