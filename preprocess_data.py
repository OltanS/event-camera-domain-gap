import numpy as np
import h5py
import math
import random
import csv
import os
import shutil
import glob
from pathlib import Path

from src.config_utils import setup_config_variables

# Useful if we face an error halfway through the script and don't want to re-run the whole thing
# Keep all True for correct operation otherwise.
SPLIT_DATASET = True
PRUNE_TRAINING = True
PRUNE_TEST = True

# ------------ Functions ------------ #

relevant_keys = [
    "dvs_frame_224x224",
    "aps_frame_224x224",
    "vehicle_speed",
    "steering_wheel_angle"
]

def search_csv(file_path, target):
    '''
    Look for the first row of CSV for target, return its second and third column if found.
    args:
        file_path: str, path to the CSV file.
        target: str, the target filename to search for in the first column.
    returns:
        int, int, bool: the start time of the recording, end time of the recording and whether the recording was flipped.
    '''
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == target:
                return int(row[1]), int(row[2]), not bool(int(row[3]))
    return None, None, None  # Fixed to return two values

def prune_dataset(source_dir, config, is_training=True):
    """
    Prune dataset files based on speed, steering thresholds and other criteria.
    
    Args:
        source_dir: Directory containing the dataset files
        config: Configuration dictionary with pruning parameters
        is_training: Whether this is training data (affects pruning logic)
    """
    pruned_path = source_dir + "/Pruned-Recordings"
    try:
        os.mkdir(pruned_path)
    except FileExistsError:
        pass

    data_files = glob.glob(f"{source_dir}/*.hdf5")
    for i, file in enumerate(data_files):
        p = Path(file)
        print(f"Pruning {'training' if is_training else 'test'} file: {p.name}. [{i + 1}/{len(data_files)}]")
        
        indices_to_remove = []
        with h5py.File(file, "r") as f:
            speeds = np.array(f["vehicle_speed"])
            steering_angles = np.array(f["steering_wheel_angle"])
            file_len = len(speeds)
            print(f"There are {file_len} entries in this file.")
            
            for j in range(0, len(speeds)):
                # Speed threshold check
                if config["PRUNE_SPEED"]:
                    if speeds[j] < config["SPEED_THRESHOLD"]:
                        indices_to_remove.append(j)
                        continue
                
                # Training-specific steering pruning
                if is_training and config["PRUNE_STEERING"]:
                    if steering_angles[j] < config["STEERING_THRESHOLD"] and random.random() < config["STEERING_DISCARD_CHANCE"]:
                        indices_to_remove.append(j)
                        continue
                
                # Steering angle limits check
                if steering_angles[j] > config["MAX_STEERING"] or steering_angles[j] < config["MIN_STEERING"]:
                    indices_to_remove.append(j)
                    continue
            
            # Keep only non-discarded indices
            indices_to_keep = np.setdiff1d(np.arange(file_len), indices_to_remove)
            
            renamed_file = f"{pruned_path}/{p.stem}_pruned.hdf5"
            with h5py.File(renamed_file, "w") as new_f:
                for dset_name in f.keys():
                    if len(f[dset_name]) != file_len:
                        continue
                    full_data = np.array(f[dset_name])
                    new_f.create_dataset(dset_name, data=full_data[indices_to_keep])
            
            print(f"Discarded {len(indices_to_remove)} batches, "
                  f"{'as their values were below the thresholds, ' if is_training else ''}"
                  f"totalling {len(indices_to_remove)} frames.")
            print(f"{len(indices_to_keep)} frames remain, with average speed {np.average(speeds[indices_to_keep])} "
                  f"and average absolute steering {np.average(np.absolute(steering_angles[indices_to_keep]))}")

    # Replace original files with pruned versions
    for f in glob.glob(f"{source_dir}/*.hdf5"):
        os.remove(f)
    for f in glob.glob(f"{pruned_path}/*.hdf5"):
        shutil.move(f, source_dir)
    os.rmdir(pruned_path)

def main():
    config = setup_config_variables()
    BINS_PER_S = int(1000 / config["BINSIZE_MS"])
    
    random.seed(config["RANDOM_SEED"]) 
    for directory in [config["TRAINING_DIR"], config["TEST_DIR"]]:
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

    if SPLIT_DATASET: 
        data_files = sorted(glob.glob(f"{config['OUT_DIR']}/*.hdf5"))
        for i, df in enumerate(data_files):
            try:
                p = Path(df)
                file_base_name = p.name.replace(f"_bin{config['BINSIZE_MS']}ms", '')
                start_s, end_s, is_flipped = search_csv(config["CSV_FILE_MANUAL"], file_base_name)
                start_index, end_index = start_s * BINS_PER_S, end_s * BINS_PER_S
                
                train_file_name = f"{config['TRAINING_DIR']}/{p.stem}_train.hdf5"
                test_file_name = f"{config['TEST_DIR']}/{p.stem}_test.hdf5"

                if glob.glob(train_file_name) or glob.glob(test_file_name):
                    print(f"Skipping {p.name} the file has already been preprocessed. If you'd like to do it again, please delete the file.")
                    continue

                print(f"Splitting {p.name} into training and testing subsections. [{i + 1}/{len(data_files)}]")
                data_len = end_index - start_index
                split_index = start_index + math.ceil(data_len * config["TRAINING_SPLIT"])
                print(f"Start(s) - End(s): {start_s} - {end_s} | Start (index) - End (index): {start_index} - {end_index}, split index {split_index}")
                
                with h5py.File(df, "r") as f, \
                     h5py.File(train_file_name, "w") as train_f, \
                     h5py.File(test_file_name, "w") as test_f:
                    for key, data in f.items():
                        if key in relevant_keys:
                            tform = lambda d : d
                            if is_flipped and key in relevant_keys[0:2]:
                                print(f"Recording was mounted upside down, rotating {key} 180 degs")
                                data = np.rot90(data, k=2, axes=(1, 2))

                            # steering wheel angle is mirrored for DDD20 for some reason, we need to negate it
                            elif not is_flipped and key == relevant_keys[3]:
                                print("Fixing rotated steering angle convention for DDD20")
                                tform = lambda d: -d

                            # tform does nothing unless we need to fix the DDD20 steering
                            train_data = tform(data[start_index:split_index])
                            test_data = tform(data[split_index:end_index])

                            train_f.create_dataset(key, data=train_data)
                            print(f"Created dataset {key} in train set with length {len(train_f[key])}")
                            test_f.create_dataset(key, data=test_data)
                            print(f"Created dataset {key} in test set with length {len(test_f[key])}")
                            
                print(f"Done preprocessing {p.name}")
            except OSError as e:
                print(f"Skipping {df}, cannot open file. Reason: {e}.\n This most likely means the file is corrupted!")
    
    if config["PERFORM_PRUNING"]:
        if PRUNE_TRAINING:
            prune_dataset(config["TRAINING_DIR"], config, is_training=True)
        if PRUNE_TEST:
            prune_dataset(config["TEST_DIR"], config, is_training=False)

if __name__ == "__main__":
    main()