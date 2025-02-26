import torch
import numpy as np
from torch.utils.data import Dataset

import glob
import h5py

import random

class LazyDataset(Dataset):
    '''
    adapted from https://stackoverflow.com/questions/74375033/pytorch-dataloader-for-multiple-files-with-sliding-window
    '''
    def __init__(self, data_folder, *, transform=None, target_transform=None, window_size=1, use_aps=True, use_dvs=True, DVS_FRAME_NAME, APS_FRAME_NAME, DTYPE):
        if not (use_aps or use_dvs):
            raise ValueError("At least one of use_aps or use_dvs must be True.")
        self.transform = transform
        self.data_file_list = glob.glob(f"{data_folder}/*.hdf5")
        self.target_transform = target_transform
        self.window_size = window_size
        self.use_aps = use_aps
        self.use_dvs = use_dvs
        self.apsdvs_min = 0
        self.DVS_FRAME_NAME = DVS_FRAME_NAME
        self.APS_FRAME_NAME = APS_FRAME_NAME
        self.DTYPE = DTYPE

        self.total_windows, self.dataset_dict = self.create_dataset_dict()

    def find_num_windows(self, path_to_file):
        # Note the integer division here, drop_last is implied true which is OK.
        with h5py.File(path_to_file, "r") as f:
            return len(f["steering_wheel_angle"]) // self.window_size
    
    def create_dataset_dict(self):
        total_windows = 0
        overall_index = 0
        dataset_dict = {}
        for data_file in self.data_file_list:
            windows_of_current_file = self.find_num_windows(data_file)
            total_windows += windows_of_current_file

            # Show which overall index corresponds to which file and window index
            for i in range(windows_of_current_file):
                dataset_dict[overall_index] = {
                    "filename" : data_file,
                    "window_index" : i
                }
                overall_index += 1

        return total_windows, dataset_dict
    
    def read_file(self, filename, window_index):
        with h5py.File(filename, "r") as f:
            start = window_index * self.window_size
            end = start + self.window_size

            # For example, if both APS and DVS streams are used,
            # load each stream and concatenate along the channel dimension.
            data_aps = f[self.APS_FRAME_NAME][start:end]  # shape: (window_size, H, W)
            data_dvs = f[self.DVS_FRAME_NAME][start:end]  # shape: (window_size, H, W)
            if self.use_aps and self.use_dvs:
                data = np.stack((data_aps, data_dvs), axis=3)  # shape: (window_size, H, W, C1+C2)
            elif self.use_aps:
                data = np.expand_dims(data_aps, axis=3)  # shape: (window_size, H, W, C)
            elif self.use_dvs:
                data = np.expand_dims(data_dvs, axis=3)  # shape: (window_size, H, W, C)

            # Read target steering angles corresponding to the window.
            target = f["steering_wheel_angle"][start:end]  # shape: (window_size,)
            # Ensure target is in numpy array format.
            target = np.array(target)
        return data, target

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        filename = self.dataset_dict[index]["filename"]
        window_index = self.dataset_dict[index]["window_index"]
        data, target = self.read_file(filename, window_index)
        
        if self.transform:
                # print(f"Data shape {data.shape} target shape {target.shape}")
                if self.transform:
                    # If the data represents a sequence (e.g. multiple frames), apply transform per frame.
                    if data.ndim == 4:  # expecting shape (num_frames, height, width, channels)
                        
                        # make all frames in sequence have same transforms
                        frames = []
                        state = torch.get_rng_state()
                        for frame in data:
                            torch.set_rng_state(state)
                            frames.append(self.transform(frame))
                        
                        # reset RNG state
                        torch.manual_seed(torch.seed())              
                        data = torch.stack(frames, dim=0)
                    else:
                        data = self.transform(data)
        
        target = target.astype(self.DTYPE)
        if self.target_transform:
            target = self.target_transform(target)
        
        return data, target
        
    def print_dataset_dict(self):
        for key, value in self.dataset_dict.items():
            print(f"Index: {key}, Filename: {value['filename']}, Window Index: {value['window_index']}")