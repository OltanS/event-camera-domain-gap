import numpy as np
from torch.utils.data import Dataset

import glob
import h5py

class LazyDDD20Dataset(Dataset):
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
        self.inv_apsdvs_max = 1 / 255
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
            if self.use_dvs:
                    dvs_frame = np.array(f[self.DVS_FRAME_NAME][window_index:window_index + self.window_size])
            if self.use_aps:
                    aps_frame = np.array(f[self.APS_FRAME_NAME][window_index:window_index + self.window_size])
            if self.use_aps and self.use_dvs:
                # the frames are 1 x 224 x 224, we want 224 x 224 x 2 after concat
                combined_frames = np.concatenate((dvs_frame, aps_frame), axis=0)
            else:
                combined_frames = aps_frame if self.use_aps else dvs_frame
            
            combined_frames = combined_frames.transpose(1, 2, 0)
            steering_angles = np.array(f["steering_wheel_angle"][window_index:window_index + self.window_size])
        # print(f"Combined frames shape is {combined_frames.shape}.")
        return combined_frames, steering_angles
    
    # used to normalise the aps and dvs frames to the range [0, 1]
    def normalise(self, data):
        # shorthand for (data - self.apsdvs_min) / (self.apsdvs_max - self.apsdvs_min) since we know the min is 0
        new_data = np.empty_like(data, dtype=self.DTYPE)
        new_data = data * self.inv_apsdvs_max
        return new_data

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        filename = self.dataset_dict[index]["filename"]
        window_index = self.dataset_dict[index]["window_index"]
        data, target = self.read_file(filename, window_index)
        data = self.normalise(data)
        
        if self.transform:
            data = self.transform(data)
        
        target = target.astype(self.DTYPE)
        if self.target_transform:
            target = self.target_transform(target)
        
        return data, target
        
    def print_dataset_dict(self):
        for key, value in self.dataset_dict.items():
            print(f"Index: {key}, Filename: {value['filename']}, Window Index: {value['window_index']}")