import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.v2 as transforms_v2

from dataset_classes.lazy_dataset import LazyDataset

def setup_transforms(augment_data: bool):
    '''
    Prepares the transforms for the dataset.  
    If augment_data is True, the transforms will include random cropping, rotation, and Gaussian noise.  
    The transforms are defined here and nowhere else, so you can modify them from here.
    args:
        augment_data (bool): Whether to augment the data
    returns:
        transform (Compose): The transform to apply to the data
        target_transform (Compose): The transform to apply to the target
    '''
    inv_apsdvs_max = 1 / 255
    def scale_dvsaps(data):
        # shorthand for (data - self.apsdvs_min) / (self.apsdvs_max - self.apsdvs_min) since we know the min is 0
        return data * inv_apsdvs_max

    transform_list = [
        lambda x: scale_dvsaps(x),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=False),
        transforms_v2.Normalize(
            # aps mean, dvs mean
            mean=[0.4827, 0.3458],
            std=[0.4672, 0.1243]
        )
    ]

    if augment_data:
        augmentations = [
            transforms_v2.RandomApply([transforms_v2.RandomResizedCrop(224, ratio=(0.75, 1.25), scale=(0.64, 1.0))], p=0.5),
            transforms_v2.RandomApply([transforms_v2.RandomRotation(degrees=5)], p=0.5),
            transforms_v2.RandomApply([transforms_v2.GaussianNoise(mean=0, sigma=0.02, clip=True)], p=0.5)
        ]

        transform_list.extend(augmentations)

    transform = transforms_v2.Compose(transform_list)

    # normalise the steering values from -180 to 180 to -1 to 1
    # Precompute normalization coefficients
    steering_slope = 1 / 180  # (output_max - output_min)/(input_max - input_min) = (1-(-1))/(180-(-180))
    def scale_steering(data):
        return data * steering_slope

    target_transform = transforms_v2.Compose([
        lambda x: scale_steering(x)
    ])

    return transform, target_transform

def prepare_dataloader(training_dir, test_dir, batch_size, augment_data):
    '''
    Creates the dataloaders for the training and test datasets, based on the lazy loading dataset class.  
    The dataset class has to lazy load the data because the dataset is too large to fit in memory.
    args:
        training_dir (str): The directory containing the training data
        test_dir (str): The directory containing the test data
        batch_size (int): The batch size for the dataloaders (the dataloader has other parameters that are pulled from the config)
        augment_data (bool): Whether to augment with setup_transforms

        returns:
            train_loader (DataLoader): The dataloader for the training dataset
            test_loader (DataLoader): The dataloader for the test
    '''
    transform, target_transform = setup_transforms(augment_data=augment_data)
    test_transform, test_target_transform = setup_transforms(augment_data=False)
    train_dataset = LazyDataset(training_dir, transform=transform, target_transform=target_transform)
    test_dataset = LazyDataset(test_dir, transform=test_transform, target_transform=test_target_transform)

    # load the data, can implement batching etc. here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, prefetch_factor=5, sampler=DistributedSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, prefetch_factor=5, sampler=DistributedSampler(test_dataset))

    return train_loader, test_loader
