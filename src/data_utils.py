import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.v2 as transforms_v2

def setup_transforms(augment_data: bool, use_aps: bool, use_dvs: bool):
    '''
    Prepares the transforms for the dataset.  
    If augment_data is True, the transforms will include random cropping, rotation, and Gaussian noise.  
    The transforms are defined here and nowhere else, so you can modify them from here.
    args:
        augment_data (bool): Whether to augment the data
        use_aps (bool): Whether to use APS data
        use_dvs (bool): Whether to use DVS data
    returns:
        transform (Compose): The transform to apply to the data
        target_transform (Compose): The transform to apply to the target
    '''
    
    # always normalise it for regular use, the option is here for the ablation study
    just_scale = False

    if not use_aps and not use_dvs:
        raise ValueError("At least one of use_aps or use_dvs must be True")
    
    # Define normalization parameters
    norm_params = {
        "aps": {"mean": [123.0759], "std": [88.1677]},
        "dvs": {"mean": [119.1442], "std": [31.6903]},
    }
    
    # Determine which means and stds to use
    if use_aps and use_dvs:
        mean = norm_params["aps"]["mean"] + norm_params["dvs"]["mean"]
        std = norm_params["aps"]["std"] + norm_params["dvs"]["std"]
    elif use_aps:
        mean = norm_params["aps"]["mean"]
        std = norm_params["aps"]["std"]
    else:  # use_dvs only
        mean = norm_params["dvs"]["mean"]
        std = norm_params["dvs"]["std"]

    transform_list = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=False)
    ]
    
    if just_scale:
        transform_list.append(lambda x: x / 255.0)
    else:
        transform_list.append(transforms_v2.Normalize(mean=mean, std=std))
    
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

