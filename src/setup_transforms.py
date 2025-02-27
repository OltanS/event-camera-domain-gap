import torchvision.transforms.v2 as transforms

def setup_transforms(augment_data: bool):
    inv_apsdvs_max = 1 / 255
    def normalise_dvsaps(data):
        # shorthand for (data - self.apsdvs_min) / (self.apsdvs_max - self.apsdvs_min) since we know the min is 0
        return data * inv_apsdvs_max

    transform_list = [
        lambda x: normalise_dvsaps(x),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]

    if augment_data:
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