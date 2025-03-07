import glob
import os

import torch
import torch.nn as nn
from torchvision import models
from models.resnet_rnn import ResnetRNN
from models.modified_r3d_18 import ModifiedR3D18
from models.modified_swin3d_s import ModifiedSwin3DS
from models.modified_mvit_v2_s import ModifiedMViTv2S
from models.simple_model import SimpleModel
from models.r3d_18_parallel import R3DParallel
from models.mvit_v2_s_parallel import MViTv2SParallel

def setup_model(MODEL_TYPE, MODEL_DIMS, device):
    '''
    Sets up the model based on the given parameters
    args:
        MODEL_TYPE: str, the type of model to use, see config for options
        MODEL_DIMS: int, the number of input channels, 2 for DVS + APS, 1 for DVS/APS only
        device: str, the device to use
    returns:
        model: nn.Module, the model to use
    '''
    if MODEL_TYPE == "Resnet":
        model = models.resnet34()
        model.conv1 = nn.Conv2d(in_channels=MODEL_DIMS, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        model.to(device)

    elif MODEL_TYPE == "ResnetRNN":
        model = ResnetRNN(hidden_dim=128, num_layers=1, in_channels=MODEL_DIMS).to(device)

    elif MODEL_TYPE == "R3D":
        model = ModifiedR3D18(in_channels=MODEL_DIMS, out_features=1).to(device)
    
    elif MODEL_TYPE == "Swin":
        model = ModifiedSwin3DS(in_channels=MODEL_DIMS, out_features=1).to(device)

    elif MODEL_TYPE == "MViTV2":
        model = ModifiedMViTv2S(in_channels=MODEL_DIMS, out_features=1).to(device)

    # only makes sense for 2 channels, so no in_channels parameter
    elif MODEL_TYPE == "R3DParallel":
        model = R3DParallel(out_features=1).to(device)

    # only makes sense for 2 channels, so no in_channels parameter
    elif MODEL_TYPE == "MviTV2Parallel":
        model = MViTv2SParallel(out_features=1).to(device)

    elif MODEL_TYPE == "simple":
        model = SimpleModel().to(device)
    else:
        raise Exception(f"There is no defined model for {MODEL_TYPE}!")
    
    return model

def warm_start(model, optimizer, local_rank, last_model_dir):
    """
    Load the last model and optimizer state from the last saved model.
    args:
        model: nn.Module, the model to load the state into
        optimizer: torch.optim, the optimizer to load the state into
        local_rank: int, the local rank
        last_model_dir: str, the directory where the last model is saved
    returns:
        epoch: int, the epoch to start from
        best_test_loss: float, the best
    """
    loc = f"cuda:{local_rank}"
    last_model_file = glob.glob(f"{last_model_dir}/*")[-1]
    load_dict = torch.load(last_model_file, map_location=loc, weights_only=False)
    model.load_state_dict(load_dict["model_state_dict"])
    optimizer.load_state_dict(load_dict["optimizer_state_dict"])
    epoch = load_dict["epoch"]
    best_test_loss = load_dict["best_test_loss"]
    print(f"Last completed epoch was {epoch + 1}, with best test loss {(best_test_loss ** 0.5) * 180} from file: {last_model_file}")
    return epoch + 1, best_test_loss

def clean_models(directory):
    '''
    Delete the models in a given directory.
    args:
        directory: str, the directory
    '''
    for model_file in glob.glob(directory + "/*.pth"):
        os.remove(model_file)

def save_model(directory, model, optimizer, epoch, loss, best_test_loss, *, clean_old=False):
    '''
    Saves the model in the given location.
    args:
        directory: str, the directory to save the model
        model: nn.Module, the model to save
        optimizer: torch.optim, the optimizer to save
        epoch: int, the epoch of the model
        loss: float, the loss of the model
        best_test_loss: float, the best test loss
        clean_old: bool, whether to clean the old models
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