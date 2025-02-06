import torch.nn as nn
from torchvision import models
from models.resnet_rnn import ResnetRNN
from models.resnet_rnn_stated import ResnetRNNStated

def setup_model(MODEL_TYPE, MODEL_DIMS, PRETRAINED, device):
    if MODEL_TYPE == "Resnet":
        model = models.resnet34(pretrained=PRETRAINED)
        model.conv1 = nn.Conv2d(in_channels=MODEL_DIMS, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        model.to(device)

    elif MODEL_TYPE == "ResnetRNN":
        model = ResnetRNN(hidden_dim=128, num_layers=1, in_channels=MODEL_DIMS, pretrained=PRETRAINED).to(device)

    elif MODEL_TYPE == "ResnetRNNStated":
        model = ResnetRNNStated(hidden_dim=128, num_layers=1, in_channels=MODEL_DIMS, pretrained=PRETRAINED).to(device)

    else:
        raise Exception(f"There is no defined model for {MODEL_TYPE}!")
    
    return model