import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResnet(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        self.model = models.resnet34()
        self.model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        # get only last one, for testing
        last_timeslots = x[:, -1, :, :, :]
        return self.model(last_timeslots)