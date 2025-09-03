import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResnet(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        self.model = models.resnet34(num_classes=out_features)
        self.model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.tanh = nn.Tanh()
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.tanh(x)
        return x

