import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from models.cbam import CBAM
from models.temporal_attention import TemporalAttention

class BottleneckCBAM(Bottleneck):
    '''
    Resnet Bottleneck with CBAM added
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv3.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.cbam(x)

        if self.downsample is not None:
            input = self.downsample(input)

        x += input
        x = self.relu(x)

        return x

class ResnetCBAM(nn.Module):
    '''
    A resnet50 model with CBAM added to each block.
    '''
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        layers = [3, 4, 6, 3]
        block = BottleneckCBAM
        self.model = ResNet(block, layers, num_classes=out_features)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        last_timeslots = x[:, -1, :, :, :]
        return self.model(last_timeslots)

class ResnetTemporalCBAM(nn.Module):
    '''
    A resnet50 model with CBAM added to each block and temporal attention added to the end.
    '''
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        base_model = ResnetCBAM(in_channels=in_channels, out_features=out_features)
        self.feature_extractor = nn.Sequential(*list(base_model.model.children())[:-2])
        self.temporal_attention = TemporalAttention(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x_reshaped = x.reshape(b * t, c, h, w)
        
        # Extract features for all time steps at once
        features = self.feature_extractor(x_reshaped)  # [B*T, 2048, H', W']
        # Get the new spatial dimensions
        _, f, h_out, w_out = features.shape
        
        # Reshape back to separate batch and time dimensions
        features = features.reshape(b, t, f, h_out, w_out)  # [B, T, 2048, H', W']
        # Apply temporal attention
        x = self.temporal_attention(features) # [B, 2048, H', W'] aggregated to one frame
        # Global average pooling and final classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet_cbam_test():
    model = ResnetCBAM(in_channels=2, out_features=1)
    x = torch.randn(32, 8, 2, 224, 224)
    out = model(x)

    assert out.shape == (32, 1)
    print("Resnet CBAM test passed")

def _resnet_temporal_cbam_test():
    model = ResnetTemporalCBAM(in_channels=2, out_features=1)
    x = torch.randn(32, 8, 2, 224, 224)
    out = model(x)

    assert out.shape == (32, 1)
    print("Resnet Temporal CBAM test passed")

if __name__ == "__main__":
    print("Running Resnet CBAM test...")
    _resnet_cbam_test()
    print("Running Resnet Temporal CBAM test...")
    _resnet_temporal_cbam_test()
