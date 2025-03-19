import torch
import torch.nn as nn
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv2Plus1D, R2Plus1dStem

from models.attention_modules.cbam import CBAM

class BasicBlockCBAM(BasicBlock):
    '''
    Resnet BasicBlock with CBAM added
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2[0][-1].out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        # CBAM expects b, c, h, w
        # But we have b, c, t, h, w
        b, c, t, h, w = out.shape
        out = out.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        out = self.cbam(out) # the change
        # reshape back to b, t, c, h, w
        out = out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)        

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class R2Plus1dCBAM(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for APS or DVS, 2 for combined sensor data).
            out_features (int): Number of regression outputs.
        """
        super().__init__()
        layers = [2, 2, 2, 2]
        stem = R2Plus1dStem
    
        self.model = VideoResNet(
            block = BasicBlockCBAM,
            conv_makers=[Conv2Plus1D] * 4,
            layers=layers,
            stem=stem,
            num_classes=out_features
        )
        # Modify the first conv layer to support custom in_channels.
        # The original conv is located in self.model.stem[0].
        self.model.stem[0] = nn.Conv3d(
            in_channels, 
            self.model.stem[0].out_channels,
            kernel_size=self.model.stem[0].kernel_size,
            stride=self.model.stem[0].stride,
            padding=self.model.stem[0].padding,
            bias=self.model.stem[0].bias is not None
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def _test_r2plus1d_spatiotemporal():
    model = R2Plus1dCBAM(in_channels=2, out_features=1)
    x = torch.randn(16, 2, 8, 224, 224)
    y = model(x)
    assert y.shape == (16, 1), y.shape
    print('R2Plus1dSpatiotemporal test passed')

def main():
    print("Running test for R2Plus1dSpatiotemporal")
    _test_r2plus1d_spatiotemporal()

if __name__ == '__main__':
    main()
