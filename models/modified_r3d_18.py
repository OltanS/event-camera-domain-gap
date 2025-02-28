import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class ModifiedR3D18(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for APS or DVS, 2 for combined sensor data).
            out_features (int): Number of regression outputs.
            pretrained (bool): If True, load pretrained weights.
        """
        super(ModifiedR3D18, self).__init__()
        # Load the base model
        self.model = r3d_18(weights=None, num_classes=out_features)
        
        # Modify the first conv layer to support custom in_channels.
        # The original conv is located in self.model.stem[0].
        old_conv = self.model.stem[0]
        if in_channels != old_conv.in_channels:
            self.model.stem[0] = nn.Conv3d(
                in_channels, 
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input tensor of shape [batch, channels, frames, height, width].
        Returns:
            Tensor: Regression output.
        """
        return self.model(x)

# Example usage:
# model = ModifiedR3D18Regression(in_channels=2, out_features=1, pretrained=True)
# dummy_input = torch.randn(1, 2, 8, 112, 112)  # batch_size=1, 2-channel input, 8 frames, 112x112 resolution
# output = model(dummy_input)
# print(output.shape)  # Expected output shape: [1, 1]