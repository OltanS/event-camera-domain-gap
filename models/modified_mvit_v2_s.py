import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s

class ModifiedMViTv2S(nn.Module):
    def __init__(self, in_channels=1, out_features=1):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for APS or DVS, 2 for combined sensor data).
            out_features (int): Number of regression outputs.
            pretrained (bool): If True, load pretrained weights.
        """
        super(ModifiedMViTv2S, self).__init__()
        # Load the base mvit_v2_s model
        self.model = mvit_v2_s(weights=None, num_classes=out_features, temporal_size=8)
        
        # Modify the patch embedding projection layer to accept the desired number of input channels.
        # The default conv is stored in self.model.patch_embed.proj.
        old_proj = self.model.conv_proj
        if in_channels != old_proj.in_channels:
            self.model.conv_proj = nn.Conv3d(
                in_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None
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
# if __name__ == "__main__":
#     model = ModifiedMViTv2SRegression(in_channels=2, out_features=1, pretrained=False)
#     dummy_input = torch.randn(1, 2, 16, 224, 224)  # batch_size=1, 2-channel, 16 frames, 224x224 resolution
#     output = model(dummy_input)
#     print(output.shape)  # Expected output shape: [1, 1]