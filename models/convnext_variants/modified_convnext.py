import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

# tiny corresponds to resnet50 equivalent
from torchvision.models.convnext import convnext_tiny
from torchvision.ops.misc import Conv2dNormActivation

# CBAM is put at the end of the basic block after second conv layer, before residual connection
# We can add CBAM to the basic block by subclassing BasicBlock and overriding
# the forward method to include CBAM

class LayerNorm2d(nn.LayerNorm):
    '''
    taken from convnext, modified with contiguous to work with DDP
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ModifiedConvNeXtTiny(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        self.model = convnext_tiny(num_classes=out_features, layer_scale=1e-1)
        tiny_first_input_channels = 96
        self.model.features[0] = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=tiny_first_input_channels,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=partial(LayerNorm2d, eps=1e-6),
            activation_layer=None,
            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def _test_convnext_tiny():
    model = ModifiedConvNeXtTiny(in_channels=2, out_features=1)
    x = torch.randn(32 * 8, 2, 224, 224)
    out = model(x)

    assert out.shape == (32 * 8, 1)
    print("Convext tiny test passed")

def main():
    _test_convnext_tiny()

if __name__ == "__main__":
    main()