import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from models.cbam import CBAM
from models.temporal_attention import TemporalAttention

# tiny corresponds to resnet50 equivalent
from torchvision.models.convnext import CNBlock, convnext_tiny, LayerNorm2d
from torchvision.ops.misc import Conv2dNormActivation

# CBAM is put at the end of the basic block after second conv layer, before residual connection
# We can add CBAM to the basic block by subclassing BasicBlock and overriding
# the forward method to include CBAM
class BasicBlockCBAM(CNBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.layer_scale.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.layer_scale * self.block(x)
        x = self.stochastic_depth(x)
        x = self.cbam(x)
        x = input + x
        return x

class ConvNeXtTinyCBAM(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        self.model = convnext_tiny(block=BasicBlockCBAM, num_classes=out_features, layer_scale=1e-1)
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

class ConvNeXtTinyTemporalCBAM(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        convnext_cbam_regressor = ConvNeXtTinyCBAM(in_channels=in_channels, out_features=out_features)
        # we must remove the last layer of the model, apply a temporal attention mechanism, and then add the last layer back
        # very conveniently, the head is a separate layer called "classifier", so we can just take the "features" part of the model
        self.feature_extractor = convnext_cbam_regressor.model.features
        # known via testing/model inspection
        num_features = 768
        self.temporal_attention = TemporalAttention(num_features)
        self.avgpool = convnext_cbam_regressor.model.avgpool
        self.regressor = convnext_cbam_regressor.model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.feature_extractor(x) # b * t, fetures, h_, w_)
        # put back batch and time dimensions
        _, features, h_, w_ = x.shape
        x = x.view(b, t, features, h_, w_)
        x = self.temporal_attention(x) # b, features, h_, w_
        # perform spatial pooling, from (b, features, h_, w_) to (b, t, features)
        x = self.avgpool(x)
        # get only the last time step, enriched with spatial and temporal attention
        x = self.regressor(x)
        return x

def _test_basic_block_cbam():
    model = BasicBlockCBAM(dim=96, layer_scale=1e-6, stochastic_depth_prob=0.1)
    x = torch.randn(32, 96, 64, 64)
    out = model(x)

    model = CNBlock(dim=96, layer_scale=1e-6, stochastic_depth_prob=0.1)
    out_original = model(x)

    assert out.shape == out_original.shape, f"Shape mismatch: {out.shape} != {out_original.shape}"
    print("BasicBlockCBAM test passed")

def _test_convnext_tiny_cbam():
    b, c = 32, 2
    features = 1
    model = ConvNeXtTinyCBAM(in_channels=c, out_features=features)
    x = torch.randn(b, c, 224, 224)
    out = model(x)
    expected_shape = (b, features)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
    print("ConvNeXtTinyCBAM test passed")

def _test_convnext_tiny_temporal_cbam():
    b, t, c = 16, 8, 2
    features = 1
    model = ConvNeXtTinyTemporalCBAM(in_channels=c, out_features=features)
    x = torch.randn(b, t, c, 224, 224)
    out = model(x)
    expected_shape = (b, features)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
    print("ConvNeXtTinyTemporalCBAM test passed")

def main():
    print("Running basic block CBAM test...")
    _test_basic_block_cbam()
    print("Running ConvNeXt Tiny CBAM test...")
    _test_convnext_tiny_cbam()
    print("Running ConvNeXt Tiny Temporal CBAM test...")
    _test_convnext_tiny_temporal_cbam()

if __name__ == "__main__":
    main()