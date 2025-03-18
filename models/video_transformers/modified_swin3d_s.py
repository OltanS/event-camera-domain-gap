import torch
import torch.nn as nn
from functools import partial
from torchvision.models.video.swin_transformer import swin3d_s, PatchEmbed3d

class ModifiedSwin3DS(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        # Patch embed 3D assumes 3 input channels, we then must pass our own version that passes in channels
        patch_embed_with_channels = partial(PatchEmbed3d, in_channels=in_channels)
        self.model = swin3d_s(weights=None, num_classes=out_features, patch_embed=patch_embed_with_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
