import torch
import torch.nn as nn
from functools import partial
from torchvision.models.video.swin_transformer import swin3d_t, PatchEmbed3d

import math

class ModifiedSwin3Dt(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        # Patch embed 3D assumes 3 input channels, we then must pass our own version that passes in channels
        patch_embed_with_channels = partial(PatchEmbed3d, in_channels=in_channels)
        self.model = swin3d_t(weights=None, num_classes=out_features, patch_embed=patch_embed_with_channels)

        self._init_weights()

    def _init_weights(self):
        """Custom initialization to improve training stability"""
        for name, module in self.model.named_modules():
            # Initialize linear layers with small values to prevent early gradient explosion
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            # Normalize layer norm to 1
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            
            # Add specific initialization for patch embedding
            elif isinstance(module, PatchEmbed3d):
                if hasattr(module, "proj"):
                    # Initialize patch embedding with smaller values
                    fan_out = module.proj.kernel_size[0] * module.proj.kernel_size[1] * module.proj.kernel_size[2] * module.proj.out_channels
                    fan_out //= module.proj.groups
                    nn.init.normal_(module.proj.weight, 0, math.sqrt(2.0 / fan_out))
                    if module.proj.bias is not None:
                            nn.init.constant_(module.proj.bias, 0)   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
