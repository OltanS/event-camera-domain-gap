import torch
import torch.nn as nn
from functools import partial
from torchvision.models.video.swin_transformer import swin3d_s, PatchEmbed3d

import math

class ModifiedSwin3Dt(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        super().__init__()
        # Patch embed 3D assumes 3 input channels, we then must pass our own version that passes in channels
        patch_embed_with_channels = partial(PatchEmbed3d, in_channels=in_channels)
        self.model = swin3d_s(weights=None, num_classes=out_features, patch_embed=patch_embed_with_channels)

        self._init_weights()

    def _init_weights(self):
        """Regression-optimized initialization with patch embedding adjustment"""
        for name, module in self.model.named_modules():
            # Significantly reduce patch embedding initialization magnitude
            if isinstance(module, PatchEmbed3d):
                if hasattr(module, "proj"):
                    # Use much smaller initialization for patch embedding
                    nn.init.trunc_normal_(module.proj.weight, std=0.01)  # Reduced from fan-out approach
                    if module.proj.bias is not None:
                        nn.init.constant_(module.proj.bias, 0)
            
            # Other layer initializations as before
            elif isinstance(module, nn.Linear):
                if "mlp.0" in name:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                elif "head" in name:
                    nn.init.trunc_normal_(module.weight, std=0.01)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
        # Attention initializations as before
        for name, parameter in self.model.named_parameters():
            if "qkv" in name and "weight" in name:
                nn.init.trunc_normal_(parameter, std=0.02)
            elif "relative_position_bias_table" in name:
                nn.init.trunc_normal_(parameter, std=0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
