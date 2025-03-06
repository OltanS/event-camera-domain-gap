from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.video import MViT
from typing import List, Dict

# A great deal of copying has been done here from the original class at https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py, due to mvit_v2_s forcing temporal_size = 16 even without pretrained...


@dataclass
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]

class ModifiedMViTv2S(nn.Module):
    def __init__(self, in_channels=2, out_features=1):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for APS or DVS, 2 for combined sensor data).
            out_features (int): Number of regression outputs.
        """
        super().__init__()
        
        config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
        }

        block_setting = []
        for i in range(len(config["num_heads"])):
            block_setting.append(
                MSBlockConfig(
                    num_heads=config["num_heads"][i],
                    input_channels=config["input_channels"][i],
                    output_channels=config["output_channels"][i],
                    kernel_q=config["kernel_q"][i],
                    kernel_kv=config["kernel_kv"][i],
                    stride_q=config["stride_q"][i],
                    stride_kv=config["stride_kv"][i],
                )
            )
        

        self.model =  MViT(
        spatial_size=(224, 224),
        temporal_size=8, # all for this change
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=0.2,
        num_classes=out_features # and this
        )
  
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

        # Add proper initialization for the new conv layer
        nn.init.kaiming_normal_(self.model.conv_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.model.conv_proj.bias is not None:
            nn.init.constant_(self.model.conv_proj.bias, 0)
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, frames, height, width].
        Returns:
            Tensor: Regression output.
        """
        return self.model(x)
