import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        # holds temporal attention weights
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        
        x_no_time = x.view(b * t, c, h, w)
        # Spatial pool to get frame features
        frame_features = F.adaptive_avg_pool2d(x_no_time, output_size=1).view(b, t, c)
        # Get attention weights
        attention_weights = self.mlp(frame_features)
        attention_weights = F.softmax(attention_weights, dim=1)

        weighted_frames = (x * attention_weights.view(b, t, c, 1, 1)).sum(dim=1)
        return weighted_frames