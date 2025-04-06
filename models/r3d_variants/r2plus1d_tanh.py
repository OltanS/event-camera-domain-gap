import torch
import torch.nn as nn
from models.r3d_variants.modified_r2plus1d import ModifiedR2Plus1d18

class R2Plus1d18tanh(ModifiedR2Plus1d18):
        def __init__(self, in_channels=2, out_features=1):
            """
            Args:
                in_channels (int): Number of input channels (e.g., 1 for APS or DVS, 2 for combined sensor data).
                out_features (int): Number of regression outputs.
            """
            super().__init__(in_channels=in_channels, out_features=out_features)
            self.tanh = nn.Tanh()
        
        def forward(self, x):
              x = self.model(x)
              x = self.tanh(x)
              return x