import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim, reduction_factor=4, dropout=0.2):
        super().__init__()

        # a bottleneck to reduce the dimensionality of the embeddings
        embed_dim = embed_dim
        reduced_dim = embed_dim // reduction_factor

        self.aps_reduce = nn.Linear(embed_dim, reduced_dim)
        self.dvs_reduce = nn.Linear(embed_dim, reduced_dim)

        self.norm_aps_reduce = nn.LayerNorm(reduced_dim)
        self.norm_dvs_reduce = nn.LayerNorm(reduced_dim)

        # Projection layers
        self.aps_gate = nn.Linear(reduced_dim, reduced_dim)
        self.dvs_gate = nn.Linear(reduced_dim, reduced_dim)

        self.aps_fusion = nn.Sequential(
            nn.Linear(embed_dim + reduced_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dvs_fusion = nn.Sequential(
            nn.Linear(embed_dim + reduced_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, aps_features, dvs_features):
        aps_reduced = self.norm_aps_reduce(self.aps_reduce(aps_features))
        dvs_reduced = self.norm_dvs_reduce(self.dvs_reduce(dvs_features))
        # cross-modal gating
        aps_gate = torch.sigmoid(self.aps_gate(dvs_reduced))
        dvs_gate = torch.sigmoid(self.dvs_gate(aps_reduced))
        # pass the "relevant" information from one modality to the other
        aps_context = aps_gate * dvs_reduced
        dvs_context = dvs_gate * aps_reduced

        enchanced_aps = self.aps_fusion(torch.cat([aps_features, aps_context], dim=1))
        enchanced_dvs = self.dvs_fusion(torch.cat([dvs_features, dvs_context], dim=1))

        # residual connection
        enchanced_aps = enchanced_aps + aps_features
        enchanced_dvs = enchanced_dvs + dvs_features

        return enchanced_aps, enchanced_dvs

class R3DParallel(nn.Module):
    '''
    Two R3D models run in parallel with late fusion for regression
    one model tackles the APS data and the other tackles the DVS data
    '''
    def __init__(self, out_features=1):
        """
        Args:
            out_features (int): Number of regression outputs.
        """
        super(R3DParallel, self).__init__()
        aps_base_model = r3d_18(weights=None)
        dvs_base_model = r3d_18(weights=None)
        
        # Modify the first conv layer to support custom in_channels.
        # The original conv is located in self.model.stem[0].
        in_channels = 1
        for m in [aps_base_model, dvs_base_model]:
            old_conv = m.stem[0]
            if in_channels != old_conv.in_channels:
                m.stem[0] = nn.Conv3d(
                    in_channels, 
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
            nn.init.kaiming_normal_(m.stem[0].weight, mode="fan_out", nonlinearity="relu")
            if m.stem[0].bias is not None:
                nn.init.constant_(m.stem[0].bias, 0)


        # remove the fc layer
        self.aps_model = nn.Sequential(*list(aps_base_model.children())[:-1])
        self.dvs_model = nn.Sequential(*list(dvs_base_model.children())[:-1])

        feature_dim = 512

        self.cross_modal_fusion = CrossModalFusion(embed_dim=feature_dim, reduction_factor=4, dropout=0.2)

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        # first channel is APS, second channel is DVS
        # data we have is (batch, channels, frames, height, width)
        aps_input = x[:, 0:1]
        dvs_input = x[:, 1:2]

        aps_features = self.aps_model(aps_input)
        dvs_features = self.dvs_model(dvs_input)

        b, f = aps_features.shape[0], aps_features.shape[1]
        aps_features = aps_features.view(b, -1)
        dvs_features = dvs_features.view(b, -1)

        enchanced_aps, enchanced_dvs = self.cross_modal_fusion(aps_features, dvs_features)

        combined_features = torch.cat((enchanced_aps, enchanced_dvs), dim=1)

        return self.regressor(combined_features)