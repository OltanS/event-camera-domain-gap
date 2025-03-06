import torch
import torch.nn as nn

from models.modified_mvit_v2_s import ModifiedMViTv2S

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
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dvs_fusion = nn.Sequential(
            nn.Linear(embed_dim + reduced_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
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


class MViTv2SParallel(nn.Module):
    def __init__(self, out_features=1):
        '''
        Creates two parallel mvitv2s models, which process APS and DVS data separately.
        They have cross-modal attention between them.

        mvit itself has spatial and temporal attention, but we can add cross-modal attention.
        '''

        super().__init__()
        aps_base = ModifiedMViTv2S(in_channels=1, out_features=out_features)
        dvs_base = ModifiedMViTv2S(in_channels=1, out_features=out_features)

        self.aps_extractor = nn.Sequential(*list(aps_base.children())[:-1])
        self.dvs_extractor = nn.Sequential(*list(dvs_base.children())[:-1])

        # output channels of last layer of mvitv2s
        feature_dim = 768

        self.cross_modal_fusion = CrossModalFusion(embed_dim=feature_dim, reduction_factor=4, dropout=0.2)

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        # first channel is APS, second channel is DVS
        # data we have is (batch, channels, frames, height, width)
        aps_input = x[:, 0:1]
        dvs_input = x[:, 1:2]

        aps_features = self.aps_extractor(aps_input) #[B, 768]
        dvs_features = self.dvs_extractor(dvs_input) #[B, 768]

        enchanced_aps, enchanced_dvs = self.cross_modal_fusion(aps_features, dvs_features)

        combined_features = torch.cat((enchanced_aps, enchanced_dvs), dim=1)

        return self.regressor(combined_features)
