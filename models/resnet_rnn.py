import torch.nn as nn
import torchvision.models as models

class ResnetRNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, in_channels=2, pretrained=False):
        super().__init__()
        base_model = models.resnet34(pretrained=pretrained)
        base_model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)

    # override
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(b, t, -1)

        rnn_out, _ = self.rnn(features)
        out = self.fc(rnn_out)
        return out
