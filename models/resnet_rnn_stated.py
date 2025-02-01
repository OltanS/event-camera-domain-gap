from models.resnet_rnn import ResnetRNN

class ResnetRNNStated(ResnetRNN):
    def __init__(self, hidden_dim=128, num_layers=1, in_channels=2, pretrained=False):
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, in_channels=in_channels, pretrained=pretrained)
        self.hidden = None

    def forward(self, x):

        # x shape: (batch, seq_len, channels, height, width)
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(b, t, -1)
        
        # Detach hidden state to prevent backpropagating through the entire history
        if self.hidden is not None:
            self.hidden = tuple([h.detach() for h in self.hidden])

        rnn_out, self.hidden = self.rnn(features, self.hidden)
        out = self.fc(rnn_out)
        return out

    def reset_hidden(self):
        self.hidden = None
