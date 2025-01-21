from models.resnet_rnn import ResnetRNN

class ResnetRNNStated(ResnetRNN):
    def __init__(self, block, layers, num_channels, num_classes=1):
        super().__init__(block, layers, num_channels, num_classes)
        self.hidden = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Detach hidden state to prevent backpropagating through the entire history
        if self.hidden is not None:
            self.hidden = tuple([h.detach() for h in self.hidden])

        x, self.hidden = self.lstm(x.unsqueeze(0), self.hidden)
        x = self.fc(x.squeeze(0))

        return x.squeeze(1)

    def reset_hidden(self):
        self.hidden = None
