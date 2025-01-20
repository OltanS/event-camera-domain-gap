import torch.nn as nn
import math

class ResnetRNN(nn.Module):
    def __init__(self, block, layers, num_channels, num_classes = 1):
        super(ResnetRNN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(128, num_classes)
        self._init_weights()
    
    # initialise as per the DDD20 Paper.
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nin = m.in_channels if hasattr(m, 'in_channels') else m.in_features
                m.weight.data.normal_(0, math.sqrt(2.0 / nin))
                if m.bias is not None:
                    m.bias.data.zero_()  

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # rint(f"Conv1 shape is {x.shape}")
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        # print(f"shape of x after avgpool is {x.shape}")
        # x = x.view(x.size(0), -1)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        '''
        Sequence length is the "batch size" as is used everywhere else.
        For each "batch", we must get the output of the LSTM for that batch.
        We will then feed those outputs as the respective outputs of the batches to the fc layer.
        '''
        lstm_out, _ = self.lstm(x)
        # output should be of size (sequence_size, batch_size, 512)
        # print(f"shape of lstm_out is {lstm_out.shape}")
        x = lstm_out.squeeze(1)
        x = self.fc(x)

        return x.squeeze(1)
