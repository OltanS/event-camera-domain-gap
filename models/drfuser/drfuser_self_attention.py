# Taken from https://github.com/azamshoaib/DRFuser for benchmarking
# Adapted to work with Event Frame + APS instead of RGB by changing head input channels to 1 for both models

import torch
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F

from models.drfuser.drfuser_attention_block import AttentionConv, AttentionStem


class DRFuser_self_attention(nn.Module):
    
    def __init__(self):
        super(DRFuser_self_attention, self).__init__()
        resnet_raw_model1 = models.resnet50()
        resnet_raw_model2 = models.resnet50()
        ########  Event ENCODER  ########
        self.encoder_event_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_event_bn1 = resnet_raw_model1.bn1
        self.encoder_event_relu = resnet_raw_model1.relu
        self.encoder_event_maxpool = resnet_raw_model1.maxpool
        self.encoder_event_layer1 = resnet_raw_model1.layer1
        self.encoder_event_layer2 = resnet_raw_model1.layer2
        self.encoder_event_layer3 = resnet_raw_model1.layer3
        self.encoder_event_layer4 = resnet_raw_model1.layer4
        ########  APS ENCODER  ########
        self.encoder_aps_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_aps_bn1 = resnet_raw_model2.bn1
        self.encoder_aps_relu = resnet_raw_model2.relu
        self.encoder_aps_maxpool = resnet_raw_model2.maxpool
        self.encoder_aps_layer1 = resnet_raw_model2.layer1
        self.encoder_aps_layer2 = resnet_raw_model2.layer2
        self.encoder_aps_layer3 = resnet_raw_model2.layer3
        self.encoder_aps_layer4 = resnet_raw_model2.layer4

        ########  Self-attention  ########
        #elif self.args.resnet == 50:
        self.Att1=AttentionConv(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.Att2=AttentionConv(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.Att3=AttentionConv(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.Att4=AttentionConv(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)
        self.Att5=AttentionConv(in_channels=2048,out_channels=2048,kernel_size=3,padding=1)

        ########  DECODER  ########
        #elif self.args.resnet ==50:					
        self.join = nn.Sequential(
                        nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=0, bias=False), 
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False), 
                        nn.BatchNorm2d(512),
                        nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=0, bias=False), 
                        nn.BatchNorm2d(256),
                        nn.Dropout(0.2),
                        nn.ReLU(),)

    def forward(self, stacked_aps_dvs):

        aps = stacked_aps_dvs[:, 0:1, :, :]  # [b, 1, h, w]
        event_frame = stacked_aps_dvs[:, 1:2, :, :]  # [b, 1, h, w]

        aps = self.encoder_aps_conv1(aps)
        aps = self.encoder_aps_bn1(aps)
        aps = self.encoder_aps_relu(aps)

        event_frame = self.encoder_event_conv1(event_frame)
        event_frame = self.encoder_event_bn1(event_frame)
        event_frame = self.encoder_event_relu(event_frame)

        aps = self.Att1(aps) + self.Att1(event_frame)
        aps = self.encoder_aps_maxpool(aps)
        event_frame = self.encoder_event_maxpool(event_frame)
        ######################################################################
        aps = self.encoder_aps_layer1(aps)
        event_frame = self.encoder_event_layer1(event_frame)
        aps = self.Att2(aps) + self.Att2(event_frame)
        ######################################################################
        aps = self.encoder_aps_layer2(aps)
        event_frame = self.encoder_event_layer2(event_frame)
        aps = self.Att3(aps) + self.Att3(event_frame)
        ######################################################################
        aps = self.encoder_aps_layer3(aps)
        event_frame = self.encoder_event_layer3(event_frame)
        aps = self.Att4(aps) + self.Att4(event_frame)
        ######################################################################
        aps = self.encoder_aps_layer4(aps)
        event_frame = self.encoder_event_layer4(event_frame)
        fuse = self.Att5(aps) + self.Att5(event_frame)   #([2, 2048, 8, 16])
        ######################################################################
        # decoder
        fuse = self.join(fuse)
        # fuse=self.avgpool(fuse)
        fuse=torch.flatten(fuse, 1) 
        if not hasattr(self, 'decoder'):
            self.decoder = nn.Sequential(
                nn.Linear(fuse.shape[1], 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1),
            ).to(fuse.device)  # Important: move to same device as fuse
        fuse=self.decoder(fuse)     
        return fuse

def _unit_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_minibatch = 4 * 8
    aps_dvs = torch.randn(num_minibatch, 2, 224, 224).to(device)
    model = DRFuser_self_attention().to(device)
    out = model(aps_dvs)
    assert out.shape == (num_minibatch, 1), f"Output shape mismatch: {out.shape}"

if __name__ == '__main__':
    print("Running unit test for DRFuser_self_attention...")
    _unit_test()
    print("Unit test passed.")