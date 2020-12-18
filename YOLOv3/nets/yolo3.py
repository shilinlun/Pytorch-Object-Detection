import torch
import torch.nn as nn
from torchsummary import summary
from nets.darknet53 import DarkNet53
from nets.darknet53 import conv_bn_leayk
from constant.Config import config

def Conv2d(in_channels,out_channels,kernel_size):
    padding = int((kernel_size-1)/2)
    layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1,inplace=True)
    )
    return layers


def Convolutional_Set(in_channels,out_channels):
    layers = [
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    ]
    return layers

class YoloBody(nn.Module):
    def __init__(self):
        super(YoloBody, self).__init__()
        '''
        out3->fenzhi3->chu3---------------------------------------------------------------------------------------->jieguo3(13)
        out3->fenzhi3->chu3_1->upsampling3--
                                            | concatenate2->fenzhi2->chu2------------------------------------------>jieguo2(26)
        out2--------------------------------                  |
                                                              |->chu2_1->upsampling2--
                                                                                     | concatenate1->fenzhi1->chu1->jieguo1(52)
        out1--------------------------------------------------------------------------
        
        
        
        '''
        self.backbone = DarkNet53()
        self.fenzhi3 = nn.Sequential(*Convolutional_Set(in_channels=1024,out_channels=512))
        self.chu3 = conv_bn_leayk(in_channels=512,out_channels=1024,kernel_size=3)
        self.jieguo3 = nn.Conv2d(in_channels=1024,out_channels=3*(5+config.num_classes),kernel_size=1)

        self.chu3_1 = conv_bn_leayk(in_channels=512, out_channels=256, kernel_size=1)
        self.upsampling3 = nn.Upsample(scale_factor=2,mode='nearest')
        self.fenzhi2 = nn.Sequential(*Convolutional_Set(in_channels=768,out_channels=256))
        self.chu2 = conv_bn_leayk(in_channels=256,out_channels=512,kernel_size=3)
        self.jieguo2 = nn.Conv2d(in_channels=512,out_channels=3*(5+config.num_classes),kernel_size=1)

        self.chu2_1 = conv_bn_leayk(in_channels=256, out_channels=128, kernel_size=1)
        self.upsampling2 = nn.Upsample(scale_factor=2,mode='nearest')
        self.fenzhi1 = nn.Sequential(*Convolutional_Set(in_channels=384,out_channels=256))
        self.chu1 = conv_bn_leayk(in_channels=256,out_channels=512,kernel_size=3)
        self.jieguo1 = nn.Conv2d(in_channels=512,out_channels=3*(5+config.num_classes),kernel_size=1)





    def forward(self, x):
        '''
        :param x:
        :return: jieguo1:52,jieguo2:26,jieguo3:13
        '''
        out1,out2,out3 = self.backbone(x)
        fenzhi3 = self.fenzhi3(out3)
        chu3 = self.chu3(fenzhi3)
        jieguo3 = self.jieguo3(chu3)
        chu3_1 = self.chu3_1(fenzhi3)
        upsampling3 = self.upsampling3(chu3_1)
        concatenate2 = torch.cat([upsampling3,out2],dim=1)
        fenzhi2 = self.fenzhi2(concatenate2)
        chu2 = self.chu2(fenzhi2)
        jieguo2 = self.jieguo2(chu2)
        chu2_1 = self.chu2_1(fenzhi2)
        upsampling2 = self.upsampling2(chu2_1)
        concatenate1 = torch.cat([upsampling2,out1],dim=1)
        fenzhi1 = self.fenzhi1(concatenate1)
        chu1 = self.chu1(fenzhi1)
        jieguo1 = self.jieguo1(chu1)
        return jieguo3,jieguo2,jieguo1



# net = YoloBody()
# net.cuda()
# summary(net,(3,416,416))