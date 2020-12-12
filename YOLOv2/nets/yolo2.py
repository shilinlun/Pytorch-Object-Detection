from torchsummary import summary
import torch
import torch.nn as nn
from nets.darket19 import Darket19
from nets.darket19 import conv_bn_leaky
from Constant.Config import config


class Passthrough_layer(nn.Module): # passthrough
    def __init__(self, stride=2):
        super(Passthrough_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x

class yolo2(nn.Module):
    def __init__(self):
        super(yolo2, self).__init__()

        # darket19的前5层
        backbone = Darket19()
        self.conv1 = nn.Sequential(backbone.layer0,backbone.layer1,backbone.layer2,
                                   backbone.layer3,backbone.layer4)
        # darket19的第6层
        self.conv2 = backbone.layer5

        # conv_bn_leaky
        self.conv_bn_leaky = nn.Sequential(*conv_bn_leaky(in_channels=512,out_channels=64,kernel_size=1))

        # passthrough_layer
        self.passthrough_layer = Passthrough_layer()

        # 2层conv_bn_leaky
        self.conv3 = nn.Sequential(
            nn.Sequential(*conv_bn_leaky(in_channels=1024,out_channels=1024,kernel_size=3)),
            nn.Sequential(*conv_bn_leaky(in_channels=1024,out_channels=1024,kernel_size=3))
        )

        # 最后的conv
        self.conv4 = nn.Sequential(
            nn.Sequential(*conv_bn_leaky(in_channels=1280,out_channels=1024,kernel_size=3)),
            nn.Conv2d(in_channels=1024,out_channels=(5+config.num_classes)*config.num_anchors,kernel_size=1)
        )



    def forward(self, x):

        x = self.conv1(x)
        # 经过一个conv_bn_leaky
        temp = self.conv_bn_leaky(x)
        # 经过一个passthrough_layer
        temp = self.passthrough_layer(temp)

        x = self.conv2(x)
        # 2层conv_bn_leaky
        x = self.conv3(x)

        x = torch.cat([temp,x],dim=1)
        x = self.conv4(x)
        # print(x.shape) torch.Size([2, 55, 13, 13])
        return x

# net = yolo2()
# # device = torch.device('cuda')
# net = net.cuda()
# summary(net,(3,416,416))