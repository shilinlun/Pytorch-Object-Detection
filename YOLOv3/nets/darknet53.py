import torch
import torch.nn as nn
from torchsummary import summary
import math

cfg = {
    '1' : 64,
    '2' : 128,
    '3' : 256,
    '4' : 512,
    '5' : 1024
}

def conv_bn_leayk(in_channels,out_channels,kernel_size,stride=1):
    padding = int((kernel_size-1)/2)
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.1,inplace=True))
    return nn.Sequential(*layers)

class BasicLayers(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BasicLayers, self).__init__()
        self.conv1 = conv_bn_leayk(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.conv2 = conv_bn_leayk(in_channels=out_channels,out_channels=in_channels,kernel_size=3)

    def forward(self, x):
        temp = x

        x = self.conv1(x)
        x = self.conv2(x)

        x += temp

        return x

class Darket53(nn.Module):
    def __init__(self,list):
        super(Darket53, self).__init__()
        self.out_channels = 32
        self.conv_bn_leaky = conv_bn_leayk(in_channels=3,out_channels=32,kernel_size=3)
        self.layer1 = self._make_layers(1, list[0])
        self.layer2 = self._make_layers(2, list[1])
        self.layer3 = self._make_layers(3, list[2])
        self.layer4 = self._make_layers(4, list[3])
        self.layer5 = self._make_layers(5, list[4])

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self,i,num):
        layers = []
        layers += conv_bn_leayk(in_channels=self.out_channels,out_channels=cfg['{}'.format(i)],kernel_size=3,stride=2)
        self.out_channels = cfg['{}'.format(i)]
        for j in range(num):
            layers += [BasicLayers(in_channels=cfg['{}'.format(i)],out_channels=int(cfg['{}'.format(i)]/2))] #要加一个[]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_leaky(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out1 = x
        x = self.layer4(x)
        out2 = x
        x = self.layer5(x)
        out3 = x
        return out1,out2,out3

def DarkNet53():
    net = Darket53([1, 1, 1, 1, 1])
    return net

# net = Darket53([1,2,8,8,4])
# net.cuda()
# summary(net,(3,416,416))