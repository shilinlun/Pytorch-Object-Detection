import torch
import torch.nn as nn
from torchsummary import summary


cfg = {
    'layer0':[32],
    'layer1':['M',64],
    'layer2':['M',128,54,128],
    'layer3':['M',256,128,256],
    'layer4':['M',512,256,512,256,512],
    'layer5':['M',1024,512,1024,512,1024]
}

def conv_bn_leaky(in_channels,out_channels,kernel_size):
    # 为了保证前后的size不变
    padding = int((kernel_size-1)/2)
    layers = [nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding),
              nn.BatchNorm2d(out_channels)]
    normal_init(layers, 0, 0.01) # 初始化权重
    layers += [nn.LeakyReLU(0.1,inplace=True)]
    return layers



class Darket19(nn.Module):
    def __init__(self):
        super(Darket19, self).__init__()

        self.in_channels = 3
        self.layer0 = nn.Sequential(*self.make_layers(cfg['layer0']))
        self.layer1 = nn.Sequential(*self.make_layers(cfg['layer1']))
        self.layer2 = nn.Sequential(*self.make_layers(cfg['layer2']))
        self.layer3 = nn.Sequential(*self.make_layers(cfg['layer3']))
        self.layer4 = nn.Sequential(*self.make_layers(cfg['layer4']))
        self.layer5 = nn.Sequential(*self.make_layers(cfg['layer5']))


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


    def make_layers(self,cfg):
        layers = []

        kernel_size = 3
        for i in cfg:
            if(i=='M'):
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += conv_bn_leaky(in_channels=self.in_channels,out_channels=i,kernel_size=kernel_size)
                # 由darket19结构可知，kernel_size是3和1交替变化
                kernel_size = 1 if kernel_size ==3 else 3
                self.in_channels = i

        return layers

def normal_init(m, mean, stddev, truncated=False):
    for i in m:
        if truncated:
            i.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            i.weight.data.normal_(mean, stddev)
            i.bias.data.zero_()

# net = Darket19()
# # device = torch.device('cuda')
# net = net.cuda()
# summary(net,(3,416,416))
