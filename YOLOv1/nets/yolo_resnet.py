import torch
from torchsummary import summary
import torch.nn as nn
import math
import torchvision.models as tvmodel
from Constant.Config import config


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels*4,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels*4,kernel_size=1,stride=stride)
        self.bn4 = nn.BatchNorm2d(out_channels*4)



    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # 这个块有残差结构，有下采样
        residual = self.conv4(residual)
        residual = self.bn4(residual)
        x += residual
        x = self.relu(x)
        return x



class IdentityBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels*4,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # 这个块有残差结构,但没有下采样
        x += residual

        x = self.relu(x)
        return x

class Resnet(nn.Module):
    def __init__(self,list_layers,num_class=1000):
        super(Resnet, self).__init__()
        self.chushi_in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True) # inplace 这样能够节省运算内存，不用多存储其他变量
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True) # ceil_mode https://www.cnblogs.com/xxxxxxxxx/p/11529343.html
        self.layer1 = self._getlayers(list_layers[0],out_channel=64)
        self.layer2 = self._getlayers(list_layers[1],out_channel=128,stride=2)
        self.layer3 = self._getlayers(list_layers[2],out_channel=256,stride=2)
        self.layer4 = self._getlayers(list_layers[3],out_channel=512,stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048,num_class)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x


    def _getlayers(self,num_layers,out_channel,stride=1):
        layers = []
        layers.append(ConvBlock(in_channels=self.chushi_in_channel,out_channels=out_channel,stride=stride))
        self.chushi_in_channel = out_channel * 4
        for i in range(1,num_layers):
            layers.append(IdentityBlock(self.chushi_in_channel,out_channel))
        return nn.Sequential(*layers)


def resnet():
    net = Resnet([3, 4, 6, 3])

    return net



class Yolo_resnet(nn.Module):
    def __init__(self):
        super(Yolo_resnet, self).__init__()
        # 原论文是使用GoogleNet作为主干网络，我们这里使用resnet作为主干网络，除了我们自己写的resnet，我们可以使用torchvision自带的resnet
        # net = tvmodel.resnet50(pretrained=False)
        # 为了和yolo连接上,我们去除最后两层,也就是下面的,然后再连接上YOLOv1的最后4个卷积层和两个全连接层，作为我们训练的网络结构
        '''
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        '''
        jichuresnet = tvmodel.resnet18(pretrained=False)
        resnet_out_channel = jichuresnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(jichuresnet.children())[:-2])  # 去除resnet的最后两层
        self.yolo_conv = nn.Sequential(
            nn.Conv2d(in_channels=resnet_out_channel,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.yolo_fc = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 16),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )


    def forward(self, x):
        x = self.resnet(x)
        x = self.yolo_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.yolo_fc(x)
        x = x.reshape(-1, (5 * config.num_boxes + config.num_classes), 7, 7)
        return x


def yolo_resnet():  # 使用tvmodel构建的resnet加上yolo的层
    net = Yolo_resnet()
    return net

# net = yolo_resnet()
# device = torch.device('cuda')
# net = net.to(device)
# summary(net,(3,448,448))