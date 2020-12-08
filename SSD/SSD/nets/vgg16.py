import torch
import torch.nn as nn
from torchsummary import summary

lists = [64,64,'M',128,128,'M',256,256,256,'M_C',512,512,512,'M',512,512,512]

def vgg(in_channel):
    '''
    :param i:  输入图像的通道数目
    :return:   vgg16的layers
    '''
    in_channels = in_channel
    layers = []
    for i in lists:
        if i =='M': # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        elif i =='M_C': # 使用了ceil_mode的最大池化
            layers += [nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels,i,kernel_size=3,padding=1)
            relu = nn.ReLU(inplace=True)
            layers += [conv2d,relu]
            in_channels = i

    # 最大池化
    pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    # 原来的vgg本来是3层FC 这里使用了2层，且FC变为conv
    conv1 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=6,dilation=6)
    relu1 = nn.ReLU(inplace=True)
    conv2 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1)
    relu2 = nn.ReLU(inplace=True)

    layers += [pool,conv1,relu1,conv2,relu2]

    return layers



# def testvgg(in_channel):
#     '''
#     :param i:  输入图像的通道数目
#     :return:   vgg16的layers
#     '''
#     in_channels = in_channel
#     layers = []
#     for i in lists:
#         if i =='M': # 最大池化层
#             layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
#         elif i =='M_C': # 使用了ceil_mode的最大池化
#             layers += [nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels,i,kernel_size=3,padding=1)
#             relu = nn.ReLU(inplace=True)
#             layers += [conv2d,relu]
#             in_channels = i
#
#     # 最大池化
#     pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
#     # 原来的vgg本来是3层FC 这里使用了2层，且FC变为conv
#     conv1 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=6,dilation=6)
#     relu1 = nn.ReLU(inplace=True)
#     conv2 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1)
#     relu2 = nn.ReLU(inplace=True)
#
#     layers += [pool,conv1,relu1,conv2,relu2]
#
#     return nn.Sequential(*layers)


# class MyVGG(nn.Module):
#     def __init__(self):
#         super(MyVGG, self).__init__()
#         self.myvgg = mytestvgg(3)
#
#     def forward(self, x):
#         x = self.myvgg(x)
#         return x
#
# device = torch.device('cuda')
# net = MyVGG()
# net = net.to(device)
# summary(net,(3,300,300))