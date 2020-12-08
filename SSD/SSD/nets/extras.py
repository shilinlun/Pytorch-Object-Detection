import torch
import torch.nn as nn


def extras(in_channel):
    extras_layers = []
    in_channels = in_channel

    # 19,19,1024 -->10,10,512
    extras_layers += [nn.Conv2d(in_channels=in_channels,out_channels=256,kernel_size=1,stride=1)]
    extras_layers += [nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1)]

    # 10,10,512 -->5,5,256
    extras_layers += [nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1,stride=1)]
    extras_layers += [nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)]

    # 5,5,256 -->3,3,256
    extras_layers += [nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1)]
    extras_layers += [nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1)]

    # 3,3,256 -->1,1,256
    extras_layers += [nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1)]
    extras_layers += [nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1)]


    return extras_layers