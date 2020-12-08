import torch
import torch.nn as nn
from Constant.Config import config
import torch.nn.init as init
import torch.nn.functional as F
from nets.vgg16 import vgg as add_vgg
from nets.extras import extras as add_extras
from torch.autograd import Variable
import numpy as np
import math

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = config.min_dim
        self.num_priors = len(config.aspect_ratios)
        self.variance = config.variance or [0.1]
        self.feature_maps = config.feature_maps
        self.min_sizes = config.min_sizes
        self.max_sizes = config.max_sizes
        self.steps = config.steps
        self.aspect_ratios = config.aspect_ratios
        self.clip = config.clip
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            x,y = np.meshgrid(np.arange(f),np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)
            for i, j in zip(y,x):
                f_k = self.image_size / self.steps[k]
                # 计算网格的中心
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 求短边
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 求长边
                s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 获得长方形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*math.sqrt(ar), s_k/math.sqrt(ar)]
                    mean += [cx, cy, s_k/math.sqrt(ar), s_k*math.sqrt(ar)]
        # 获得所有的先验框
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class SSD(nn.Module):
    def __init__(self,vgg,extras,huigui_layers,fenlei_layers,num_classes):
        super(SSD, self).__init__()
        self.vgg = nn.Sequential(*vgg)
        self.extras = nn.Sequential(*extras)
        self.huigui_layers = nn.Sequential(*huigui_layers)
        self.fenlei_layers = nn.Sequential(*fenlei_layers)
        self.num_classes = num_classes
        self.L2Norm = L2Norm(512,20) # 38x38x512这一层出来之后先进行了一个Normalization
        self.config = config
        self.priors = PriorBox(self.config) # 先验框
        with torch.no_grad():
            self.priors = Variable(self.priors.forward())





    def forward(self, x):

        feizhis = list()
        huiguis = list()
        fenleis = list()

        # vgg前面23层结束后，就到了第一个分支
        for i in range(23):
            x = self.vgg[i](x)
        x = self.L2Norm(x)
        feizhis.append(x)

        # vgg从23到最后，是第二个分支
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        feizhis.append(x)


        # 获得后面的内容,添加额外层的第1，3，5，7
        for index, content in enumerate(self.extras):

            '''
            print(index,content)
            0 Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
            1 Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            2 Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
            3 Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            4 Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            5 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
            6 Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            7 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
            '''
            x = F.relu(content(x), inplace=True)
            if index % 2 == 1:
                feizhis.append(x)
        # -----------------------------------------------------
        # 这时候sources中就有了6个分支，后面就是对6个分支分别添加分类和回归
        
        # zip https://www.runoob.com/python/python-func-zip.html
        for (x,fenlei,huigui) in zip(feizhis,self.fenlei_layers,self.huigui_layers):
            huiguis.append(huigui(x).permute(0,2,3,1).contiguous())
            fenleis.append(fenlei(x).permute(0,2,3,1).contiguous())
        # 这样就对每一个分支添加了对应的分类和回归
        # 这里是对其进行resize
        
        huiguis = torch.cat([o.view(o.size(0), -1) for o in huiguis], 1)
        fenleis = torch.cat([o.view(o.size(0), -1) for o in fenleis], 1)
        
        output = (
            huiguis.view(huiguis.size(0),-1,4),
            fenleis.view(fenleis.size(0),-1,self.num_classes),
            self.priors
        )

        return output

num_anchors = [4,6,6,6,4,4]

def myssd(num_classes):
    vgg = add_vgg(in_channel=3)
    extras = add_extras(in_channel=1024)

    '''
    print(extras)
    [Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)), 
    Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), 
    Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)), 
    Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), 
    Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)), 
    Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)), 
    Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)), 
    Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))]
    '''
    
    fenlei_layers = []
    huigui_layers = []

    vgg_source = [21,-2]
    for index,content in enumerate(vgg_source):
        fenlei_layers += [nn.Conv2d(vgg[content].out_channels,num_anchors[index]*num_classes,kernel_size=3,padding=1)]
        huigui_layers += [nn.Conv2d(vgg[content].out_channels,num_anchors[index]*4,kernel_size=3,padding=1)]
        
    for index,content in enumerate(extras[1::2],start=2):
        huigui_layers += [nn.Conv2d(content.out_channels,num_anchors[index]*4,kernel_size=3,padding=1)]
        fenlei_layers += [nn.Conv2d(content.out_channels,num_anchors[index]*num_classes,kernel_size=3,padding=1)]
        
    MODEL = SSD(vgg,extras,huigui_layers,fenlei_layers,num_classes)
    return MODEL