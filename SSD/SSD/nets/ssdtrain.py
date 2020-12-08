from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Constant.Config import config
from utils.box_utils import match, log_sum_exp
from collections import namedtuple
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import cv2

LossTuple = namedtuple('LossTuple',
                       ['huigui_loss',
                        'fenlei_loss',
                        'total_loss'
                        ])

MEANS = (104, 117, 123)
class Train(nn.Module):
    def __init__(self,model,optimizer):
        super(Train, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.variance = config.variance
        self.num_classes = config.num_classes
        self.threshold = 0.5
        self.negpos_ratio = 3
        self.neg_overlap = 0.5


    def forward(self, predicts, targets):
        # 回归信息，置信度，先验框
        loc_data, conf_data, priors = predicts
        # print(conf_data.shape) torch.Size([batch_size, 8732, num_classes+1])
        # print(conf_data[0][2]) tensor([ 0.5261, -0.1007,  0.1242, -0.0905,  0.0839, -0.7308,  0.0174],device='cuda:0', grad_fn=<SelectBackward>)
        # 计算出batch_size
        num = loc_data.size(0)
        # print(loc_data.shape) torch.Size([1, 8732, 4])
        # print('1',priors.shape) torch.Size([8732, 4])
        # 取出所有的先验框
        priors = priors[:loc_data.size(1), :]  # 这一步就是保证priors的个数是和loc_data、conf_data的大小一样，其实本身就是一样的
        # print('2',priors.shape) torch.Size([8732, 4])
        # 先验框的数量
        num_priors = (priors.size(0))
        # 创建一个tensor进行处理
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)


        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        priors = priors.cuda()

        for idx in range(num):
            # 获得框
            truths = targets[idx][:, :-1]  # target存放的很多行，每一行就是一张照片，里面包括了照片里面的每一个框和对应的标签
            # 获得标签
            labels = targets[idx][:, -1]
            # 获得先验框
            defaults = priors
            # 找到标签对应的先验框
            match(self.threshold, truths, defaults, self.variance, labels,
                  # 每一个标签都对应了先验框，虽然这里没有返回值，但是loc_t和conf_t是一个tensor，函数里面对其改变了值，主函数也会跟着变化
                  loc_t, conf_t, idx)
        # 转化成Variable
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # 所有conf_t>0的地方，代表内部包含物体
        pos = conf_t > 0  # conf_t 有8732行，找到大于0的个数，相当于一张图片中8732个先验框中有pos个框是正样本
        # print(pos.shape) torch.Size([1, 8732])
        # 求和得到每一个图片内部有多少正样本
        num_pos = pos.sum(dim=1, keepdim=True)
        # print(num_pos) tensor([[12]], device='cuda:0')
        # 计算回归loss，只是对正样本进行求解回归loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print(pos_idx)
        loc_p = loc_data[pos_idx].view(-1, 4)  # 此时loc_data和pos_idx维度一样，选择出positive的loc
        # print(loc_p.shape)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # 转化形式
        batch_conf = conf_data.view(-1, self.num_classes)
        # 你可以把softmax函数看成一种接受任何数字并转换为概率分布的非线性方法
        # 获得每个框预测到真实框的类的概率
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(num, -1)

        loss_c[pos] = 0
        # 获得每一张图新的softmax的结果
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 计算每一张图的正样本数量
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算正样本的loss和负样本的loss
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        total_loss = loss_l + loss_c
        losses = [loss_l,loss_c,total_loss]
        return LossTuple(*losses)



    def train_step(self, predicts, targets):
        self.optimizer.zero_grad()
        losses = self.forward(predicts, targets)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses