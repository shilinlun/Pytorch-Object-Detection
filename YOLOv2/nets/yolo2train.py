import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.loss import build_target, yolo_loss
import torch.nn.functional as F
from collections import namedtuple
from Constant.Config import config

LossTuple = namedtuple('LossTuple',
                       ['box_loss',
                        'iou_loss',
                        'class_loss',
                        'total_loss'
                        ])

class Train(nn.Module):
    def __init__(self,model,optimizer):
        super(Train, self).__init__()
        self.optimizer = optimizer
        self.model = model

    def forward(self,x, gt_boxes=None, gt_classes=None, num_boxes=None):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        # x = self.conv1(x)
        # shortcut = self.reorg(self.downsampler(x))
        # # print(shortcut.shape) layers torch.Size([1, 256, 13, 13])
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = torch.cat([shortcut, x], dim=1)
        # out = self.conv4(x)
        out = self.model(x)



        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        # print(out.shape) torch.Size([2, (5+6)*5, 13, 13])
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * config.num_anchors, 5 + config.num_classes) # [batch_size,13x13x5,5+6]
        # print(out.shape) torch.Size([2, 845, 11])

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2]) # 2个数
        # print(xy_pred.shape)
        conf_pred = torch.sigmoid(out[:, :, 4:5]) # 1个数
        hw_pred = torch.exp(out[:, :, 2:4]) # 2个数
        class_score = out[:, :, 5:] # 6个数
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)


        # print(class_score.shape)
        output_variable = (delta_pred, conf_pred, class_score)
        output_data = [v.data for v in output_variable]
        gt_data = (gt_boxes, gt_classes, num_boxes)
        # print(output_data[0].shape) torch.Size([1, 845, 4]) 4代表dx,dy,dw,dh
        # print(gt_data[0].shape) torch.Size([1, 5, 4]) 5代表有5个框
        # output_data : [(1,845,4),(1,845,1),(1,845,6)]
        # gt_data : [(1,5,4),(1,5),(1,1)]
        # (1,5,4)表示该图片有5个框，每个框有4个坐标，比如[[0.4,0.3,0.6,0.7],[],[],[],[]]
        # (1,5) 5表示5个框分别对应的类别比如[1,4,5,2,1],第一类，第四类...
        # (1,1) 表示里面只有1个数字，值为框的个数，比如[5]
        # print(gt_data[0])
        # print(gt_data[1])
        # print(gt_data[2].shape)
        target_data = build_target(output_data, gt_data, h, w)

        target_variable = [Variable(v) for v in target_data]
        # print(target_variable[0].shape) torch.Size([1, 845, 1])
        # print(output_variable[0].shape) torch.Size([1, 845, 4])
        box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

        losses = [box_loss, iou_loss, class_loss]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)



    def train_step(self,x, gt_boxes=None, gt_classes=None, num_boxes=None):
        self.optimizer.zero_grad()
        losses = self.forward(x, gt_boxes, gt_classes, num_boxes)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses