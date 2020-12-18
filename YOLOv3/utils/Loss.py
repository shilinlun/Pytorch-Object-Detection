import torch.nn as nn
from constant.Config import config
import torch

from utils.get_ignore import get_ignore
from utils.get_target import get_target
from collections import namedtuple
LossTuple = namedtuple('LossTuple',
                       ['total_loss',
                        'loss_x',
                        'loss_y',
                        'loss_w',
                        'loss_h',
                        'loss_conf',
                        'loss_cls'
                        ])
def clip_by_tensor(t, t_min, t_max):
    t = t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

def MSELoss(pred,target):
    return (pred-target)**2

class Loss(nn.Module):
    def __init__(self,anchors):
        super(Loss, self).__init__()
        self.anchors = anchors
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        '''
        print(self.anchors)
        [[116  90]
         [156 198]
         [373 326]
         [ 30  61]
         [ 62  45]
         [ 59 119]
         [ 10  13]
         [ 16  30]
         [ 33  23]]
        '''


    def forward(self, output,target):
        # print(output.shape)torch.Size([1, 33, 13, 13]) batch_size,_,h,w
        # print(target)[tensor([[0.6118, 0.7849, 0.3582, 0.1611, 0.0000]])]

        # 一个批次中图片的数量，也就是batch_size
        batch_size = output.size(0)
        # 特征层的h
        t_h = output.size(2)
        # 特征层的w
        t_w = output.size(3)
        # 步长
        stride_h = config.image_size[0] // t_h
        stride_w = config.image_size[1] // t_w
        # 将anchors的大小也按照步长进行缩放
        scaled_anchors = [[w/stride_w,h/stride_h] for w,h in self.anchors]
        # print(scaled_anchors)

        # 将output的形式转为[batch,3,13,13,5+6] 一张图片有13x13个grid，一个grid有3个框，一个框有11个参数
        # print(output.shape)torch.Size([1, 33, 13, 13])
        prediction = output.view(batch_size,3,11,t_h,t_w).permute(0,1,3,4,2).contiguous()
        # print(prediction.shape) # torch.Size([1, 3, 13, 13, 11])
        # print(prediction[...,:].shape)
        p_x = torch.sigmoid(prediction[..., 0]) # 存储的是中心的x坐标
        p_y = torch.sigmoid(prediction[..., 1])  # 存储的是中心的y坐标
        p_w = prediction[..., 2]
        p_h = prediction[..., 3]
        p_conf = torch.sigmoid(prediction[..., 4])
        p_cls = torch.sigmoid(prediction[..., 5:])

        # 获得那些先验框里面有目标，且返回类别，从先验框到真实框的调整参数，类别
        obj_mask,noobj_mask,tx,ty,tw,th,box_loss_scale_x,box_loss_scale_y,tconf,tcls = get_target(target,scaled_anchors,t_w,t_h)

        noobj_mask = get_ignore(prediction, target, scaled_anchors, t_w, t_h, noobj_mask)  # 不太懂这里

        # print(obj_mask.nonzero())

        box_loss_scale_x = (box_loss_scale_x).cuda()
        box_loss_scale_y = (box_loss_scale_y).cuda()
        obj_mask, noobj_mask = obj_mask.cuda(), noobj_mask.cuda()
        tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
        tconf, tcls = tconf.cuda(), tcls.cuda()
        box_loss_scale = 2 - box_loss_scale_x*box_loss_scale_y

        #  losses.
        loss_x = torch.sum(BCELoss(p_x, tx) / batch_size * box_loss_scale * obj_mask)  # * box_loss_scale * mask 不太懂
        loss_y = torch.sum(BCELoss(p_y, ty) / batch_size * box_loss_scale * obj_mask)
        loss_w = torch.sum(MSELoss(p_w, tw) / batch_size * 0.5 * box_loss_scale * obj_mask)
        loss_h = torch.sum(MSELoss(p_h, th) / batch_size * 0.5 * box_loss_scale * obj_mask)

        loss_conf = torch.sum(BCELoss(p_conf, obj_mask) * obj_mask / batch_size) + \
                    torch.sum(BCELoss(p_conf, obj_mask) * noobj_mask / batch_size)

        loss_cls = torch.sum(BCELoss(p_cls[obj_mask == 1], tcls[obj_mask == 1]) / batch_size)

        total_loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
        # print(loss, loss_x.item() + loss_y.item(), loss_w.item() + loss_h.item(),
        #         loss_conf.item(), loss_cls.item(), \
        #         torch.sum(mask),torch.sum(noobj_mask))
        losses = [total_loss,loss_x.item(), loss_y.item(), loss_w.item(), \
               loss_h.item(), loss_conf.item(), loss_cls.item()]
        # return LossTuple(*losses)
        return total_loss
        # total_loss = 1
        # return total_loss