# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from Constant.Config import config
from utils.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, box_ious, xxyy2xywh, box_transform
import torch.nn.functional as F


def build_target(output, gt_data, H, W):
    """
    Build the training target for output tensor

    Arguments:

    output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
    gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

    delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)

    gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values
                       (x1, y1, x2, y2) range 0~1
    gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
    num_obj_batch -- tensor of shape (B, 1). number of objects


    Returns:
    iou_target -- tensor of shape (B, H * W * num_anchors, 1)  是模型的结果中的xywh调整参数使用在anchor上的后调整框和真实框的iou
    iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
    box_target -- tensor of shape (B, H * W * num_anchors, 4)  是需要学习的xydh，anchor经过这个值的变换才可以到真实框
    box_mask -- tensor of shape (B, H * W * num_anchors, 1)
    class_target -- tensor of shape (B, H * W * num_anchors, 1) 是需要学习的参数，该框的类别
    class_mask -- tensor of shape (B, H * W * num_anchors, 1)

    """
    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    gt_boxes_batch = gt_data[0]
    gt_classes_batch = gt_data[1]
    num_boxes_batch = gt_data[2]

    bsize = delta_pred_batch.size(0)

    num_anchors = 5  # hard code for now

    # initial the output tensor
    # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
    # what tensor is used doesn't matter
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    # print(iou_target.shape) torch.Size([1, 13x13, 5, 1]) 全是0
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * config.noobject_scale
    # print(iou_mask.shape) torch.Size([1, 169, 5, 1]) 全是1*1
    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    # print(box_target.shape) torch.Size([1, 169, 5, 4]) 全是0
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    # print(box_mask.shape) torch.Size([1, 169, 5, 1]) 全是0

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    # print(class_target.shape) torch.Size([1, 169, 5, 1]) 全是0
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    # print(class_mask.shape) torch.Size([1, 169, 5, 1]) 全是0

    # get all the anchors

    anchors = torch.FloatTensor(config.anchors)

    # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
    # this is very crucial because the predict output is normalized to 0~1, which is also
    # normalized by the grid width and height
    all_grid_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    # 有13x13x5行，4列，里面放的每一个点对应的框的坐标，一共有169个点，每个点生成5个框，框的长宽比列就是cfg.anchors
    '''
    print(all_grid_xywh)
    tensor([[ 0.0000,  0.0000,  1.3221,  1.7314],
        [ 0.0000,  0.0000,  3.1927,  4.0094],
        [ 0.0000,  0.0000,  5.0559,  8.0989],
        ...,
        [12.0000, 12.0000,  5.0559,  8.0989],
        [12.0000, 12.0000,  9.4711,  4.8405],
        [12.0000, 12.0000, 11.2364, 10.0071]])
    '''

    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh) # 这一行不知道有什么用处
    '''
    print(all_grid_xywh)
    tensor([[ 0.0000,  0.0000,  1.3221,  1.7314],
        [ 0.0000,  0.0000,  3.1927,  4.0094],
        [ 0.0000,  0.0000,  5.0559,  8.0989],
        ...,
        [12.0000, 12.0000,  5.0559,  8.0989],
        [12.0000, 12.0000,  9.4711,  4.8405],
        [12.0000, 12.0000, 11.2364, 10.0071]])
    '''
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    '''
    print(all_anchors_xywh)
    tensor([[ 0.5000,  0.5000,  1.3221,  1.7314], (中心xy，宽长wh)
        [ 0.5000,  0.5000,  3.1927,  4.0094],
        [ 0.5000,  0.5000,  5.0559,  8.0989],
        ...,
        [12.5000, 12.5000,  5.0559,  8.0989],
        [12.5000, 12.5000,  9.4711,  4.8405],
        [12.5000, 12.5000, 11.2364, 10.0071]], device='cuda:0')
    '''

    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)
    '''
    print(all_anchors_xxyy)
    tensor([[-0.1611, -0.3657,  1.1611,  1.3657], (左上角的xy，右下角的xy)
        [-1.0964, -1.5047,  2.0964,  2.5047],
        [-2.0279, -3.5495,  3.0279,  4.5495],
        ...,
        [ 9.9721,  8.4505, 15.0279, 16.5495],
        [ 7.7644, 10.0797, 17.2356, 14.9203],
        [ 6.8818,  7.4964, 18.1182, 17.5035]], device='cuda:0')
    '''

    # process over batches
    for b in range(bsize):
        num_obj = num_boxes_batch[b].item() # 本实验就是2
        delta_pred = delta_pred_batch[b] # 假如是0.3,0.3,0.1,0.4 cx,cy,w,h
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        '''
        print(gt_boxes)
        tensor([[0.5210, 0.3113, 0.6534, 0.6071],
        [0.3422, 0.4967, 0.4746, 0.6269]], device='cuda:0')
        '''
        gt_classes = gt_classes_batch[b][:num_obj]
        '''
        print(gt_classes)
        tensor([2., 4.], device='cuda:0')
        '''

        # rescale ground truth boxes
        gt_boxes[:, 0::2] *= W
        gt_boxes[:, 1::2] *= H
        '''
        print(gt_boxes) 相当于是在13x13上面确定位置
        tensor([[6.7726, 4.0464, 8.4945, 7.8918],
        [4.4481, 6.4570, 6.1700, 8.1501]], device='cuda:0')
        '''

        # step 1: process IoU target

        # apply delta_pred to pre-defined anchors
        # print(all_anchors_xywh.shape)
        all_anchors_xywh = all_anchors_xywh.view(-1, 4) # 值是一样的，应该是把batch_size给处理掉
        # print(all_anchors_xywh.shape)

        '''
        print(all_grid_xywh) #这个是13x13生成的anchors，一共845个
        tensor([[ 0.0000,  0.0000,  1.3221,  1.7314],
        [ 0.0000,  0.0000,  3.1927,  4.0094],
        [ 0.0000,  0.0000,  5.0559,  8.0989],
        ...,
        [12.0000, 12.0000,  5.0559,  8.0989],
        [12.0000, 12.0000,  9.4711,  4.8405],
        [12.0000, 12.0000, 11.2364, 10.0071]], device='cuda:0')
        '''

        '''
        print(delta_pred) # 这个是预测的dx,dy,dw,dh
        tensor([[0.4576, 0.5612, 0.8732, 0.9816],
        [0.5845, 0.5348, 1.3881, 1.1090],
        [0.5137, 0.4990, 1.1416, 0.9874],
        ...,
        [0.5478, 0.5210, 0.9887, 1.0456],
        [0.5587, 0.4846, 1.0993, 1.2534],
        [0.5254, 0.4511, 1.1534, 1.1708]], device='cuda:0')
        '''
        box_pred = box_transform_inv(all_grid_xywh, delta_pred) # 相当于box_pred就是通过学习到的dx,dy,dw,dh来对anchors进行调整

        '''
        print(box_pred)
        tensor([[ 0.5210,  0.4612,  1.3657,  1.9739], cx,cy,w,h
        [ 0.4855,  0.5015,  3.6202,  4.3269],
        [ 0.4892,  0.5337,  4.9342,  6.7088],
        ...,
        [12.4543, 12.5305,  5.6022,  6.7554],
        [12.4537, 12.4909, 10.8255,  4.3415],
        [12.4064, 12.4736, 12.2311, 13.7396]], device='cuda:0')
        '''
        box_pred = xywh2xxyy(box_pred)
        '''
        print(box_pred)
        tensor([[-0.3113, -0.3341,  1.2396,  1.2901], xxyy(每一轮是和前面匹配的，但是不同轮是不一样的，所以不要按照上面的进行计算)
        [-1.4217, -1.4503,  2.2727,  2.4420],
        [-1.9601, -2.9279,  2.8802,  3.8738],
        ...,
        [10.0931,  8.8594, 14.9269, 16.0764],
        [ 7.2448,  9.7277, 17.8272, 15.3263],
        [ 7.4760,  7.5047, 17.6143, 17.5238]], device='cuda:0')
        '''

        # for each anchor, its iou target is corresponded to the max iou with any gt boxes
        # box_pred (845,4) gt_boxes (2,4)
        ious = box_ious(box_pred, gt_boxes) # shape: (H * W * num_anchors, num_obj)
        # print(ious) (845,2),意思就是在第一列中，比如第7个数字最大，则第7个框就和第一个目标的框的iou最大，就更可能是第一个目标
        # 第二列中，比如第10个数字最大，则第10个框就和第二个目标的框的iou最大，就更可能是第二个目标
        ious = ious.view(-1, num_anchors, num_obj)
        # print(ious) (169,5,2)
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True) # shape: (H * W, num_anchors, 1) dim=-1表示找每一行的最大值

        '''
        print(max_iou)
        tensor([[[0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [7.2817e-04]],

        [[0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [2.1793e-03]],

        [[0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [1.7213e-02]],

        [[0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [9.3445e-03]],

        [[0.0000e+00],
         [0.0000e+00],
         [4.7415e-03],
         [0.0000e+00],
         [1.8997e-02]],

        [[0.0000e+00],
         [0.0000e+00],
         [1.3284e-02],
         [0.0000e+00],
         [1.0497e-02]],

        [[0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [0.0000e+00],
         [1.2457e-03]],

        [[0.0000e+00],
         [0.0000e+00],
         [2.8610e-02],
         [0.0000e+00],
         [1.7553e-03]],
        '''


        # iou_target[b] = max_iou

        # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold

        '''
        print(max_iou.view(-1))
        tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 每5个就是一个点的框的iou，但应该这里只能知道第几个点的哪一个anchors可以调整到最后有目标，但不知道是哪个类别
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.5640e-02, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.2098e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 2.6339e-02, 0.0000e+00, 0.0000e+00, 3.4884e-03, 0.0000e+00,
        1.0437e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7399e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.2665e-02, 0.0000e+00,
        0.0000e+00, 8.2178e-03, 0.0000e+00, 8.0863e-03, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.1858e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 3.0100e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        1.1220e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6352e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.3251e-02, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.8070e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 3.2046e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        3.5162e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.8719e-02,
        0.0000e+00, 0.0000e+00, 2.3394e-02, 1.2346e-02, 3.2767e-02, 0.0000e+00,
        0.0000e+00, 7.0038e-02, 7.3301e-04, 3.5359e-02, 0.0000e+00, 0.0000e+00,
        6.4701e-02, 3.2703e-03, 3.2102e-02, 0.0000e+00, 0.0000e+00, 6.1244e-02,
        0.0000e+00, 3.9091e-02, 0.0000e+00, 0.0000e+00, 1.6556e-02, 0.0000e+00,
        3.2905e-02, 0.0000e+00, 0.0000e+00, 1.8021e-02, 0.0000e+00, 3.7667e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.6490e-02, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 5.6092e-03, 3.2198e-02, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 1.1116e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 4.1568e-02, 0.0000e+00, 0.0000e+00, 6.2163e-03, 1.7235e-02,
        4.4665e-02, 0.0000e+00, 0.0000e+00, 2.7735e-02, 2.5414e-02, 4.9889e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7983e-02, 5.6209e-02, 0.0000e+00,
        8.2520e-03, 7.6297e-02, 1.8399e-02, 3.2712e-02, 0.0000e+00, 3.9794e-02,
        9.0482e-02, 2.9318e-02, 5.2858e-02, 0.0000e+00, 5.4972e-02, 1.1581e-01,
        3.4513e-03, 3.7382e-02, 0.0000e+00, 3.1070e-02, 9.3850e-02, 1.7858e-02,
        4.6780e-02, 0.0000e+00, 7.3401e-03, 5.1310e-02, 2.2485e-02, 4.2806e-02,
        0.0000e+00, 0.0000e+00, 2.2860e-02, 3.2773e-02, 5.2414e-02, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.9834e-02, 5.5735e-02, 0.0000e+00, 0.0000e+00
        '''
        iou_thresh_filter = max_iou.view(-1) > config.thresh
        # print(max_iou)

        '''
        print(iou_thresh_filter)
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0], device='cuda:0', dtype=torch.uint8)
        '''
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        '''
        print(torch.nonzero(iou_thresh_filter)) # 找到非0的地方的index
        tensor([[426],
        [480]], device='cuda:0') ,相当于第426和第480是非0的，说明这里的框调整后和gt有大于0.6的iou
        numel()函数：返回数组中元素的个数
        '''

        # iou_mask本身是1x169x5x1的全是1，现在相当于iou大于0.6的地方设置为0
        if n_pos > 0:
            iou_mask[b][max_iou >= config.thresh] = 0


        # step 2: process box target and class target
        # calculate overlaps between anchors and gt boxes
        # all_anchors_xxyy就是生成的anchors(845,4),gt_boxes就是真实的框(2,4),下面的代码相当于可以计算出每一个anchors和第一个框的重叠，每一个anchors和第二个框的重叠
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
        # print(overlaps.shape) torch.Size([169, 5, 2])

        '''
        print(gt_boxes)
        tensor([[6.7726, 4.0464, 8.4945, 7.8918],
        [4.4481, 6.4570, 6.1700, 8.1501]], device='cuda:0')
        '''
        gt_boxes_xywh = xxyy2xywh(gt_boxes) # 把 x,x,y,y转化为cx,cy,w,h

        '''
        print(gt_boxes_xywh)
        tensor([[7.6336, 5.9691, 1.7219, 3.8455],
        [5.3091, 7.3035, 1.7219, 1.6932]], device='cuda:0')
        '''

        # iterate over all objects

        for t in range(gt_boxes.size(0)): # 循环次数为目标的个数
            # compute the center of each gt box to determine which cell it falls on
            # assign it to a specific anchor by choosing max IoU

            gt_box_xywh = gt_boxes_xywh[t]
            # print(gt_classes)tensor([2., 4.], device='cuda:0')
            gt_class = gt_classes[t]
            # print('1',gt_class)tensor(2., device='cuda:0')
            # print(gt_box_xywh)tensor ([7.6336, 5.9691, 1.7219, 3.8455], device='cuda:0')
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            # print(cell_idx_x, cell_idx_y) tensor(7., device='cuda:0') tensor(5., device='cuda:0')相当于是第5行第7列的grid是真实目标框的中心
            cell_idx = cell_idx_y * W + cell_idx_x # 因为前面的overlaps是13x13,所以这里使用5x13+7就相当于找到了第72个grid是真实目标的中心
            cell_idx = cell_idx.long()

            # update box_target, box_mask
            # print(cell_idx)
            overlaps_in_cell = overlaps[cell_idx, :, t] # overlaps是169x5x2 相当于是找到169个中的第72个grid，且是2中的第一个（第一个目标）
            # print(overlaps_in_cell) tensor([0.3457, 0.4422, 0.1617, 0.1444, 0.0589], device='cuda:0') 这5个数字就是第72个grid的5个anchors
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)
            # print(argmax_anchor_idx) tensor(1, device='cuda:0') 找到5个anchors中哪一个的值最大，就说明更有可能是这个anchors

            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0) #这里返回的是169个grid中第72个grid中的第1个anchors的xywh,其实对于一张图片而言，这个值都是定的，
            # print(assigned_grid) tensor([[7.0000, 5.0000, 3.1927, 4.0094]], device='cuda:0')
            gt_box = gt_box_xywh.unsqueeze(0)
            # print(gt_box) tensor([[7.6336, 5.9691, 1.7219, 3.8455]], device='cuda:0')
            # assigned_grid是生成的anchors中的一个anchor的4个坐标，这个anchor又是根据真实框的位置，来选定的和他iou最大的一个anchor
            # gt_box是实际框的4个坐标
            # print(assigned_grid) tensor([[7.0000, 5.0000, 3.1927, 4.0094]], device='cuda:0')
            target_t = box_transform(assigned_grid, gt_box) #这个就是需要学习的参数，怎么让anchor调整到真实框
            # print(target_t) tensor([[0.6336, 0.9691, 0.5393, 0.9591]], device='cuda:0')
            '''
            比如anchor的cx,cy,w,h为7.0000, 5.0000, 3.1927, 4.0094
            若学习到的参数为0.6336, 0.9691, 0.5393, 0.9591，则
            转换后的cx,cy,w,h为(7.0000+0.6336,5.0000+0.9691,3.1927*0.5393,4.0094*0.9591)
            真实框的cx,cy,w,h为7.6336, 5.9691, 1.7219, 3.8455
            
            '''

            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0) #box_target中第1个batch中第72个grid中的第1个anchor的4个转换参数就是target_t，其余全是0
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1 # box_mask中的第1个batch中第72个grid中的第1个anchor的值是1，其余是0，相当于知道这个anchor是目标，其余是背景

            # update cls_target, cls_mask
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class # class_target中的第1个batch中第72个grid中的第1个anchor的值是该anchor的类别
            # print(class_target.shape)
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1 # class_mask中的第1个batch中第72个grid中的第1个anchor的值是1，其余是0，相当于该anchor是属于有类别的，其余的都是没有类别的

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :] # iou_target第1个batch中第72个grid中的第1个anchor的值是max_iou中的第72个grid中的第1个anchor的iou
            # 其中max_iou里面所有的值都是预测框和真实的框的iou

            iou_mask[b, cell_idx, argmax_anchor_idx, :] = config.object_scale # iou_mask第1个batch中第72个grid中的第1个anchor的值是5，其余都是1，表明该anchor是属于目标

    return iou_target.view(bsize, -1, 1), \
           iou_mask.view(bsize, -1, 1), \
           box_target.view(bsize, -1, 4),\
           box_mask.view(bsize, -1, 1), \
           class_target.view(bsize, -1, 1).long(), \
           class_mask.view(bsize, -1, 1)


def yolo_loss(output, target):
    """
    Build yolo loss

    Arguments:
    output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
    target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

    delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

    iou_target -- Variable of shape (B, H * W * num_anchors, 1)
    iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
    box_target -- Variable of shape (B, H * W * num_anchors, 4)
    box_mask -- Variable of shape (B, H * W * num_anchors, 1)
    class_target -- Variable of shape (B, H * W * num_anchors, 1)
    class_mask -- Variable of shape (B, H * W * num_anchors, 1)

    Return:
    loss -- yolo overall multi-task loss
    """

    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes) # torch.Size([845, 6])
    class_target = class_target.view(-1) # 展开，里面的值其实相当于第72*5+1=361中的值为2（第二类）
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1) # [361,480],480先不看，就是第361个值为非0
    # print(class_keep)
    class_score_batch_keep = class_score_batch[class_keep, :] # 找到class_score_batch第361个值，class_score_batch就是模型输出的值中的类别得分

    '''
    print(class_score_batch_keep)
    tensor([[ 0.3250,  0.8892, -0.5601, -0.0491,  0.2582, -0.3736],
        [ 0.0091, -0.2019,  0.3260,  0.1245, -0.0793, -0.0901]],
       device='cuda:0', grad_fn=<IndexBackward>)
    '''
    class_target_keep = class_target[class_keep]
    # print(class_target_keep) 2 (第二类)

    # if cfg.debug:
    #     print(class_score_batch_keep)
    #     print(class_target_keep)

    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * config.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * config.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss







































