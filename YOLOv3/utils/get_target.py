from constant.Config import config
import torch
import numpy as np
import math


def cal_iou(gt_box, anchors):
    b1_x1, b1_x2 = gt_box[:, 0] - gt_box[:, 2] / 2, gt_box[:, 0] + gt_box[:, 2] / 2
    b1_y1, b1_y2 = gt_box[:, 1] - gt_box[:, 3] / 2, gt_box[:, 1] + gt_box[:, 3] / 2
    b2_x1, b2_x2 = anchors[:, 0] - anchors[:, 2] / 2, anchors[:, 0] + anchors[:, 2] / 2
    b2_y1, b2_y2 = anchors[:, 1] - anchors[:, 3] / 2, anchors[:, 1] + anchors[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # print(inter_area)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    print(iou)
    return iou

def get_target(target,scaled_anchors,w,h):
    '''
    :param target:
    [tensor([[0.2139, 0.6130, 0.1635, 0.3606, 0.0000]])]

    :param scaled_anchors:
    [[3.625, 2.8125],
    [4.875, 6.1875],
    [11.65625, 10.1875],
    [0.9375, 1.90625],
    [1.9375, 1.40625],
    [1.84375, 3.71875],
    [0.3125, 0.40625],
    [0.5, 0.9375],
    [1.03125, 0.71875]]

    :param w: 13

    :param h: 13

    :return:
    '''
    # print(target,scaled_anchors,w,h)
    feature_map_size = [config.image_size[0]//32,config.image_size[0]//16,config.image_size[0]//8]  # [13,26,52]
    anchor_index = [[0,1,2],[3,4,5],[6,7,8]][feature_map_size.index(w)]
    subtract_index = [0,3,6][feature_map_size.index(w)]

    batch_size = len(target)

    # --------------------下面的这些参数都是目标，参数是通过实际框的参数得来的
    # 有目标的mask,初始化为0，是目标则为1 【1,3,13,13】
    obj_mask = torch.zeros(batch_size,3,w,h,requires_grad=False)
    # 非目标的mask，和目标mask刚好相反，初始化为1，有目标为0，[1,3,13,13]
    noobj_mask = torch.ones(batch_size,3,w,h,requires_grad=False)

    # 存放中心x的调整参数,初始化为0，[1,3,13,13]
    tx = torch.zeros(batch_size,3,h,w,requires_grad=False)
    ty = torch.zeros(batch_size,3,h,w,requires_grad=False)
    tw = torch.zeros(batch_size,3,h,w,requires_grad=False)
    th = torch.zeros(batch_size,3,h,w,requires_grad=False)

    tconf = torch.zeros(batch_size,3,h,w,requires_grad=False)
    tcls = torch.zeros(batch_size,3,h,w,config.num_classes,requires_grad=False)

    box_loss_scale_x = torch.zeros(batch_size, 3, h,w, requires_grad=False)  # [1,3,13,13] 0
    box_loss_scale_y = torch.zeros(batch_size, 3, h,w, requires_grad=False)

    # ---------------------------------------------------------------
    for b in range(batch_size): # 一共有多少张图片
        for num_gt in range(target[b].shape[0]): # 一张图片中有多少个gt
            # print(target[b][num_gt]) tensor([0.3882, 0.7849, 0.3582, 0.1611, 0.0000]) [cx,cy,w,h]
            # 计算出x,y,w,h在feature map上面对应的值
            gx = target[b][num_gt][0] * w
            gy = target[b][num_gt][1] * h
            gw = target[b][num_gt][2] * w
            gh = target[b][num_gt][3] * h

            # 计算中心点属于哪个grid
            index_grid_x = int(gx)
            index_grid_y = int(gy)
            # print(index_grid_x,index_grid_y)  5 10 第5行第10列的grid

            # 将target转换到feature map上面的值,将x和y定为0，只关注w h
            gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)
            # print(gt_box) tensor([[ 0, 0,  4.6563,  2.0938]]),和下面的anchors求出iou
            anchors = torch.FloatTensor(
                np.concatenate(
                    (
                        np.zeros((9,2)),
                        np.array(scaled_anchors)
                    ),
                    axis=1
                )
            )
            '''
            print(anchors)
            tensor([[ 0.0000,  0.0000,  3.6250,  2.8125],
                    [ 0.0000,  0.0000,  4.8750,  6.1875],
                    [ 0.0000,  0.0000, 11.6562, 10.1875],
                    [ 0.0000,  0.0000,  0.9375,  1.9062],
                    [ 0.0000,  0.0000,  1.9375,  1.4062],
                    [ 0.0000,  0.0000,  1.8438,  3.7188],
                    [ 0.0000,  0.0000,  0.3125,  0.4062],
                    [ 0.0000,  0.0000,  0.5000,  0.9375],
                    [ 0.0000,  0.0000,  1.0312,  0.7188]])
            '''
            ious = cal_iou(gt_box,anchors)
            # 计算gt_box和哪一个anchors的iou最大，就可以知道，和第5行第10列的哪一个anchor的iou最大，前面令x和y为0就是为了好计算，
            # 但假如得出和第5个anchor的iou最大，但是13x13的feature map只关注0,1,2这三个，
            # 则该gt_box实际应该算作和26x26的feature map上的第5行第10列的第2（5-3）个anchor的iou最大
            best_index = np.argmax(ious)

            if best_index not in anchor_index:
                continue

            if((gx<w)and(gy<h)):
                best_index = best_index - subtract_index
                obj_mask[b,best_index,index_grid_y,index_grid_x] = 1
                noobj_mask[b,best_index,index_grid_y,index_grid_x] = 0
                tx[b,best_index,index_grid_y,index_grid_x] = gx-index_grid_x
                ty[b,best_index,index_grid_y,index_grid_x] = gy-index_grid_y
                tw[b,best_index,index_grid_y,index_grid_x] = math.log(
                    gw / scaled_anchors[best_index + subtract_index][0])
                th[b,best_index,index_grid_y,index_grid_x] = math.log(
                    gh / scaled_anchors[best_index + subtract_index][1])

                box_loss_scale_x[b,best_index,index_grid_y,index_grid_x] = target[b][num_gt][2]
                box_loss_scale_y[b,best_index,index_grid_y,index_grid_x] = target[b][num_gt][3]

                tconf[b,best_index,index_grid_y,index_grid_x] = 1
                tcls[b,best_index,index_grid_y,index_grid_x,int(target[b][num_gt][4])] = 1



            else:
                print('Step {0} out of bound'.format(b))
                print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gy, h, gx, w))
                continue
    return obj_mask,noobj_mask,tx,ty,tw,th,box_loss_scale_x,box_loss_scale_y,tconf,tcls