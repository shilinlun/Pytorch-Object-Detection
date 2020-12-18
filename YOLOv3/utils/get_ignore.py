import torch
import numpy as np
def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def get_ignore(prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
    bs = len(target)
    feature_length = [13,26,52]
    anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][feature_length.index(in_w)]
    scaled_anchors = np.array(scaled_anchors)[anchor_index]
    # print(scaled_anchors)
    # 先验框的中心位置的调整参数
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    # 先验框的宽高调整参数
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

    # 生成网格，先验框中心，网格左上角
    grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
        int(bs * 3), 1, 1).view(x.shape).type(FloatTensor)
    # print(grid_x)
    grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
        int(bs * 3), 1, 1).view(y.shape).type(FloatTensor)

    # 生成先验框的宽高
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))

    '''
    print(anchor_w)
    tensor([[ 3.6250],
    [ 4.8750],
    [11.6562]], device='cuda:0')
    '''
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

    # 计算调整后的先验框中心与宽高
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    for i in range(bs):
        pred_boxes_for_ignore = pred_boxes[i]
        pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

        if len(target[i]) > 0:
            gx = target[i][:, 0:1] * in_w
            gy = target[i][:, 1:2] * in_h
            gw = target[i][:, 2:3] * in_w
            gh = target[i][:, 3:4] * in_h
            gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh], -1)).type(FloatTensor)

            anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
            for t in range(target[i].shape[0]):
                anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_iou > 0.5] = 0
            # print(torch.max(anch_ious))
    # print(noobj_mask)
    return noobj_mask