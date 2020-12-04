# coding = utf-8
# Author : shi 
# Description :
import numpy as np
import torch
from torch.nn import functional as F


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc