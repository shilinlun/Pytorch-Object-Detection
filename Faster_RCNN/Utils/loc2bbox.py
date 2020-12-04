# coding = utf-8
# Author : shi 
# Description :
import numpy as np
import torch
from torch.nn import functional as F

def loc2bbox(src_bbox, loc): #src_bbox先验框，loc为建议框网络的训练参数，其实就是 dx,dy,dw,dh，然后将先验框按照loc中的dx,dy,dw,dh进行变换，
    # 使得调整后的框与真实框逼近，比如在12996个框中，肯定有些是无论如何都无法逼近的，这些其实就是背景框，所在要先选是前景宽，再对他们进行调整,
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = loc[:, 0::4] # dx,dy,dw,dh都是要学习的参数
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox