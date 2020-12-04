# coding = utf-8
# Author : shi 
# Description :
import numpy as np
import torch
from torch.nn import functional as F

from Utils.bbox2loc import bbox2loc
from Utils.bbox_iou import bbox_iou


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        # 利用先验框和其对应的真实框进行编码
        loc = bbox2loc(anchor, bbox[argmax_ious])

        return loc, label

    def _create_label(self, anchor, bbox):
        # 1是正样本，0是负样本，-1忽略
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # argmax_ious为每个先验框对应的最大的真实框的序号
        # max_ious为每个真实框对应的最大的真实框的iou
        # gt_argmax_ious为每一个真实框对应的最大的先验框的序号
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox)

        # 如果小于门限函数则设置为负样本
        label[max_ious < self.neg_iou_thresh] = 0

        # 每个真实框至少对应一个先验框
        label[gt_argmax_ious] = 1

        # 如果大于门限函数则设置为正样本
        label[max_ious >= self.pos_iou_thresh] = 1

        # 判断正样本数量是否大于128，如果大于的话则去掉一些
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox):
        # 计算所有
        ious = bbox_iou(anchor, bbox)
        # 行是先验框，列是真实框
        argmax_ious = ious.argmax(axis=1)
        # 找出每一个先验框对应真实框最大的iou
        max_ious = ious[np.arange(len(anchor)), argmax_ious]
        # 行是先验框，列是真实框
        gt_argmax_ious = ious.argmax(axis=0)
        # 找到每一个真实框对应的先验框最大的iou
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 每一个真实框对应的最大的先验框的序号
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious
