# coding = utf-8
# Author : shi 
# Description :

class Config():
    def __init__(self):
        self.ratios = [0.5,1,2]
        self.anchor_scales = [8,16,32]
        self.feat_stride = 16
        self.num_anchors = 9
        self.num_classes = 6
        self.backbone = 'resnet50'
        self.val_percent = 0.1 # 训练时候，从训练集中划分验证集的比例
        self.lr = 0.0001
        self.second_lr = 0.00001
        self.start_epoch = 0
        self.freeze_epoch = 50 # 冻结步数
        self.end_epoch = 100
        self.weight_decay = 0.0005 # Adam优化器的权重衰减系数
        self.batch_size = 1


config = Config()