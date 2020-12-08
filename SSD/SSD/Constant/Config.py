# coding = utf-8
# Author : shi 
# Description :

class Config():
    def __init__(self):
        self.num_classes = 7
        self.val_percent = 0.1 # 训练时候，从训练集中划分验证集的比例
        self.lr = 0.0001
        self.second_lr = 0.00001
        self.start_epoch = 0
        self.freeze_epoch = 50 # 冻结步数
        self.end_epoch = 100
        self.weight_decay = 0.0005 # Adam优化器的权重衰减系数
        self.batch_size = 1

        self.min_dim = 300 #图片的大小
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.variance =  [0.1, 0.2]
        self.clip =  True


config = Config()