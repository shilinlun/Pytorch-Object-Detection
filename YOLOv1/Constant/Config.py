class Config():
    def __init__(self):
        self.num_classes = 6
        self.num_boxes = 2
        self.image_size = 448
        self.val_percent = 0.1

        self.lr = 0.0001
        self.second_lr = 0.00001
        self.start_epoch = 0
        self.freeze_epoch = 50 # 冻结步数
        self.end_epoch = 100
        self.weight_decay = 0.0005 # Adam优化器的权重衰减系数
        self.batch_size = 1

config = Config()