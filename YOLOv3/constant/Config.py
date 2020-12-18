class Config():
    def __init__(self):
        self.num_classes = 6
        self.val_percent = 0
        self.lr = 0.0001
        self.start_epoch = 0
        self.freeze_epoch = 50
        self.end_epoch = 100
        self.weight_decay = 0.95
        self.batch_size = 1
        self.anchors = [
                        [116, 90],
                        [156, 198],
                        [373, 326],
                        [30, 61],
                        [62, 45],
                        [59, 119],
                        [10, 13],
                        [16, 30],
                        [33, 23],
                       ]
        self.image_size = [416,416]
config = Config()