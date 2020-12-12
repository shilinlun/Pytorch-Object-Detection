class Config():
    def __init__(self):
        self.num_classes = 6
        self.num_anchors = 5
        self.anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

        self.object_scale = 5
        self.noobject_scale = 1
        self.class_scale = 1
        self.coord_scale = 1

        self.saturation = 1.5
        self.exposure = 1.5
        self.hue = .1

        self.jitter = 0.3

        self.thresh = .6

        self.batch_size = 1

        self.lr = 0.0001

        self.decay_lrs = {
            60: 0.00001,
            90: 0.000001
        }

        self.momentum = 0.9
        self.weight_decay = 0.0005

        # multi-scale training:
        # {k: epoch, v: scale range}
        self.multi_scale = True

        # number of steps to change input size
        self.scale_step = 40

        self.scale_range = (3, 4)

        self.epoch_scale = {
            1: (3, 4),
            15: (2, 5),
            30: (1, 6),
            60: (0, 7),
            75: (0, 9)
        }

        self.input_sizes = [(320, 320),
                       (352, 352),
                       (384, 384),
                       (416, 416),
                       (448, 448),
                       (480, 480),
                       (512, 512),
                       (544, 544),
                       (576, 576)]

        self.input_size = (416, 416)

        self.test_input_size = (416, 416)

        self.strides = 32
        self.num_workers = 0
        self.start_epoch = 0
        self.end_epoch = 100
        self.freeze_epoch = 50

config = Config()