from nets.yolo3 import YoloBody
from constant.Config import config
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable

from utils.Loss import Loss

from nets.yolo3train import Train
from utils.MyDataset import MyDataset,yolo_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_val(net, epoch, epoch_train_size, train_data, val_data):
    # 总的训练损失
    train_total_loss = 0
    # 位置总损失
    train_weizhi_total_loss = 0
    # 分类总损失
    train_fenlei_loss = 0
    # 置信度损失
    train_conf_loss = 0
    with tqdm(total = epoch_train_size,desc="Current Epoch:{}".format(epoch),postfix=dict,mininterval=0.3) as pbar:
        for index,data in enumerate(train_data):
            image,target_box = data[0],data[1]
            with torch.no_grad():
                image = Variable(torch.from_numpy(image).type(torch.FloatTensor)).cuda()
                target = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in target_box]
            loss_function = Loss(np.reshape(config.anchors, [-1, 2]))
            total_loss = trainer.step(image,target,loss_function) # 给输入图片，目标值，损失函数，求出损失值
            total_loss.backward()
            train_total_loss += total_loss

            pbar.set_postfix(**{'total_loss': train_total_loss.item() / (index + 1),
                                'lr': get_lr(optimizer)
                                })
            pbar.update(1)


if __name__ == '__main__':
    # 初始化相关参数
    TRAIN_ANNOTATION_PATH = "train.txt"
    # 类别
    NUM_CLASSES = config.num_classes
    # 设置将原图resize到的大小
    IMAGE_SIZE = [416,416,3]
    # 构建Yolo3模型
    MODEL = YoloBody()
    model_dict = MODEL.state_dict();
    # 不加载预训练模型
    MODEL.load_state_dict(model_dict)

    net = MODEL.train()
    net = MODEL.cuda()

    # 分割验证和训练
    val_percent = config.val_percent
    # 打开训练集文件
    with open(TRAIN_ANNOTATION_PATH) as f:
        train_val_lines = f.readlines()
    # print(train_val_lines)
    # 设定随机数种子，使得以后每次使用随机的时候，生成的数字都是确定的，应该里面的数随便填多少,我调试了下，每一个随机种子，都对应每一种随机方式，意思是你一旦确定了里面的种子数字，那么给一定一堆数据，随机分配的方案是确定的
    np.random.seed(1)
    np.random.shuffle(train_val_lines)
    # print(train_val_lines)
    np.random.seed(None)
    # 训练数据集的长度
    num_train_lines = int(len(train_val_lines)*(1-val_percent))
    # 验证数据集的长度
    num_val_linse = len(train_val_lines) - num_train_lines
    # print(num_train_lines)

    if True:
        lr = config.lr # 0.0001
        start_epoch = config.start_epoch # 0
        freeze_epoch = config.freeze_epoch # 50
        # 优化器
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config.weight_decay)
        # 调整学习率策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        for parameter in MODEL.backbone.parameters():
            parameter.requires_grad = False # 相当于不发生梯度更新，所以就冻结了前面网络的参数


        train_dataset = MyDataset(train_val_lines[:num_train_lines],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        train_data = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=0,collate_fn=yolo_dataset_collate)
        val_dataset = MyDataset(train_val_lines[num_train_lines:],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        val_data = DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=0,collate_fn=yolo_dataset_collate)

        epoch_train_size = num_train_lines//config.batch_size
        epoch_val_size = num_val_linse//config.batch_size


        trainer = Train(net,optimizer)

        for epoch in range(start_epoch,freeze_epoch):
            train_val(net,epoch,epoch_train_size,train_data,val_data)
            lr_scheduler.step()
            # break