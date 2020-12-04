# coding = utf-8
# Author : shi 
# Description :
from torch.utils.data import DataLoader

from Constant.Config import config
from Utils.MyDataset import MyDataset
from nets.fasterrcnn import FasterRCNN

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from nets.train import Train

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    return images, bboxes, labels


def train_val(net, epoch, num_train_lines, num_val_linse, train_dataloader, val_dataloader):
    # 训练总的损失
    total_train_loss = 0
    # RPN 部分的回归损失
    total_rpn_huigui_loss = 0
    # RPN 部分的分类损失
    total_rpn_fenlei_loss = 0
    # ROI 部分的回归损失
    total_roi_huigui_loss = 0
    # ROI 部分的分类损失
    total_roi_fenlei_loss = 0
    # 验证总的损失
    total_val_loss = 0

    # tqdm 参数解释
    # iterable = None,
    # desc = None, 传入str类型，作为进度条标题（类似于说明）
    # total = None, 预期的迭代次数
    # leave = True,
    # file = None,
    # ncols = None, 可以自定义进度条的总长度
    # mininterval = 0.1, 最小的更新间隔
    # maxinterval = 10.0, 最大更新间隔
    # miniters = None,
    # ascii = None,
    # unit = 'it',
    # unit_scale = False,
    # dynamic_ncols = False,

    # smoothing = 0.3,
    # bar_format = None,
    # initial = 0,
    # position = None,
    # postfix 以字典形式传入

    with tqdm(desc='Current Epoch:{}'.format(epoch),total=num_train_lines,postfix=dict,mininterval=0.1) as pbar:
        for step,data in enumerate(train_dataloader):
            img,boxes,labels = data[0],data[1],data[2]
            # 由于这三项是确定的，不需要梯度下降
            with torch.no_grad():
                img = Variable(torch.from_numpy(img).type(torch.FloatTensor)).cuda()
                boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]


            rpn_huigui_loss,rpn_fenlei_loss,roi_huigui_loss,roi_fenlei_loss,total_loss = trainer.train_step(img,boxes,labels,scale=1)

            total_train_loss += total_loss
            total_rpn_huigui_loss += rpn_huigui_loss
            total_rpn_fenlei_loss += rpn_fenlei_loss
            total_roi_huigui_loss += roi_huigui_loss
            total_roi_fenlei_loss += roi_fenlei_loss

            pbar.set_postfix(**{'total': total_train_loss.item() / (step + 1),
                                'rpn_loc': total_rpn_huigui_loss.item() / (step + 1),
                                'rpn_cls': total_rpn_fenlei_loss.item() / (step + 1),
                                'roi_loc': total_roi_huigui_loss.item() / (step + 1),
                                'roi_cls': total_roi_fenlei_loss.item() / (step + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('\nStart Validation')
    with tqdm(total=num_val_linse, desc=f'Current Epoch {epoch}',postfix=dict,mininterval=0.1) as pbar:
        for iteration, batch in enumerate(val_dataloader):
            img,boxes,labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                img = Variable(torch.from_numpy(img).type(torch.FloatTensor)).cuda()
                boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
                labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]


                trainer.optimizer.zero_grad()
                losses = trainer.forward(img, boxes, labels, scale=1)
                _,_,_,_, val_loss = losses
                total_val_loss += val_loss
            pbar.set_postfix(**{'total_loss': total_val_loss.item() / (iteration + 1)})
            pbar.update(1)

    print('\nFinish Validation')
    print('Epoch:'+ str(epoch))
    print('Train Loss: %.4f || Val Loss: %.4f ' % (total_train_loss/(num_train_lines+1),total_val_loss/(num_val_linse+1)))
    print('Saving state, iter:', str(epoch))
    torch.save(MODEL.state_dict(), 'logs/Epoch%d-Train_Loss%.4f-Val_Loss%.4f.pth'%((epoch),total_train_loss/(num_train_lines+1),total_val_loss/(num_val_linse+1)))

if __name__ == '__main__':
    # 初始化相关参数
    TRAIN_ANNOTATION_PATH = "train.txt"
    # 类别
    NUM_CLASSES = config.num_classes
    # 设置主干网络,现在还没有使用到，因为默认就是使用这个
    BACKBONE = config.backbone
    # 构建Faster-RCNN模型
    MODEL = FasterRCNN(num_classes=NUM_CLASSES)
    # 设置将原图resize到的大小
    IMAGE_SIZE = [600,600,3]

    # 加载预训练模型
    pre_model_path = r'model_data/voc_weights_resnet.pth'
    print("Start load pre model:{}".format(pre_model_path))
    # 设置deice，这里没有考虑使用cpu
    device = torch.device('cuda')
    # 读取模型的网络结构和参数,比如第一层权重，第二层权重。。。（），是随机的
    model_dict = MODEL.state_dict()
    # 将预训练模型加载进去，也就是有了初始模型参数
    pre_model_dict = torch.load(pre_model_path,map_location=device)
    # print(pre_model_dict),我调试了下，发现这个预训练模型是作者自己按照自己的网络结构训练的，所以需要按照他的结构对网络参数取名字,所以我训练的时候没有预加载
    # pretrained_dict = {k: v for k, v in pre_model_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    MODEL.load_state_dict(model_dict)
    print('Load Finished')

    # 设置训练模式
    net = MODEL.train()
    # 若是多GPU，则设置平均分配
    net = nn.DataParallel(MODEL)
    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异
    cudnn.benchmark = True
    # 加载到GPU
    net = net.to(device)

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


    if True:

        lr = config.lr # 0.0001
        start_epoch = config.start_epoch # 0
        freeze_epoch = config.freeze_epoch # 50

        # ------------------------------------------------------
        # 优化器
        optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=config.weight_decay)
        # 调整学习率策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
        # for epoch in range(start_epoch,freeze_epoch):
        #     print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        #     optimizer.zero_grad()
        #     optimizer.step()
        #     lr_scheduler.step()

        # 先冻结一部分网络，此处冻结的是ROIPooling之前的网络
        for parameter in MODEL.extractor.parameters():
            parameter.requires_grad = False # 相当于不发生梯度更新，所以就冻结了前面网络的参数

        # 看网上说eval时候和train时候若精度差距很大，可能就是bn层的原因，原作者在这里冻结了bn层，但我也不是很清楚
        MODEL.freeze_bn()

        # ------------------------------------------------------
        # 构建数据集,原作者只能设置为batch size为1,意思是使用了这个类，输出参数是所有的训练样本的数据（train.txt），resize的大小，
        # 然后输出的是增强后的图片，包括框也变化了
        train_data = MyDataset(train_val_lines[:num_train_lines],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        # train_data就是一个一个list，可以使用下面代码看下是什么，其实就是图片像素值、图片中的框的dx,dy,dw,dh、每一个框所属的类别
        # print(train_data.__getitem__(1))
        val_data = MyDataset(train_val_lines[num_train_lines:],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        # 设置pin_memory = True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些

        # drop_last(bool,optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，
        # 而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
        # 如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点

        # 一般的，默认的collate_fn函数是要求一个batch中的图片都具有相同size（因为要做stack操作），当一个batch中的图片大小都不同时，
        # 可以使用自定义的collate_fn函数，则一个batch中的图片不再被stack操作，可以全部存储在一个list中，当然还有对应的label
        train_dataloader = DataLoader(train_data,shuffle=True,batch_size=config.batch_size,num_workers=0,pin_memory=True,
                                      drop_last=False,collate_fn=frcnn_dataset_collate)
        val_dataloader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
                                      drop_last=False, collate_fn=frcnn_dataset_collate)
        # print(train_dataloader)
        # -----------------------------------------------------

        trainer = Train(MODEL,optimizer)

        # 训练及验证开始
        for epoch in range(start_epoch,freeze_epoch):
            train_val(net,epoch,num_train_lines,num_val_linse,train_dataloader,val_dataloader)
            lr_scheduler.step()

    if True:
        lr = config.second_lr # 0.00001
        Freeze_Epoch = config.freeze_epoch # 50
        End_Epoch = config.end_epoch # 100

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=config.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


        train_data = MyDataset(train_val_lines[:num_train_lines],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        val_data = MyDataset(train_val_lines[num_train_lines:],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        val_dataloader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)



        # 解冻后训练

        for param in MODEL.extractor.parameters():
            param.requires_grad = True


        MODEL.freeze_bn()

        train_util = Train(MODEL, optimizer)

        for epoch in range(Freeze_Epoch, End_Epoch):
            train_val(net, epoch, num_train_lines, num_val_linse, train_dataloader, val_dataloader)
            lr_scheduler.step()

