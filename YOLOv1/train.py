from torch.utils.data import DataLoader

from Constant.Config import config

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable

from nets.yolo1train import Train
from nets.yolo_resnet import yolo_resnet
from utils.MyDataset import MyDataset


# def yolo1_dataset_collate(batch):
#     images = []
#     bboxes = []
#     for img, box in batch:
#         images.append(img)
#         bboxes.append(box)
#     images = np.array(images)
#     bboxes = np.array(bboxes)
#     return images, bboxes

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_val(net, epoch, num_train_lines, num_val_linse, train_dataloader,val_dataloader):
    # 训练总的损失
    total_train_loss = 0
    # 回归损失
    total_huigui_loss = 0
    # 分类损失
    total_fenlei_loss = 0
    # 正样本confidence损失
    total_obj_confi_loss = 0
    # 负样本confidence损失
    total_noobj_confi_loss = 0
    # 验证总的损失
    total_val_loss = 0
    with tqdm(desc='Current Epoch:{}'.format(epoch), total=num_train_lines, postfix=dict, mininterval=0.1) as pbar:
        for step,data in enumerate(train_dataloader):
            img,targets = data[0],data[1] #注意这里其实和faster-rcnn一样,只是把第二和第三维合在一起了
            # 由于这三项是确定的，不需要梯度下降
            with torch.no_grad():
                img = img.cuda()
                targets = targets.float().cuda()
            predicts = net(img)

            coor_loss,obj_confi_loss,noobj_confi_loss,class_loss,total_loss = trainer.train_step(predicts,targets)
            total_train_loss += total_loss
            total_fenlei_loss += class_loss
            total_huigui_loss += coor_loss
            total_obj_confi_loss += obj_confi_loss
            total_noobj_confi_loss += noobj_confi_loss

            pbar.set_postfix(**{'total': total_train_loss.item() / (step + 1),
                                'fenlei': total_fenlei_loss.item() / (step + 1),
                                'huigui': total_huigui_loss.item() / (step + 1),
                                'zheng_confi': total_obj_confi_loss.item() / (step + 1),
                                'fu_confi': total_noobj_confi_loss.item() / (step + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('\nStrart Validation')
    with tqdm(desc='Current Epoch:{}'.format(epoch), total=num_val_linse, postfix=dict, mininterval=0.1) as pbar:
        for step,data in enumerate(val_dataloader):
            img,targets = data[0],data[1]
            with torch.no_grad():
                img = img.cuda()
                targets = targets.float().cuda()
            trainer.optimizer.zero_grad()
            predicts = net(img)
            _,_,_,_,val_loss = trainer.forward(predicts, targets)
            total_val_loss += val_loss
            pbar.set_postfix(**{'total_loss': total_val_loss.item() / (step + 1)})
            pbar.update(1)
    print('\nFinish Validation')
    print('Epoch:'+ str(epoch))
    print('Train Loss: %.4f || Val Loss: %.4f ' % (total_train_loss/(num_train_lines+1),total_val_loss/(num_val_linse+1)))
    print('Saving state, iter:', str(epoch))
    torch.save(MODEL.state_dict(), 'logs/Epoch%d-Train_Loss%.4f-Val_Loss%.4f.pth'%((epoch),total_train_loss/(num_train_lines+1),total_val_loss/(num_val_linse+1)))


if __name__ == '__main__':
    # 初始化相关参数
    TRAIN_ANNOTATION_PATH = "/home/shi/data/Deep_Learing/Object_Detection/YOLOv1/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
    VAL_ANNOTATION_PATH = "/home/shi/data/Deep_Learing/Object_Detection/YOLOv1/VOCdevkit/VOC2007/ImageSets/Main/val.txt"
    # 构建YOLO1模型
    MODEL = yolo_resnet()
    # 设置将原图resize到的大小
    IMAGE_SIZE = [config.image_size,config.image_size,3] # [448,448,3]
    # 设置deice，这里没有考虑使用cpu
    device = torch.device('cuda')
    # 读取模型的网络结构和参数,比如第一层权重，第二层权重。。。（），是随机的
    model_dict = MODEL.state_dict()
    # 这里没有使用迁移学习
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
    # val_percent = config.val_percent

    # 打开训练集文件
    with open(TRAIN_ANNOTATION_PATH) as f:
        train_lines = f.readlines()
    with open(VAL_ANNOTATION_PATH) as f:
        val_lines = f.readlines()
    # print(train_val_lines)
    # 设定随机数种子，使得以后每次使用随机的时候，生成的数字都是确定的，应该里面的数随便填多少,我调试了下，每一个随机种子，都对应每一种随机方式，意思是你一旦确定了里面的种子数字，那么给一定一堆数据，随机分配的方案是确定的
    # np.random.seed(1)
    # np.random.shuffle(train_val_lines)
    # print(train_val_lines)
    # np.random.seed(None)
    # 训练数据集的长度
    num_train_lines = len(train_lines)
    # 验证数据集的长度
    num_val_linse = len(val_lines)

    if True:

        lr = config.lr # 0.0001
        start_epoch = config.start_epoch # 0
        freeze_epoch = config.freeze_epoch # 50

        # 优化器
        optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=config.weight_decay)
        # 调整学习率策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)


        # ---------------------------------- 这里不知道先冻结哪里,按道理应该冻结resnet，因为resnet是我们载入的预训练模型
        # 先冻结yolo_fc层
        for param in MODEL.yolo_fc.parameters():
            param.requires_grad = False
        # 先冻结yolo_conv层
        for param in MODEL.yolo_conv.parameters():
            param.requires_grad = False
        # 先冻结resnet层
        # for param in MODEL.resnet.parameters():
        #     param.requires_grad = False

        train_data = MyDataset([IMAGE_SIZE[0], IMAGE_SIZE[1]],mode="train")
        val_data = MyDataset([IMAGE_SIZE[0], IMAGE_SIZE[1]],mode="val")

        train_dataloader = DataLoader(train_data,shuffle=True,batch_size=config.batch_size,num_workers=0,pin_memory=True,
                                      drop_last=False)
        val_dataloader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
                                      drop_last=False)

        trainer = Train(optimizer)

        # 训练及验证开始
        for epoch in range(start_epoch,freeze_epoch):
            train_val(net,epoch,num_train_lines,num_val_linse,train_dataloader,val_dataloader)
            lr_scheduler.step()

    # if True:
    #
    #     lr = config.second_lr # 0.00001
    #     start_epoch = config.freeze_epoch # 50
    #     freeze_epoch = config.end_epoch # 100
    #
    #     # 优化器
    #     optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=config.weight_decay)
    #     # 调整学习率策略
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    #
    #     # 解冻vgg层
    #     for param in MODEL.vgg.parameters():
    #         param.requires_grad = True
    #
    #     train_data = MyDataset(train_val_lines[:num_train_lines], [IMAGE_SIZE[0], IMAGE_SIZE[1]])
    #     val_data = MyDataset(train_val_lines[num_train_lines:], [IMAGE_SIZE[0], IMAGE_SIZE[1]])
    #
    #     train_dataloader = DataLoader(train_data,shuffle=True,batch_size=config.batch_size,num_workers=0,pin_memory=True,
    #                                   drop_last=False,collate_fn=ssd_dataset_collate)
    #     val_dataloader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
    #                                   drop_last=False, collate_fn=ssd_dataset_collate)
    #
    #     trainer = Train(MODEL,optimizer)
    #
    #     # 训练及验证开始
    #     for epoch in range(start_epoch,freeze_epoch):
    #         train_val(net,epoch,num_train_lines,num_val_linse,train_dataloader,val_dataloader)
    #         lr_scheduler.step()