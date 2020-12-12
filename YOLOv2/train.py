from Constant.Config import config
from nets.yolo2 import yolo2
import torch
import torch.backends.cudnn as cudnn
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset,detection_collate
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo2train import Train
from tqdm import tqdm
from torch.autograd import Variable

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_dataset(datasetnames):
    names = datasetnames
    dataset = RoiDataset(get_imdb(names))
    return dataset

def train_val(net,epoch, num_train_lines, train_data_iter,num_val_lines,val_data_iter):
    # 训练总的损失
    total_train_loss = 0
    # 回归损失
    total_box_loss = 0
    # 分类损失
    total_class_loss = 0
    # 置信损失
    total_iou_loss = 0
    # 验证总的损失
    total_val_loss = 0


    with tqdm(desc='Current Epoch:{}'.format(epoch),total=num_train_lines,postfix=dict,mininterval=0.1) as pbar:
        for step in range(num_train_lines):
            im_data, boxes, gt_classes, num_obj = next(train_data_iter)
            im_data = im_data.cuda()
            boxes = boxes.cuda()
            gt_classes = gt_classes.cuda()
            num_obj = num_obj.cuda()
            im_data_variable = Variable(im_data)
            box_loss, iou_loss, class_loss,total_loss = train_util.train_step(im_data_variable, boxes, gt_classes, num_obj)
        # for step,data in enumerate(train_dataloader):
        #     img,boxes,labels = data[0],data[1],data[2]
        #     # 由于这三项是确定的，不需要梯度下降
        #     with torch.no_grad():
        #         img = Variable(torch.from_numpy(img).type(torch.FloatTensor)).cuda()
        #         boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
        #         labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
        #
        #     box_loss, iou_loss, class_loss = model.forward(im_data_variable, boxes, gt_classes, num_obj, training=True)
        #     rpn_huigui_loss,rpn_fenlei_loss,roi_huigui_loss,roi_fenlei_loss,total_loss = trainer.train_step(img,boxes,labels,scale=1)
        #
            total_train_loss += total_loss
            total_box_loss += box_loss
            total_iou_loss += iou_loss
            total_class_loss += class_loss

            pbar.set_postfix(**{'total': total_train_loss.item() / (step + 1),
                                'box_loss': total_box_loss.item() / (step + 1),
                                'iou_loss': total_iou_loss.item() / (step + 1),
                                'class_loss': total_class_loss.item() / (step + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('\nStart Validation')
    with tqdm(total=num_val_lines, desc=f'Current Epoch {epoch}',postfix=dict,mininterval=0.1) as pbar:
        for step in range(num_val_lines):
            im_data, boxes, gt_classes, num_obj = next(val_data_iter)
            im_data = im_data.cuda()
            boxes = boxes.cuda()
            gt_classes = gt_classes.cuda()
            num_obj = num_obj.cuda()
            im_data_variable = Variable(im_data)
            _, _, _,val_loss = train_util.train_step(im_data_variable, boxes, gt_classes, num_obj)
        # for step,data in enumerate(train_dataloader):
        #     img,boxes,labels = data[0],data[1],data[2]
        #     # 由于这三项是确定的，不需要梯度下降
        #     with torch.no_grad():
        #         img = Variable(torch.from_numpy(img).type(torch.FloatTensor)).cuda()
        #         boxes = [Variable(torch.from_numpy(box).type(torch.FloatTensor)).cuda() for box in boxes]
        #         labels = [Variable(torch.from_numpy(label).type(torch.FloatTensor)).cuda() for label in labels]
        #
        #     box_loss, iou_loss, class_loss = model.forward(im_data_variable, boxes, gt_classes, num_obj, training=True)
        #     rpn_huigui_loss,rpn_fenlei_loss,roi_huigui_loss,roi_fenlei_loss,total_loss = trainer.train_step(img,boxes,labels,scale=1)
        #
            total_val_loss += val_loss
            # total_box_loss += box_loss
            # total_iou_loss += iou_loss
            # total_class_loss += class_loss

            pbar.set_postfix(**{'val': total_val_loss.item() / (step + 1),
                                # 'rpn_loc': total_box_loss.item() / (step + 1),
                                # 'rpn_cls': total_iou_loss.item() / (step + 1),
                                # 'roi_loc': total_class_loss.item() / (step + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('\nFinish Validation')
    print('Epoch:'+ str(epoch))
    print('Train Loss: %.4f || Val Loss: %.4f ' % (total_train_loss/(num_train_lines+1),total_val_loss/(num_val_lines+1)))
    print('Saving state, iter:', str(epoch))
    torch.save(net.state_dict(), 'logs/Epoch%d-Train_Loss%.4f-Val_Loss%.4f.pth'%((epoch),total_train_loss/(num_train_lines+1),total_val_loss/(num_val_lines+1)))


if __name__ == '__main__':
    # 类别
    NUM_CLASSES = config.num_classes
    model = yolo2()
    IMAGE_SIZE = [416,416,3]
    device = torch.device('cuda')
    model_dict = model.state_dict()
    model.load_state_dict(model_dict)
    print('Load Finished')

    # 设置训练模式
    net = model.train()
    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异
    cudnn.benchmark = True
    # 加载到GPU
    net = net.to(device)

    # load dataset
    print('loading dataset....')
    train_imdb_name = 'voc_2007_train'
    val_imdb_name = 'voc_2007_val'
    # train_dataset = get_dataset(imdb_name)


    # train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
    #                               shuffle=True, num_workers=config.num_workers,
    #                               collate_fn=detection_collate, drop_last=True)

    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                          momentum=config.momentum, weight_decay=config.weight_decay)

    if True:
        lr = config.lr # 0.00001
        Start_Epoch = config.start_epoch # 50
        Freeze_Epoch = config.freeze_epoch # 100
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        train_dataset = get_dataset(train_imdb_name)
        val_dataset = get_dataset(val_imdb_name)
        print('dataset loaded.')
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers,
                                      collate_fn=detection_collate, drop_last=True)


        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers,
                                      collate_fn=detection_collate, drop_last=True)

        # train_data = MyDataset(train_val_lines[:num_train_lines],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        # val_data = MyDataset(train_val_lines[num_train_lines:],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        # train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
        #                  drop_last=True, collate_fn=frcnn_dataset_collate)
        # val_dataloader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
        #                      drop_last=True, collate_fn=frcnn_dataset_collate)



        # 解冻前训练

        for param in model.conv1.parameters():
            param.requires_grad = False


        # MODEL.freeze_bn()

        train_util = Train(net,optimizer)
        num_train_lines = int(len(train_dataset) / config.batch_size)
        num_val_lines = int(len(val_dataset) / config.batch_size)

        for epoch in range(Start_Epoch, Freeze_Epoch):
            train_data_iter = iter(train_dataloader)
            val_data_iter = iter(val_dataloader)
            train_val(net,epoch, num_train_lines,train_data_iter,num_val_lines,val_data_iter)
            lr_scheduler.step()
    if True:

        lr = config.lr # 0.00001
        Freeze_Epoch = config.freeze_epoch # 50
        End_Epoch = config.end_epoch # 100
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        train_dataset = get_dataset(train_imdb_name)
        val_dataset = get_dataset(val_imdb_name)
        print('dataset loaded.')
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers,
                                      collate_fn=detection_collate, drop_last=True)


        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers,
                                      collate_fn=detection_collate, drop_last=True)

        # train_data = MyDataset(train_val_lines[:num_train_lines],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        # val_data = MyDataset(train_val_lines[num_train_lines:],[IMAGE_SIZE[0],IMAGE_SIZE[1]])
        # train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
        #                  drop_last=True, collate_fn=frcnn_dataset_collate)
        # val_dataloader = DataLoader(val_data, shuffle=True, batch_size=config.batch_size, num_workers=0, pin_memory=True,
        #                      drop_last=True, collate_fn=frcnn_dataset_collate)



        # 解冻前训练

        for param in model.conv1.parameters():
            param.requires_grad = True


        # MODEL.freeze_bn()

        train_util = Train(net,optimizer)
        num_train_lines = int(len(train_dataset) / config.batch_size)
        num_val_lines = int(len(val_dataset) / config.batch_size)

        for epoch in range(Freeze_Epoch, End_Epoch):
            train_data_iter = iter(train_dataloader)
            val_data_iter = iter(val_dataloader)
            train_val(net,epoch, num_train_lines,train_data_iter,num_val_lines,val_data_iter)
            lr_scheduler.step()