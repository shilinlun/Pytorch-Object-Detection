import torch.nn as nn

class Train(nn.Module):
    def __init__(self,net,optimizer,):
        super(Train, self).__init__()
        self.net = net
        self.optimizer = optimizer

    def forward(self, images,target,loss_funtion): # loss_funtion是对output和target求解
        output = self.net(images)
        # print(output[0].shape)torch.Size([1, 33, 13, 13])
        # print(output[1].shape)torch.Size([1, 33, 26, 26])
        # print(output[2].shape)torch.Size([1, 33, 52, 52])
        total_loss = []
        for i in range(3):
            total_loss.append(loss_funtion(output[i],target))
        total_loss = sum(total_loss)
        return total_loss

    def step(self,images,target,loss_funtion):
        self.optimizer.zero_grad()
        total_loss = self.forward(images,target,loss_funtion)
        self.optimizer.step()
        return total_loss
