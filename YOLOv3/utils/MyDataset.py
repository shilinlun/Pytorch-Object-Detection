from random import shuffle
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data.dataset import Dataset
from PIL import ImageDraw,ImageEnhance

import cv2

# https://www.cnblogs.com/gongxijun/p/6117588.html

def huatu_kuang(img,box):
    a = ImageDraw.ImageDraw(img)
    '''
    ---------------------------------------> x
    |               w
    |    ------------------------
    |    |                      |       
    |  h |                      |
    |    |                      |
    |    -----------------------
    v
    y 
    '''
    for i in range(len(box)):
        a.rectangle([(box[i][0],box[i][1]),(box[i][2],box[i][3])],fill=None,outline='red',width=1) #左上角和右下角
    img.show()

class MyDataset(Dataset):
    def __init__(self, train_lines, image_size):
        super(MyDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def __len__(self):
        return self.train_batches


    def get_random_data(self, annotation_line, resize_shape):
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        rw, rh = resize_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        # huatu_kuang(image,box[0])
        # print(box[0])
        # 首先先对图片resize到416,同时改变了box的位置
        image = image.resize((rw,rh),Image.BICUBIC)
        w_rate = rw / iw
        h_rate = rh / ih
        for i in range(len(box)):
            box[i][0] *= w_rate
            box[i][2] *= w_rate
            box[i][1] *= h_rate
            box[i][3] *= h_rate
        # huatu_kuang(image,box[0])


        mode = np.random.rand() # [0,1)
        # mode设置图片增强方式

        # mode 属于[0,0.2)  水平翻转变换
        if(0<=mode<0.2):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for i in range(len(box)):
                w = box[i][2]-box[i][0]
                box[i][0] = box[i][0] - [w-2*(rw/2-box[i][0])]
                box[i][2] = box[i][0] + w
            # huatu_kuang(image, box)

        # mode 属于[0.2,0.4)  上下翻转变换
        if(0.2<=mode<0.4):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            for i in range(len(box)):
                h = box[i][3]-box[i][1]
                box[i][1] = box[i][1] - [h-2*(rh/2-box[i][1])]
                box[i][3] = box[i][1] + h
            # huatu_kuang(image, box)

        # mode 属于[0.4,0.6)  一般旋转90/270角度，若是其他角度，会扩大box
        if(0.4<=mode<0.6):
            if(np.random.rand()>=0.5):
                image = image.rotate(270, Image.BICUBIC)
                for i in range(len(box)):
                    scale = 1
                     # 逆时针旋转
                    rangle = np.deg2rad(270)
                    nw = (abs(np.sin(rangle) * rh) + abs(np.cos(rangle) * rw)) * scale
                    nh = (abs(np.cos(rangle) * rh) + abs(np.sin(rangle) * rw)) * scale
                    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), 270, scale)
                    rot_move = np.dot(rot_mat, np.array([(nw - rw) * 0.5, (nh - rh) * 0.5, 0]))
                    rot_mat[0, 2] += rot_move[0]
                    rot_mat[1, 2] += rot_move[1]
                    xmin = box[i][0]
                    ymin = box[i][1]
                    xmax = box[i][2]
                    ymax = box[i][3]
                    point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
                    point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
                    point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
                    point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
                    concat = np.vstack((point1, point2, point3, point4))
                    concat = concat.astype(np.int32)
                    x, y, w, h = cv2.boundingRect(concat)
                    rx_min = x
                    ry_min = y
                    rx_max = x + w
                    ry_max = y + h
                    box[i][0] = rx_min
                    box[i][1] = ry_min
                    box[i][2] = rx_max
                    box[i][3] = ry_max
                # huatu_kuang(image, box)
            else:
                image = image.rotate(90, Image.BICUBIC)  # 逆时针旋转
                for i in range(len(box)):
                    scale = 1
                    rangle = np.deg2rad(90)
                    nw = (abs(np.sin(rangle) * rh) + abs(np.cos(rangle) * rw)) * scale
                    nh = (abs(np.cos(rangle) * rh) + abs(np.sin(rangle) * rw)) * scale
                    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), 90, scale)
                    rot_move = np.dot(rot_mat, np.array([(nw - rw) * 0.5, (nh - rh) * 0.5, 0]))
                    rot_mat[0, 2] += rot_move[0]
                    rot_mat[1, 2] += rot_move[1]
                    xmin = box[i][0]
                    ymin = box[i][1]
                    xmax = box[i][2]
                    ymax = box[i][3]
                    point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
                    point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
                    point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
                    point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
                    concat = np.vstack((point1, point2, point3, point4))
                    concat = concat.astype(np.int32)
                    x, y, w, h = cv2.boundingRect(concat)
                    rx_min = x
                    ry_min = y
                    rx_max = x + w
                    ry_max = y + h
                    box[i][0] = rx_min
                    box[i][1] = ry_min
                    box[i][2] = rx_max
                    box[i][3] = ry_max
                # huatu_kuang(image, box)
        # mode 属于[0.6,0.8)  颜色抖动
        if(0.6<=mode<0.8):
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
            # huatu_kuang(image, box[0])

        # mode 属于[0.8,1)  高斯噪声
        if(0.8<=mode<1):
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand()))
            # huatu_kuang(image, box[0])

        # huatu_kuang(image, box[0])
        test = image
        image = np.array(image)
        return image,box,test



    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        image, box,test = self.get_random_data(lines[index], self.image_size[0:2])
        # huatu_kuang(test,box)
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image / 255.0, (2, 0, 1))

        boxes = np.array(box[:, :4], dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] / self.image_size[1]
        boxes[:, 1] = boxes[:, 1] / self.image_size[0]
        boxes[:, 2] = boxes[:, 2] / self.image_size[1]
        boxes[:, 3] = boxes[:, 3] / self.image_size[0]

        boxes = np.maximum(np.minimum(boxes, 1), 0)
        # 转为长和宽
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # 转为中心坐标
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        box = np.concatenate([boxes, box[:, -1:]], axis=-1)
        box = np.array(box, dtype=np.float32)
        return image,box


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes




# img_path = '/home/shi/data/Deep_Learing/Object_Detection/YOLOv3/VOCdevkit/VOC2007/JPEGImages/3000.jpg 237,142,297,276,2 156,226,216,285,4'
# # img_path = '/home/shi/data/Deep_Learing/Object_Detection/YOLOv3/VOCdevkit/VOC2007/JPEGImages/1.jpg 193,237,353,291,0' 237,142,297,276,2 156,226,216,285,4
# mydataset = MyDataset([img_path],[416,416])
# image,box,test= mydataset.get_random_data(img_path,[416,416])
# huatu_kuang(test, box[0])
# print(box)
# # image = np.array(image, dtype=np.float32)
# # image = np.transpose(image / 255.0, (2, 0, 1))
#
# image_size = [416,416]
# boxes = np.array(box[:, :4], dtype=np.float32)
# boxes[:, 0] = boxes[:, 0] / image_size[1]
# boxes[:, 1] = boxes[:, 1] / image_size[0]
# boxes[:, 2] = boxes[:, 2] / image_size[1]
# boxes[:, 3] = boxes[:, 3] / image_size[0]
#
# boxes = np.maximum(np.minimum(boxes, 1), 0)
# boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
# boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
#
# boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
# boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
# box = np.concatenate([boxes, box[:, -1:]], axis=-1)
# box = np.array(box, dtype=np.float32)
# print(box)
# print(0.35817304/0.16105771) 2.223880123466303
# print(0.36057696/0.1634615) 2.2058830978548465