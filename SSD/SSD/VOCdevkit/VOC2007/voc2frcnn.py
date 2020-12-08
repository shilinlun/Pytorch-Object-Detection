import os
import random

xml_path = "/home/shi/data/Deep_Learing/Object_Detection/SSD/VOCdevkit/VOC2007/Annotations"
result_path = "/home/shi/data/Deep_Learing/Object_Detection/SSD/VOCdevkit/VOC2007/ImageSets/Main/"

# 获得xml_path路径下xml文件的个数
xmls = os.listdir(xml_path)
num_xmls = len(xmls)
print(num_xmls)

# 总数据集分为trainval和test,trainval分为train和val
#训练及交叉验证的比列
trainval_percent = 1
trainval_num = int(num_xmls * trainval_percent)
# 训练及交叉验证的样本
# 返回的是样本的下标
# list = [0,1,2,3,4]
# rs = random.sample(list, 2)
# rs = [2,4]
trainval = random.sample(range(len(xmls)),trainval_num)
# print(trainval)

#训练集比列
train_percent = 1
train_num = int(trainval_num * train_percent)
train = random.sample(range(len(xmls)),train_num)
# print(train)

# 把xml写入到train test val trainval

file_tainval = open(os.path.join(result_path,'trainval.txt'),'w')
file_train = open(os.path.join(result_path,'train.txt'),'w')
file_val = open(os.path.join(result_path,'val.txt'),'w')
file_test = open(os.path.join(result_path,'test.txt'),'w')

for i in range(len(xmls)):
    name = xmls[i][:-4] + '\n'  # xmls格式为 xxx.xml,所以name就是xxx
    if i in trainval:
        file_tainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)

    else:
        file_test.write(name)

file_test.close()
file_val.close()
file_train.close()
file_tainval.close()
