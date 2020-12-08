# coding = utf-8
# Author : shi 
# Description :
import os
import xml.etree.ElementTree as ET

base_path = os.getcwd()

models = ['train','val','test']
classes = ["spring", "bearing", "bolt", "nut", "gasket", "screw"]

for model in models:
    image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/%s.txt'%(model)).read().strip().split() # read()把文件里面的值读出来,再分成一个一个的
    # image_ids = ['3029', '3171', '3037', '3056', '3036', '3061', '3145', '3160'...]
    result_file = open('%s.txt'%(model),'w')
    for image_id in image_ids:
        # 先写入图片id所在的路径
        result_file.write('%s/VOCdevkit/VOC2007/JPEGImages/%s.jpg'%(base_path,image_id))
        # 接着写入该图片xml中的信息，主要是用了Etree
        # 找到该id所在的xml
        image_id_xml = open('VOCdevkit/VOC2007/Annotations/%s.xml'%(image_id))
        tree = ET.parse(image_id_xml)
        root = tree.getroot()
        for obj in root.iter('object'): # 根据xml结构可知道，一个xml中存在几个object,就是存在几个目标
            # 然后去一个一个object中去取出其中的内容
            # 目标的difficult 参数
            difficult = 0
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text

            # 目标的name 参数
            name = obj.find('name').text
            # 下面两句其实可以不要，一般都不会发生
            # if name not in classes or int(difficult) == 1:
            #     continue
            name_id = classes.index(name)
            # 目标的box 参数
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            result_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(name_id))

        result_file.write('\n')
    result_file.close()

    # break