# Faster-RCNN

![Faster-RCNN结构](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMDEyMTE0ODEyNTk1?x-oss-process=image/format,png)



上面就是Faster-RCNN结构，主要包括三部分（Feature提取、RPN网络、ROIPooling及后面的回归分类）

以下是我自己的一个见解：

首先一张图片输入到feature网络后，得到feature map，然后在feature map上面产生 $38\times38\times9$ 个anchors，然后首先构造分类网络和回归网络，对这些anchors进分类得分，选前面多少名的框进行回归，这样相当于训练的参数就是对框的分类和对框的回归，当输入新的框之后，就会自动得到那些框是目标框，并自动回归到真实位置。

## 具体做法其实网上有很多教程，因为本程序只能实现训练，且并没有加载预训练模型，所以只需按照以下步骤进行即可

1. voc2frcnn.py

2. voc2annotation.py

3. train.py



