# SSD

引用下别人的结构图



![网络结构](https://pic3.zhimg.com/80/v2-cad15a190c94ac354cfa3616ba305bce_1440w.jpg)



其实对比Faster-RCNN,少了RPN网络,然后使用了多尺度预测,从上面结构可以知道,采用了6个尺度,然后分别得到8732个anchors,重点可以看下对应的loss计算.
