# 图像中的目标定位

**解决方案**

1 使用cv2的归一化相关系数匹配法的模板匹配matchTemplate(), 得到一张灰度图， 灰度图的取值范围是(0, 1), 每个位置的值表示以当前点为左上角坐标的图片和目标图片的相似度得分， 得分越高表示越相似， 根据得分设定阈值可得到与目标图片相似的框

2 由于第一步得到的box很多都是重复的框（两个框之间重叠90%以上）， 所以使用非极大值抑制（nms）删除多余的框

3 第1步中使用的模板匹配方法是将图像灰度之后求取的， 在本例中绿色飞行器和棕色飞行器在灰度化之后， 模板匹配的结果区分不出来， 使用颜色分布直方图特征， 比较第2步中的候选框和目标框之间的距离， 设定阈值， 删除距离较远的图片，得到最后的候选框

![](/data/dong/research/image/myplot.png)

![myplot1](/data/dong/research/image/myplot1.png)

紫色和绿色飞行器通过matchTemplate无法区分, 如下图:

![](/data/dong/research/image/result.jpg)

使用颜色直方图距离之后的效果， 下图是检测绿色飞行器和棕色飞行器的结果： 

![res_tanker_a](/data/dong/research/image1/res_tanker_a.png)

![res_tanker_b](/data/dong/research/image1/res_tanker_b.jpg)



**存在的问题：**

本实验中设置的两个阈值对检测结果影响很大， 两个阈值分别是matchTemplate的阈值threshold_score以及颜色特征距离的阈值threshold_dist， 在matchTemplate中,  只有得分超过threshold_score的框被保留下来， 在求指定图片和检测得到的框的颜色特征距离时，只有距离低于threshold_dist的框才被保留下来。

在检测绿色飞行器(tanker_a.png)时， threshold_score=0.9, threshold_dist=0.4, 才能得到正确的结果

在检测棕色飞行器(tanker_b.jpg)时， threshold_score=0.8, threshold_dist=0.6， 才能得到正确结果

因此检测不同的目标对象， 需要根据实际情况调整阈值，才能得到想要的结果



**其他思路：**

可以使用预训练好的模型，如inception, resnet等网络， 提取最后一层的值作为图片特征，实现图片的分类