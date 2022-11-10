import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("img.jpg")  # 读取要处理的图片
height = img.shape[0]
width = img.shape[1]
# 平移（右90像素，下90像素>
obj = np.float32([[1, 0, 90], [0, 1, 90]])
imgl = cv.warpAffine(img, obj, (width, height))  # 旋转（逆时针60度）
M = cv.getRotationMatrix2D((width / 2, height / 2), 60, 1)
img2 = cv.warpAffine(img, M, (width, height))
# 水平镜像
img3 = cv.flip(img, 1)  # 放大到1080*840像素
img4 = cv.resize(img, (1080, 840), interpolation=cv.INTER_LINEAR)  # 缩小到256*128
img5 = cv.resize(img, (256, 128), interpolation=cv.INTER_LINEAR)  # 显示各个变换图片
# cv2.cvtColor (p1,p2)是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。#cv2.COLOR_BGR2RGB将BGR格式转换成RGB格式
# cv2.COLOR_BGR2GRAY将BGR格式转换成灰度图片
img2_list = [imgl, img2, img3, img4, img5]
plt.figure(figsize=(10, 10))
for i in range(1, 6):
    plt.subplot(1, 5, i)
    plt.imshow(cv.cvtColor(img2_list[i - 1], cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
plt.show()
# subplots_adjust函数的功能为调整子图的布局参数
# hspace:子图间高度内边距，距离单位为子图平均高度的比例（小数）默认值为0.2
