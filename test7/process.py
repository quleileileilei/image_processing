# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('img.jpg')

# 图像灰度转换
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 获取图像高度和宽度
height = grayImage.shape[0]
width = grayImage.shape[1]

# 创建一幅图像
result1 = np.zeros((height, width), np.uint8)
result2 = np.zeros((height, width), np.uint8)

# 图像线性处理
for i in range(height):
    for j in range(width):
        if (int(grayImage[i, j] + 50) > 255):
            gray = 255
        else:
            gray = int(grayImage[i, j] + 50)

        result1[i, j] = np.uint8(gray)

# 图像灰度非线性变换：DB=DA×DA/255
for i in range(height):
    for j in range(width):
        gray = int(grayImage[i, j]) * int(grayImage[i, j]) / 255
        result2[i, j] = np.uint8(gray)


# 图像分段线性处理
def linear_transform(img):
    height, width = img.shape[:2]
    r1, s1 = 80, 10
    r2, s2 = 140, 200
    k1 = s1 / r1  # 第一段斜率
    k2 = (s2 - s1) / (r2 - r1)  # 第二段斜率
    k3 = (255 - s2) / (255 - r2)  # 第三段斜率
    img_copy = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            if any(img[i, j] )< r1:
                img_copy[i, j] = k1 * img[i, j]
            elif r1 <= img[i, j] <= r2:
                img_copy[i, j] = k2 * (img[i, j] - r1) + s1
            else:
                img_copy[i, j] = k3 * (img[i, j] - r2) + s2
    return img_copy


result3 = linear_transform(img)

# 显示图像
cv2.imshow("Gray Image", grayImage)
cv2.imshow("Result1", result1)
cv2.imshow("Result2", result2)
cv2.imshow("Result3", result3)

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
