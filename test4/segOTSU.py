import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

'''大津法
Summary:大津阈值分割
Paramaters：img - 输入的灰度图像 是二维矩阵
'''


def OTSU(img):
    # 类间方差g初始最小
    g_raw = -1
    # 要返回的阈值
    T_return = 0
    # 获得图像大小
    M_N = img.shape[0] * img.shape[1]
    # 大津阈值算法
    for T in range(256):
        # 获取阈值大于T和小于T的两个列表
        array0 = img[img < T]
        array1 = img[img > T]
        # 算出w0和w1
        w0 = len(array0) / M_N  # 公式1
        w1 = len(array1) / M_N  # 公式2
        # 算出μ0和μ1 这里需要特判除数为0
        if len(array0) == 0:
            mu0 = 0
        else:
            mu0 = sum(array0) / len(array0)  # 公式3
        if len(array1) == 0:
            mu1 = 0
        else:
            mu1 = sum(array1) / len(array1)  # 公式4
        # 算出g
        g = w0 * w1 * math.pow((mu0 - mu1), 2)  # 公式6
        if g > g_raw:
            g_raw = g
            T_return = T
    return T_return


'''大津法读取'''

# 读取图片
img = Image.open('img.jpg')
# 转换成灰度图
img = img.convert('L')
# 转换成array
arr = np.array(img)
# # 获得最佳域值分割
T = OTSU(arr)
print(f"Best threshold = {T}")
# 根据阈值进行二值分割
arr[arr > T] = 255
arr[arr < T] = 0
# 展示分割结果
pil_image = Image.fromarray(arr)
plt.figure()
plt.imshow(img, cmap='gray')
plt.title("original picture")
plt.figure()
plt.imshow(pil_image, cmap='gray')
plt.title("picture by OTSU")

plt.show()
