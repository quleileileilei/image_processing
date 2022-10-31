# import cv2 as cv
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
# '''迭代法阈值分割'''
# # 读入图片并转化为矩阵
# img = plt.imread('img.jpg')
# im = np.array(img)
# # 矩阵大小
# l = len(im)
# w = len(im[0])
# # 求初始阈值
# zmin = np.min(im)
# zmax = np.max(im)
# t0 = int((zmin + zmax) / 2)
# # 初始化相关变量初始化
# t1 = 0
# res1 = 0
# res2 = 0
# s1 = 0
# s2 = 0
# # 迭代法计算最佳阈值
# while abs(t0 - t1) > 0:
#     for i in range(0, l - 1):
#         for j in range(0, w - 1):
#             if all(im[i, j] < t0):
#                 res1 = res1 + im[i, j]
#                 s1 = s1 + 1
#             elif all(im[i, j] > t0):
#                 res2 = res2 + im[i, j]
#                 s2 = s2 + 1
#     avg1 = res1 / s1
#     avg2 = res2 / s2
#     res1 = 0
#     res2 = 0
#     s1 = 0
#     s2 = 0
#     t1 = t0  # 旧阈值储存在t1中
#     t0 = int((avg1 + avg2) / 2)  # 计算新阈值
#
# # 阈值化分割
# # 像素点灰度值小于最佳阈值t0用0填充，其余用255填充
# im = np.where(im[..., :] < t0, 0, 255)
#
# # 绘制原图窗口
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.title('original')
#
# # 绘制阈值化分割后图像
# plt.figure()
# plt.imshow(Image.fromarray(im), cmap='gray')
# plt.title('new')
#
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("picture by Iteration")
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("picture by Iteration")
    plt.show()


img = cv.imread('img.jpg', 0)

T = img.mean()

while True:
    t0 = img[img < T].mean()
    t1 = img[img >= T].mean()
    t = (t0 + t1) / 2
    if abs(T - t) < 1:
        break
    T = t
T = int(T)

print(f"Best threshold = {T}")
th, img_bin = cv.threshold(img, T, 255, 0)

show(img_bin)
