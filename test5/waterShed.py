# -*- coding=GBK -*-
import cv2 as cv
import numpy as np


# 分水岭算法
def water_image():
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)  # 去除噪点
    # =========确定前景对象==========
    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # 转灰度图
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    cv.imshow("Bibarization", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # 形态学开操作 先腐蚀后膨胀
    sure_bg = cv.dilate(mb, kernel, iterations=3)  # 膨胀
    cv.imshow("Operation_Morphological", sure_bg)
    # =============================
    # distance transform
    dist = cv.distanceTransform(sure_bg, cv.DIST_L2, 3)  # 提取前景
    # dist = cv2.distanceTransform(src=gaussian_hsv, distanceType=cv2.DIST_L2, maskSize=5) 距离变换函数
    # dist – 具有计算距离的输出图像。它是一个与 src 大小相同的 32 位浮点单通道图像。
    # src – 8 位、单通道（二进制）源图像。
    # distanceType – 距离类型。它可以是 CV_DIST_L1、CV_DIST_L2 或 CV_DIST_C。
    # maskSize – 距离变换掩码的大小。它可以是 3、5 或 CV_DIST_MASK_PRECISE（后一个选项仅由第一个函数支持）。
    #     在 CV_DIST_L1 或 CV_DIST_C 距离类型的情况下，参数被强制为 3，因为 3\times 3 掩码给出与 5\times 5 或任何更大孔径相同的结果。
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)  # 归一化在0~1之间
    cv.imshow("Distance_Transformation", dist_output * 70)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("Find_Seed", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)

    ret, markers = cv.connectedComponents(surface_fg)
    # ret: 计算最大连通域  连通域：是由具有相同像素值的相邻像素组成像素集合
    # makers：将图像的背景标记为0
    print(ret)
    markers += 1  # OpenCV 分水岭算法对物体做的标注必须都大于1，背景为标号为0
    markers[unknown == 255] = 0

    markers = cv.watershed(src, markers)  # 分水岭算法后，所有轮廓的像素点被标注为 -1
    src[markers == -1] = [0, 0, 255]
    cv.imshow("Watershed_results", src)


src = cv.imread("img.jpg")
cv.imshow("before", src)
water_image()
cv.waitKey(0)
cv.destroyAllWindows()