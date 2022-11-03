# -*- coding=GBK -*-
import cv2 as cv
import numpy as np


# ��ˮ���㷨
def water_image():
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)  # ȥ�����
    # =========ȷ��ǰ������==========
    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # ת�Ҷ�ͼ
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # ��ֵ��
    cv.imshow("Bibarization", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # ��̬ѧ������ �ȸ�ʴ������
    sure_bg = cv.dilate(mb, kernel, iterations=3)  # ����
    cv.imshow("Operation_Morphological", sure_bg)
    # =============================
    # distance transform
    dist = cv.distanceTransform(sure_bg, cv.DIST_L2, 3)  # ��ȡǰ��
    # dist = cv2.distanceTransform(src=gaussian_hsv, distanceType=cv2.DIST_L2, maskSize=5) ����任����
    # dist �C ���м����������ͼ������һ���� src ��С��ͬ�� 32 λ���㵥ͨ��ͼ��
    # src �C 8 λ����ͨ���������ƣ�Դͼ��
    # distanceType �C �������͡��������� CV_DIST_L1��CV_DIST_L2 �� CV_DIST_C��
    # maskSize �C ����任����Ĵ�С���������� 3��5 �� CV_DIST_MASK_PRECISE����һ��ѡ����ɵ�һ������֧�֣���
    #     �� CV_DIST_L1 �� CV_DIST_C �������͵�����£�������ǿ��Ϊ 3����Ϊ 3\times 3 ��������� 5\times 5 ���κθ���׾���ͬ�Ľ����
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)  # ��һ����0~1֮��
    cv.imshow("Distance_Transformation", dist_output * 70)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)
    cv.imshow("Find_Seed", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)

    ret, markers = cv.connectedComponents(surface_fg)
    # ret: ���������ͨ��  ��ͨ�����ɾ�����ͬ����ֵ����������������ؼ���
    # makers����ͼ��ı������Ϊ0
    print(ret)
    markers += 1  # OpenCV ��ˮ���㷨���������ı�ע���붼����1������Ϊ���Ϊ0
    markers[unknown == 255] = 0

    markers = cv.watershed(src, markers)  # ��ˮ���㷨���������������ص㱻��עΪ -1
    src[markers == -1] = [0, 0, 255]
    cv.imshow("Watershed_results", src)


src = cv.imread("img.jpg")
cv.imshow("before", src)
water_image()
cv.waitKey(0)
cv.destroyAllWindows()