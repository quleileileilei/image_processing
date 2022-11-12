
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
img = cv.imread('camera.jpg', 0)  # 读入图片
f = np.fft.fft2(img, axes=(0, 1))
fshift = np.fft.fftshift(f)
res = np.log(np.abs(fshift))  # img的幅度谱
ag = np.angle(fshift)  # img的相位谱


ishift2 = np.fft.ifftshift(fshift)  # 整体逆变换
iimg2 = np.fft.ifft2(ishift2)
iimg2 = np.abs(iimg2)


# 分离
def magnitude_phase_split(img):
    # 分离幅度谱与相位谱
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    # 幅度谱
    magnitude_spectrum = np.abs(dft_shift)
    # 相位谱
    phase_spectrum = np.angle(dft_shift)
    return magnitude_spectrum, phase_spectrum


# 交换相位
def magnitude_phase_combine(img_m, img_p):
    # 幅度谱与相位谱结合
    img_mandp = img_m * np.e ** (1j * img_p)
    img_mandp = np.uint8(np.abs(np.fft.ifft2(img_mandp)))
    img_mandp = img_mandp / np.max(img_mandp) * 255
    return img_mandp


# 分离图像的幅度谱和相位谱
img1_m, img1_p = magnitude_phase_split(img)
# img2_m, img2_p = magnitude_phase_split(img2)
# 相位为0
img2_p = 0
# 幅度谱为常数
img3_m = 5000
# 将图像1的幅度谱与相位谱为0结合
img_1mAnd2p = magnitude_phase_combine(img1_m, img2_p)
# 将常数幅度谱与1图像的相位谱结合
img_2mAnd1p = magnitude_phase_combine(img3_m, img1_p)

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')  # 原图
plt.axis('off')
plt.axis('off')
plt.subplot(222), plt.imshow(iimg2, 'gray'), plt.title('Global inverse transformation')  # 整体逆变换
plt.axis('off')
plt.subplot(223), plt.imshow(img_1mAnd2p, 'gray'), plt.title('1 fudu 0 xiangwei')  # 1幅度加0相位
plt.axis('off')
plt.subplot(224), plt.imshow(img_2mAnd1p, 'gray'), plt.title('A fudu 1 xiangwei')  # A幅度加1相位
plt.axis('off')
plt.show()
