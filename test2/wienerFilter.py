import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2

'''
使用python实现维纳滤波，要求:比较信噪比未知，信噪比已知，图像和噪声自相关函数已知，这三种情况下的图像复原结果。
'''


# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


# 信噪比未知
def inverse(input, PSF, eps):  # 不带参数的维纳滤波就是逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result


# 信噪比已知
def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


image = cv2.imread('camera.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_h = image.shape[0]
img_w = image.shape[1]

# 进行运动模糊处理
PSF = motion_process((img_h, img_w), 60)
blurred = np.abs(make_blurred(image, PSF, 1e-3))


result1 = inverse(blurred, PSF, 1e-3)  # 逆滤波
result2 = wiener(blurred, PSF, 1e-3)  # 维纳滤波

blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)  # 添加噪声,standard_normal产生随机的函数
result3 = inverse(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加随机噪声的图像进行逆滤波
result4 = wiener(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加随机噪声的图像进行维纳滤波


# # 计算噪声的自相关函数
# NP = np.abs(fft.fftn(blurred_noisy)) * np.abs(fft.fftn(blurred_noisy))
# NCORR = np.real(fft.ifftn(NP))
# # 计算信号的自相关函数
# IP = np.abs(fft.fftn(image)) * np.abs(fft.fftn(image))
# ICORR = np.real(fft.ifftn(IP))

# 知道图像和噪声的自相关函数即噪声为高斯噪声
def gaussian_noise(img, mean, sigma):
    """
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    """
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out


guassian_noisy = gaussian_noise(blurred, 0, 0.03)
result5 = inverse(guassian_noisy, PSF, 0.1 + 1e-3)  # 对添加高斯噪声的图像进行逆滤波
result6 = wiener(guassian_noisy, PSF, 0.1 + 1e-3)  # 对添加高斯噪声的图像进行维纳滤波

plt.subplot(5, 3, 1), plt.axis('off'), plt.imshow(image, plt.cm.gray), plt.title('original')
plt.subplot(5, 3, 3), plt.axis('off'), plt.imshow(blurred, plt.cm.gray), plt.title('motion_process')
plt.subplot(5, 3, 4), plt.axis('off'), plt.imshow(result1, plt.cm.gray), plt.title(' inverseFilter')
plt.subplot(5, 3, 6), plt.axis('off'), plt.imshow(result2, plt.cm.gray), plt.title(' wienerFilter')
plt.subplot(5, 3, 7), plt.axis('off'), plt.imshow(blurred_noisy, plt.cm.gray), plt.title('motion_process_noisy')
plt.subplot(5, 3, 9), plt.axis('off'), plt.imshow(guassian_noisy, plt.cm.gray), plt.title('motion_process_Gnoisy')
plt.subplot(5, 3, 10), plt.axis('off'), plt.imshow(result3, plt.cm.gray), plt.title(' inverseFilter_noisy')
plt.subplot(5, 3, 12), plt.axis('off'), plt.imshow(result4, plt.cm.gray), plt.title(' wienerFilter_noisy')
plt.subplot(5, 3, 13), plt.axis('off'), plt.imshow(result5, plt.cm.gray), plt.title(' inverseFilter_Gnoisy')
plt.subplot(5, 3, 15), plt.axis('off'), plt.imshow(result6, plt.cm.gray), plt.title(' wienerFilter_Gnoisy')
plt.show()
