import numpy as np
import cv2


def calculate_psnr(original_img, denoised_img):
    mse = ((original_img - denoised_img) ** 2).mean()
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def calculate_ssim(original_img, denoised_img):
    ssim = cv2.SSIM(original_img, denoised_img)
    return ssim
