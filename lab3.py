import math

import numpy as np
import os
import cv2

def noise_gauss(Im, scale = 50):
    row, col, ch = Im.shape
    mean = 0
    gauss = np.random.normal(mean, scale, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noise = Im + gauss
    return np.array(noise, np.uint8)

def mse(arr1, arr2):
    diff = arr1 - arr2
    return np.sum(diff * diff) / arr1.shape[0] / arr1.shape[1]

def psnr(arr1, arr2):
    mean_se = mse(arr1, arr2)
    if mean_se == 0:
        return 1
    else:
        return 10 * math.log10(255 * 255 / mean_se)

img_path = os.getcwd() + "\\img\\"
img_name = "tst_small"
img_ext = ".png"
initial = cv2.imread(img_path + img_name + img_ext)

noised = noise_gauss(initial)

try:
    cv2.imshow("Noised", noised)
    cv2.waitKey()
    cv2.imwrite(img_path + img_name + "_custom_gauss" + img_ext, noised)
except OSError:
    pass

noised = cv2.cvtColor(noised, cv2.COLOR_BGR2GRAY)
noised_80 = cv2.cvtColor(noise_gauss(initial, 120), cv2.COLOR_BGR2GRAY)
noised_30 = cv2.cvtColor(noise_gauss(initial, 10), cv2.COLOR_BGR2GRAY)

initial = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)

MSE = [
    mse(initial, noised_80),
    mse(initial, noised),
    mse(initial, noised_30)
]
PSNR = [
    psnr(initial, noised_80),
    psnr(initial, noised),
    psnr(initial, noised_30)
]

print(MSE)
print(PSNR)