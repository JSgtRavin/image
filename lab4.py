import cv2
import numpy as np
import os
import math
from math import floor

# TODO: rearrange output data as arrays
# TODO: rewrite function to process three channels simultaneously

def mse(arr1, arr2):
    diff = arr1 - arr2
    return np.sum(diff * diff) / arr1.shape[0] / arr1.shape[1]

def psnr(arr1, arr2):
    mean_se = mse(arr1, arr2)
    if mean_se == 0:
        return 1
    else:
        return 10 * math.log10(255 * 255 / mean_se)

def ssim(img1, img2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def lab_test(image, scale = 2):
    halved = cv2.resize(initial,
                        (int(floor(initial.shape[0] / scale)), int(floor(initial.shape[1] / scale))),
                        interpolation = cv2.INTER_AREA
    )

    nearest = cv2.resize(halved,
                         dsize = (halved.shape[0] * scale, halved.shape[1] * scale),
                         interpolation = cv2.INTER_NEAREST
    )
    linear = cv2.resize(halved,
                         dsize = (halved.shape[0] * scale, halved.shape[1] * scale),
                         interpolation = cv2.INTER_LINEAR
    )
    cubic = cv2.resize(halved,
                         dsize = (halved.shape[0] * scale, halved.shape[1] * scale),
                         interpolation = cv2.INTER_CUBIC
    )
    lanczos = cv2.resize(halved,
                         dsize = (halved.shape[0] * scale, halved.shape[1] * scale),
                         interpolation = cv2.INTER_LANCZOS4
    )

    cv2.imwrite(img_path + img_name + "_nearest" + img_ext, nearest)
    cv2.imwrite(img_path + img_name + "_linear" + img_ext, linear)
    cv2.imwrite(img_path + img_name + "_cubic" + img_ext, cubic)
    cv2.imwrite(img_path + img_name + "_lancoz" + img_ext, lanczos)

    _, initial_g, _ = cv2.split(initial)
    _, nearest_g, _ = cv2.split(nearest)
    _, linear_g, _ = cv2.split(linear)
    _, cubic_g, _ = cv2.split(cubic)
    _, lanczos_g, _ = cv2.split(lanczos)

    MSE = [
        mse(initial_g, nearest_g),
        mse(initial_g, linear_g),
        mse(initial_g, cubic_g),
        mse(initial_g, lanczos_g)
    ]
    PSNR = [
        psnr(initial_g, nearest_g),
        psnr(initial_g, linear_g),
        psnr(initial_g, cubic_g),
        psnr(initial_g, lanczos_g)
    ]
    SSIM = [
        ssim(initial_g, nearest_g),
        ssim(initial_g, linear_g),
        ssim(initial_g, cubic_g),
        ssim(initial_g, lanczos_g)
    ]
    return [MSE, PSNR, SSIM]


img_path = os.getcwd() + "\\img\\"
img_name = "tst_small"
img_ext = ".png"
initial = cv2.imread(img_path + img_name + img_ext)

w = initial.shape[0]
h = initial.shape[1]

while w % 8 > 0:
    w -= 1
while h % 8 > 0:
    h -= 1

initial = initial[0:h, 0:w]

cv2.imwrite(img_path + img_name + "_cropped" + img_ext, initial)

hlf = lab_test(initial, 2)
qrt = lab_test(initial, 4)
oct = lab_test(initial, 8)

with open('111.txt', "w") as file:
    file.write()
    file.write(hlf)
    file.write('\n')