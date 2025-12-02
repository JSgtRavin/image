import cv2
import os
import numpy as np

img_path = r"E:\tst.png"
img_src = cv2.imread(img_path)

filename, fileex = os.path.splitext(img_path)
cv2.imshow("Ambush", img_src)
cv2.waitKey()

print(f"Размер: {img_src.shape}\nРасширение: {fileex}")

cv2.imwrite(f"{filename}.jpeg", img_src)
calc_resize = (img_src.shape[1] // 5, img_src.shape[0] // 5)
img_dbl = cv2.resize(img_src, dsize = calc_resize, interpolation = cv2.INTER_CUBIC)
print("Уменьшенный размер: ", img_dbl.shape)
cv2.imshow("Уменьшенное изображение", img_dbl)
cv2.waitKey()
cv2.imwrite(f"{filename}_double.jpeg", img_dbl)