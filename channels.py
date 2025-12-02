import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

img_path = "E:\\python\\img\\tst.png"
filedir = os.path.dirname(os.path.abspath(img_path))
img_src = cv2.imread(img_path)

filename, fileex = os.path.splitext(img_path)
#cv2.imshow("Ambush", img_src)
cv2.waitKey()

print(f"Размер: {img_src.shape}\nРасширение: {fileex}")

calc_resize = (img_src.shape[1] // 5, img_src.shape[0] // 5)
img_small = cv2.resize(img_src, dsize = calc_resize, interpolation = cv2.INTER_CUBIC)
print("Уменьшенный размер: ", img_small.shape)
cv2.imwrite(f"{filename}_small{fileex}", img_small)
filename = filename + "_small"

# Задание 1.
# Загрузите цветное изображение, разделите его на отдельные каналы,
# сохраните каналы как отдельные изображения на диск.
img_src = cv2.imread(f"{filename}{fileex}")
cv2.imshow("Ambush", img_src)
cv2.waitKey()

red_blur = np.zeros_like(img_src [:, 0])
##Три канала исходного изображения
channels_bgr = cv2.split(img_src) # Три канала
names = ('b', 'g', 'r')

for channel, name in zip(channels_bgr, names):
    cv2.imwrite(f"{filename}_{name}{fileex}", channel)
    if name == "r":
        red_blur = cv2.blur(channel, (15, 15))
        cv2.imwrite(f"{filename}_{name}_blur{fileex}", red_blur)

cv2.imshow("Merged image", cv2.merge(channels_bgr))
cv2.waitKey()

# Задание 2
# Примените фильтр blur для красного канала и восстановите цветное изображение.
# Проанализируйте как изменилось изображение.
img_red_blur = cv2.merge((channels_bgr[0], channels_bgr[1], red_blur))
cv2.imwrite(f"{filename}_red_blurred{fileex}", img_red_blur)
cv2.imshow("Red blur", img_red_blur)
cv2.waitKey()

# Задание 3.
# Отобразите и сохраните гистограммы каналов используя библиотеку matplotlib.
hist = cv2.calcHist([img_red_blur], [2], None, [256], [0, 256])
plt.plot(hist)
plt.title("Гистограмма изображения")
matplotlib.use("Qt5Agg")
plt.show()
# plt.savefig(f"{filedir}\\hist_r.png", dpi=400)
cv2.waitKey()