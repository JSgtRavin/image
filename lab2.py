from PIL import Image
from PIL import ImageFilter
import numpy as np
import os

img_path = os.getcwd() + "\\img\\"
img_name = "tst_small"
img_ext = ".png"
initial = Image.open(img_path + img_name + img_ext)
filtered = initial.filter(ImageFilter.GaussianBlur(3))
filtered.show()
filtered.save(img_path + img_name + "_filtered" + img_ext)

initial_a = np.array(initial)
filtered_a = np.array(filtered)

mask = (initial_a - filtered_a / 2) * 2
mask[mask < 0] = 0
mask[mask >= 255] = 255

masked = Image.fromarray(np.uint8(mask))
try:
    masked.save(img_path + img_name + "_masked" + img_ext)
    masked.show()
except OSError:
    pass