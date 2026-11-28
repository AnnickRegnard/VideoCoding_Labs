import os
from humanize import naturalsize
from first_seminar import ColorTranslator
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

path = 'images/'

size411 = os.stat(path + 'amazonas_411.yuv').st_size
size420 = os.stat(path + 'amazonas_420.yuv').st_size
size422 = os.stat(path + 'amazonas_422.yuv').st_size
size444 = os.stat(path + 'amazonas_444.yuv').st_size

print("size of 4:4:4 is "+ naturalsize(size444))
print("size of 4:2:2 is "+ naturalsize(size422))
print("size of 4:2:0 is "+ naturalsize(size420))
print("size of 4:1:1 is "+ naturalsize(size411))


img = np.asarray(Image.open(path+"amazonas.jpg"))

yuv = ColorTranslator(img).rgb_to_yuv()
c1 = yuv.copy()
c2 = yuv.copy()

blur = ImageFilter.GaussianBlur(radius=8)

c1[:, :, 0] = np.asarray(Image.fromarray(c1[:, :, 0]).filter(blur))
c2[:, :, 1] = np.asarray(Image.fromarray(c2[:, :, 1]).filter(blur))
c2[:, :, 2] = np.asarray(Image.fromarray(c2[:, :, 2]).filter(blur))

rgb1 = ColorTranslator(c1).yuv_to_rgb()
rgb2 = ColorTranslator(c2).yuv_to_rgb()


fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(rgb1)
axes[0].set_title("Blur in Luma")
axes[0].axis("off")

axes[1].imshow(rgb2)
axes[1].set_title("Blur in Chromas")
axes[1].axis("off")

axes[2].imshow(img)
axes[2].set_title("Original Image")
axes[2].axis("off")

plt.tight_layout()
plt.show()


'''
Blurring luma (Y) is immediately noticeable because the human eye is much more sensitive to brightness 
and fine spatial detail than to color. JPEG and video codecs exploit this by subsampling or compressing 
chroma (U, V) more aggressively, saving significant bandwidth and storage with minimal perceived quality loss.
'''


