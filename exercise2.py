import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from first_seminar import ColorTranslator


path = 'images/'
img = np.asarray(Image.open(path+"amazonas.jpg"))

print(img.shape) 

c1 = ColorTranslator(img)
yuv = c1.rgb_to_yuv()


c2 = ColorTranslator(yuv)
rgb = c2.yuv_to_rgb()

print(img[:, :, :][100][100])
print(rgb[:, :, :][100][100])


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(yuv[:, :, 0], cmap='gray')
axes[0].set_title("Y Channel (Luma)")
axes[0].axis("off")

axes[1].imshow(rgb)
axes[1].set_title("Reconstructed RGB")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# Round-trip RGB -> YUV -> RGB isn't returning the same values due to:
# - Rounded BT.601 coefficients (3 decimals)
# - Integer clipping to uint8 [0, 255]
# - Floating-point precision limits

