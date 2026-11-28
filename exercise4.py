import subprocess
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
from first_seminar import ColorTranslator

path = "images/"
img_name = "amazonas.jpg"
img_path = f"{path}{img_name}"
scale = 8
other_flag = "neighbor"  

translator = ColorTranslator()

# 1. Default scaling (bicubic) using the class method
translator.resize(scale, img_path, path)
default_out = Path(path) / f"image_scaled_by_{scale}.jpg"

# 2. Second image with nearest-neighbor via -sws_flags
alt_out = Path(path) / f"image_scaled_by_{scale}_{other_flag}.jpg"
cmd = (
    f'ffmpeg -y -i "{img_path}" -sws_flags {other_flag} '
    f'-vf "scale=iw/{scale}:ih/{scale}" "{alt_out}"'
)
subprocess.run(cmd, check=True, shell=True)

# 3. Open and display both for comparison
img_default = Image.open(default_out)
img_alt = Image.open(alt_out)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_default)
axes[0].set_title(f"Default (bicubic) - scale 1/{scale}")
axes[0].axis("off")

axes[1].imshow(img_alt)
axes[1].set_title(f"-sws_flags {other_flag} - scale 1/{scale}")
axes[1].axis("off")

plt.tight_layout()
plt.show()

'''
FFmpeg uses bicubic scaling by default, whereas -sws_flags neighbor uses nearest-neighbor interpolation. 
In the 1/8 scale comparison of amazonas.jpg, the default bicubic produces a smoother, cleaner result with 
less harsh pixelation, while neighbor looks noticeably blocky and pixelated with stronger aliasing across 
the forest and river edges.
'''