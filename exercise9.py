import os
import re
import subprocess
import numpy as np
import pywt
from PIL import Image


class DWTProcessor:
    """Class that encodes and decodes a 2D image using the Discrete Wavelet Transform (DWT)."""

    def __init__(self, wavelet="haar"):
        self.wavelet = wavelet

    def encode(self, image_2d):
        """
        Perform 1-level 2D DWT decomposition.
        Returns (LL, (LH, HL, HH)) subbands.
        """
        coeffs2 = pywt.dwt2(image_2d.astype(np.float64), self.wavelet)
        LL, (LH, HL, HH) = coeffs2
        return LL, LH, HL, HH

    def decode(self, LL, LH, HL, HH):
        """Reconstruct the 2D image from the four DWT subbands."""
        reconstructed = pywt.idwt2((LL, (LH, HL, HH)), self.wavelet)
        return np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)

    def save_subbands_grid(self, LL, LH, HL, HH, output_path="dwt_subbands.png"):
        """
        Normalize and arrange LL, LH, HL, HH into a single 2x2 grid image for visualization.
        - LL (top-left): approximation (low frequencies)
        - LH (top-right): horizontal details
        - HL (bottom-left): vertical details
        - HH (bottom-right): diagonal details
        """
        # Normalize LL to [0, 255]
        ll_norm = ((LL - LL.min()) / (LL.max() - LL.min() + 1e-8) * 255).astype(np.uint8)

        # High-frequency subbands have positive and negative values centered around 0;
        # shift by +128 and boost contrast slightly so edges are clearly visible
        def norm_detail(band):
            scaled = np.clip(band * 2.0 + 128, 0, 255)
            return scaled.astype(np.uint8)

        lh_norm = norm_detail(LH)
        hl_norm = norm_detail(HL)
        hh_norm = norm_detail(HH)

        top_row = np.hstack((ll_norm, lh_norm))
        bottom_row = np.hstack((hl_norm, hh_norm))
        grid = np.vstack((top_row, bottom_row))

        Image.fromarray(grid).save(output_path)
        print(f"Saved DWT subbands (LL, LH, HL, HH) visualization to: {output_path}")


def get_ffmpeg_psnr_ssim(ref_path, dist_path):
    """Run FFmpeg's built-in psnr and ssim filters and parse the numerical results."""
    cmd = (
        f'ffmpeg -i "{dist_path}" -i "{ref_path}" '
        f'-lavfi "[0:v][1:v]psnr;[0:v][1:v]ssim" -f null -'
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    stderr = res.stderr

    # Extract average PSNR (e.g., "average:28.45") and All SSIM (e.g., "All:0.841234")
    psnr_match = re.search(r"PSNR\s+.*average:([0-9.]+)", stderr)
    ssim_match = re.search(r"SSIM\s+.*All:([0-9.]+)", stderr)

    psnr_val = float(psnr_match.group(1)) if psnr_match else 0.0
    ssim_val = float(ssim_match.group(1)) if ssim_match else 0.0
    return psnr_val, ssim_val


def run_battle(input_path, jpeg_q=31, j2k_compression_ratio=80):
    """
    Compress the same image to JPEG and JPEG2000 at a small, matched file size using FFmpeg,
    then compare PSNR, SSIM, and visual artifacts.
    """
    # Ensure reference image is lossless PNG with even dimensions
    orig_img = Image.open(input_path).convert("RGB")
    w, h = orig_img.size
    orig_img = orig_img.crop((0, 0, w - (w % 2), h - (h % 2)))
    ref_png = "battle_ref.png"
    orig_img.save(ref_png)

    jpeg_out = "battle_jpeg.jpg"
    j2k_out = "battle_jpeg2000.jp2"

    # 1. Compress with standard JPEG at strong compression (q:v 31 is lowest quality in FFmpeg MJPEG)
    cmd_jpg = f'ffmpeg -y -loglevel error -i "{ref_png}" -q:v {jpeg_q} "{jpeg_out}"'
    subprocess.run(cmd_jpg, shell=True, check=True)
    jpg_size = os.path.getsize(jpeg_out)

    # 2. Compress with JPEG2000 (libopenjpeg) matching the JPEG file size
    # In FFmpeg's libopenjpeg, -compression_level sets the target compression ratio (raw_size / target_size)
    raw_bytes = orig_img.size[0] * orig_img.size[1] * 3
    auto_ratio = int(round(raw_bytes / jpg_size))
    ratio = j2k_compression_ratio if j2k_compression_ratio else auto_ratio

    cmd_j2k = (
        f'ffmpeg -y -loglevel error -i "{ref_png}" '
        f'-c:v libopenjpeg -compression_level {ratio} "{j2k_out}"'
    )
    subprocess.run(cmd_j2k, shell=True, check=True)
    j2k_size = os.path.getsize(j2k_out)

    # 3. Compute PSNR & SSIM via FFmpeg filters
    jpg_psnr, jpg_ssim = get_ffmpeg_psnr_ssim(ref_png, jpeg_out)
    j2k_psnr, j2k_ssim = get_ffmpeg_psnr_ssim(ref_png, j2k_out)

    print("\n=== JPEG vs JPEG2000 BATTLE RESULTS ===")
    print(f"{'Codec':<12} | {'File Size (Bytes)':<18} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 60)
    print(f"{'JPEG':<12} | {jpg_size:<18} | {jpg_psnr:<12.2f} | {jpg_ssim:<10.4f}")
    print(f"{'JPEG2000':<12} | {j2k_size:<18} | {j2k_psnr:<12.2f} | {j2k_ssim:<10.4f}")

    # 4. Save side-by-side zoomed crop to easily screenshot artifacts (blocking vs ringing/blur)
    crop_box = (w // 4, h // 4, w // 4 + 300, h // 4 + 300)
    crop_orig = orig_img.crop(crop_box)
    crop_jpg = Image.open(jpeg_out).convert("RGB").crop(crop_box)
    crop_j2k = Image.open(j2k_out).convert("RGB").crop(crop_box)

    comparison = Image.new("RGB", (900, 300))
    comparison.paste(crop_orig, (0, 0))
    comparison.paste(crop_jpg, (300, 0))
    comparison.paste(crop_j2k, (600, 0))
    comparison.save("artifacts_comparison_zoom.png")
    print("\nSaved zoomed crop comparison (Original | JPEG | JPEG2000) to: artifacts_comparison_zoom.png")


if __name__ == "__main__":
    image_path = "images/amazonas.jpg"

    # Part 1: DWT Encode, Subbands Visualization & Decode
    gray_img = np.array(Image.open(image_path).convert("L"))
    dwt = DWTProcessor(wavelet="haar")

    LL, LH, HL, HH = dwt.encode(gray_img)
    dwt.save_subbands_grid(LL, LH, HL, HH, "dwt_subbands.png")

    reconstructed = dwt.decode(LL, LH, HL, HH)
    rec_error = np.max(np.abs(gray_img[:reconstructed.shape[0], :reconstructed.shape[1]].astype(int) - reconstructed.astype(int)))
    print(f"DWT Perfect Reconstruction max pixel error: {rec_error}")

    # Part 2: JPEG vs JPEG2000 FFmpeg Battle (passing None auto-matches JPEG2000 size to JPEG size)
    run_battle(image_path, jpeg_q=31, j2k_compression_ratio=24)


'''
If JPEG2000 wins… why is the whole world still using JPEG?
Despite JPEG2000's superior rate-distortion performance, progressive decoding, and absence of blocking 
artifacts, standard JPEG remains dominant due to ecosystem inertia, computational complexity, and early 
patent concerns. When JPEG2000 was standardized in 2000, JPEG was already universally supported across 
every web browser, operating system, and digital camera, creating a massive chicken-and-egg adoption barrier.
Moreover, JPEG2000's arithmetic coder (EBCOT) and full-frame DWT require significantly more CPU/memory 
resources and are much harder to parallelize in low-power hardware than JPEG's simple 8 times8 block-based
DCT. Combined with initial fears over patent licensing and the lack of native support in major web browsers 
(historically only Safari supported it), JPEG2000 remained confined to niche professional industries—such 
as digital cinema (DCP), medical imaging (DICOM), and archival preservation—while consumer 
devices stuck with JPEG until modern replacements like WebP and AVIF arrived.

'''