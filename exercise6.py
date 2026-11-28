import numpy as np
from first_seminar import ColorTranslator

# Standard 8x8 luminance test block (pixel values 0-255)
block_8x8 = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
], dtype=float)

translator = ColorTranslator()

# 1. Apply forward DCT-II
dct_block = translator.DCT(block_8x8)
print("DCT original (rounded for display):")
print(np.round(dct_block, 1))

# 2. Compare different Quality Factors (high, medium, low)
for qf in [90, 50, 10]:
    print(f"\n{'='*50}")
    print(f"QUALITY FACTOR = {qf}")
    print(f"{'='*50}")

    # Quantize and dequantize
    dct_q = translator.quantization(dct_block, qf)
    dct_deq = translator.dequantize(dct_q, qf)

    # Reconstruct block with Inverse DCT
    reconstructed = translator.inv_DCT(dct_deq)

    # Count zeroed coefficients (especially high frequencies)
    zeros_count = np.sum(dct_q == 0)
    mse = np.mean((block_8x8 - reconstructed) ** 2)

    print("Quantized matrix (DCT_q):")
    print(dct_q.astype(int))
    print(f"\nZeroed coefficients: {zeros_count} / 64")
    print(f"MSE after IDCT: {mse:.2f}")
    print("\nReconstructed block:")
    print(reconstructed.astype(int))