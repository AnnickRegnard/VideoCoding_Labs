# Zigzag reference: https://www.geeksforgeeks.org/python/python-program-for-print-matrix-in-zag-zag-fashion/

from first_seminar import ColorTranslator
import numpy as np

matrix1 = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61, 62, 63, 64]
])

zigzag = ColorTranslator()
zigzagged_matrix = zigzag.zigzag(matrix1)

zigzagged = [1, 2, 9, 17, 10, 3, 4, 11, 18, 25, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 49, 42, 35, 28, 21, 14, 7, 8, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 16, 24, 31, 38, 45, 52, 59, 60, 53, 46, 39, 32, 40, 47, 54, 61, 62, 55, 48, 56, 63, 64]

matrix2 = ColorTranslator()
matrix_recovered = matrix2.inverse_zigzag(zigzagged)

print("---- Zigzagged matrix result ----\n")
for i in zigzagged_matrix:
    for j in i:
        print(j, end=" ")
print("\n")
print("----------------------------------")

print("\n---- Recovered matrix ----\n")
print(matrix_recovered)
print("\n----------------------------------")


# Zigzag orders DCT coefficients low-to-high frequency so RLE
# can efficiently compress trailing zeros (high-frequency coeffs)