import numpy as np
import subprocess
import heapq
import scipy


# Standard JPEG luminance quantization table (QY)
QY = np.array([
            [16, 11, 10, 16, 24,  40,  51,  61],
            [12, 12, 14, 19, 26,  58,  60,  55],
            [14, 13, 16, 24, 40,  57,  69,  56],
            [14, 17, 22, 29, 51,  87,  80,  62],
            [18, 22, 37, 56, 68, 109, 103,  77],
            [24, 35, 55, 64, 81, 104, 113,  92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103,  99]
        ])


class ColorTranslator:
    def __init__(self, picture=None):
        self.picture = picture

    def rgb_to_yuv(self):
        """Convert RGB to YUV using BT.601 full-range coefficients."""
        R = self.picture[:, :, 0].astype(float)
        G = self.picture[:, :, 1].astype(float)
        B = self.picture[:, :, 2].astype(float)

        # BT.601 full-range conversion with offset
        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
        U = -0.148 * R - 0.291 * G + 0.439 * B + 128
        V = 0.439 * R - 0.368 * G - 0.071 * B + 128

        picture_yuv = np.dstack((Y, U, V))
        return np.clip(np.round(picture_yuv), 0, 255).astype(np.uint8)

    def yuv_to_rgb(self):
        """Convert YUV back to RGB (inverse of rgb_to_yuv)."""
        Y = self.picture[:, :, 0].astype(float)
        U = self.picture[:, :, 1].astype(float)
        V = self.picture[:, :, 2].astype(float)

        # Inverse BT.601 full-range conversion
        B = 1.164 * (Y - 16) + 2.018 * (U - 128)
        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
        R = 1.164 * (Y - 16) + 1.596 * (V - 128)

        picture_rgb = np.dstack((R, G, B))
        return np.clip(np.round(picture_rgb), 0, 255).astype(np.uint8)

    def resize(self, scale, img_path, exit_path):
        """Downscale image by factor using FFmpeg (bicubic by default)."""
        cmd = f'ffmpeg -y -i {img_path} -vf "scale=iw/{scale}:ih/{scale}" "{exit_path}/image_scaled_by_{scale}.jpg"'
        subprocess.run(cmd, check=True, shell=True)

    def zigzag(self, matrix):
        """Zigzag scan 8x8 matrix -> list of diagonals (low-to-high frequency order)."""
        rows, columns = matrix.shape
        solution = [[] for _ in range(rows + columns - 1)]
        for i in range(rows):
            for j in range(columns):
                s = i + j
                if s % 2 == 0:
                    solution[s].insert(0, matrix[i][j])  # upward diagonal
                else:
                    solution[s].append(matrix[i][j])     # downward diagonal
        return solution

    def inverse_zigzag(self, zigzag):
        """Reconstruct 8x8 matrix from flat zigzag-ordered list."""
        matrix = np.zeros((8, 8))
        direction = 1  # 1 = up, 0 = down
        diagonal_num = 0

        while zigzag:
            for i in range(8):
                for j in range(8):
                    if i + j == diagonal_num:
                        if direction == 0:  # downwards
                            matrix[i][j] = zigzag.pop(0)
                        else:  # upwards
                            matrix[j][i] = zigzag.pop(0)
            diagonal_num += 1
            direction = 1 - direction
        return matrix

    def RLE(self, zigzag):
        """Run-length encode a flat list: [val, count, val, count, ...]."""
        rle = []
        count = 1
        for i in range(1, len(zigzag)):
            if zigzag[i] == zigzag[i - 1]:
                count += 1
            else:
                rle.append(zigzag[i - 1])
                rle.append(count)
                count = 1
        rle.append(zigzag[-1])
        rle.append(count)
        return rle

    def RLE_decoder(self, rle):
        """Decode RLE list back to flat array."""
        decoded = []
        for i in range(1, len(rle), 2):
            value = rle[i - 1]
            count = rle[i]
            decoded.extend([value] * count)
        return np.array(decoded)

    def DCT(self, matrix):
        """Forward DCT-II on 8x8 block with JPEG level shift (-128)."""
        matrix = matrix.astype(float) - 128.0
        dct = np.zeros((8, 8))
        for u in range(8):
            a = np.sqrt(1/8) if u == 0 else np.sqrt(2/8)
            for v in range(8):
                b = np.sqrt(1/8) if v == 0 else np.sqrt(2/8)
                for x in range(8):
                    for y in range(8):
                        dct[u][v] += a * b * matrix[x][y] * \
                            np.cos((np.pi / 8) * (x + 0.5) * u) * \
                            np.cos((np.pi / 8) * (y + 0.5) * v)
        return dct

    def inv_DCT(self, matrix):
        """Inverse DCT using SciPy's orthonormal IDCT, then add back 128."""
        return scipy.fft.idctn(matrix, norm='ortho') + 128.0

    def quantization(self, dct, qual_factor):
        """Quantize DCT coefficients using scaled JPEG Q-table."""
        if qual_factor < 50:
            S = 5000 / qual_factor
        else:
            S = 200 - 2 * qual_factor

        q_scaled = np.clip(np.round(QY * (S / 100)), 1, 255)
        return np.round(dct / q_scaled)

    def dequantize(self, dct_q, qual_factor):
        """Dequantize (approximate inverse of quantization)."""
        if qual_factor < 50:
            S = 5000 / qual_factor
        else:
            S = 200 - 2 * qual_factor

        q_scaled = np.clip(np.round(QY * (S / 100)), 1, 255)
        return np.round(dct_q * q_scaled)

    def RLE_frequencies(self, rle):
        """Compute symbol frequencies from RLE output for Huffman coding."""
        counts = {}
        for i in range(1, len(rle), 2):
            val = rle[i - 1]
            cnt = rle[i]
            counts[val] = counts.get(val, 0) + cnt
        values = np.array(list(counts.keys()))
        freq = np.array(list(counts.values()))
        return values, freq
   


class Node:
    """Huffman tree node."""
    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


def build_huffman_tree(chars, freq):
    """Build Huffman tree from symbols and frequencies using min-heap."""
    pq = [Node(c, f) for c, f in zip(chars, freq)]
    heapq.heapify(pq)

    while len(pq) > 1:
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)
        merged = Node(frequency=left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(pq, merged)

    return pq[0]


def generate_huffman_codes(node, code="", huffman_codes=None):
    """Traverse Huffman tree to generate prefix-free codes for each symbol."""
    if huffman_codes is None:
        huffman_codes = {}
    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        generate_huffman_codes(node.left, code + "0", huffman_codes)
        generate_huffman_codes(node.right, code + "1", huffman_codes)
    return huffman_codes
    
