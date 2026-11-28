# References:
# https://www.geeksforgeeks.org/python/run-length-encoding-python/
# https://www.geeksforgeeks.org/dsa/huffman-coding-in-python/

from first_seminar import ColorTranslator, build_huffman_tree, generate_huffman_codes

zigzagged = [1, 1, 1, 6, 6, 4, 7, 7, 7, 7, 8, 8, 5, 3, 3, 3, 3, 3, 3]

rle_encoder = ColorTranslator()
rle = rle_encoder.RLE(zigzagged)

print("RLE output: ", rle)

rle_decoder = ColorTranslator()
zigzagged_recovered = rle_encoder.RLE_decoder(rle)

print("RLE decoder output: ", zigzagged_recovered)

# 2. Compute frequencies from RLE
values, freq = rle_encoder.RLE_frequencies(rle)
print("\nUnique values:", values)
print("Frequencies:", freq)

# 3. Build Huffman tree and codes
root = build_huffman_tree(values, freq)
huffman_codes = generate_huffman_codes(root)

print("\nGenerated Huffman codes:")
for val, code in huffman_codes.items():
    print(f"  Value {val} -> {code}")

# 4. Encode original sequence using Huffman codes
encoded_bitstream = "".join(huffman_codes[val] for val in zigzagged)
print("\nEncoded bitstream:", encoded_bitstream)

