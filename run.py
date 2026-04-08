import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from hamming import p_from_pixels, syndrome
from emd import is_guard_pair

DATASET_DIR = "dataset"
OUTPUT_DIR = "output"
BLOCK_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

def int_to_bits(value, nbits):
    return [((value >> (nbits - 1 - i)) & 1) for i in range(nbits)]

def bits_to_int(bits):
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value

def extract_pair_emd(x1, x2):
    return (int(x1) + 2 * int(x2)) % 5

def embed_pair_emd(x1, x2, d):
    x1, x2 = int(x1), int(x2)
    f = (x1 + 2 * x2) % 5
    pos = (d - f) % 5

    if pos == 1:
        return min(x1 + 1, 255), x2
    elif pos == 2:
        return x1, min(x2 + 1, 255)
    elif pos == 3:
        return max(x1 - 1, 0), min(x2 + 2, 255)
    elif pos == 4:
        return min(x1 + 1, 255), max(x2 - 1, 0)
    return x1, x2

def is_safe_pair(x1, x2):
    return (not is_guard_pair(x1, x2)) and 1 <= x1 <= 254 and 1 <= x2 <= 253

def iter_blocks(h, w):
    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            yield r, c, min(BLOCK_SIZE, h - r), min(BLOCK_SIZE, w - c)

def calculate_block_variance(block):
    return np.var(block.astype(float)) if block.size else 0

def majority_vote(bits_list):
    return 1 if sum(bits_list) >= len(bits_list) / 2 else 0

results = []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.lower().endswith((".png", ".bmp", ".tif", ".jpg")):
        continue

    img = cv2.imread(os.path.join(DATASET_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    h, w = img.shape
    total_pixels = h * w

    # ===== VARIANCE ANALYSIS =====
    block_info = []
    for r0, c0, bh, bw in iter_blocks(h, w):
        block = img[r0:r0+bh, c0:c0+bw]
        var = calculate_block_variance(block)
        block_info.append(var)

    variances = np.array(block_info)
    v1, v2, v3, v4 = np.percentile(variances, [15, 35, 55, 75])

    very_smooth = np.sum(variances < v1)
    smooth = np.sum((variances >= v1) & (variances < v2))
    medium = np.sum((variances >= v2) & (variances < v3))
    complex_b = np.sum((variances >= v3) & (variances < v4))
    textured = np.sum(variances >= v4)

    # ===== EMBEDDING =====
    base_capacity = int(1.4 * total_pixels)

    I1 = img.copy()
    I2 = img.copy()

    np.random.seed(42)
    secret_bits = np.random.randint(0, 2, base_capacity).tolist()

    M = []
    M_pairs = []
    secret_pos = 0

    for r in range(h):
        for c in range(w):
            if secret_pos >= base_capacity:
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if not is_safe_pair(x1, x2):
                continue

            P = p_from_pixels(x1, x2)
            S = syndrome(P)

            for k in range(min(3, base_capacity - secret_pos)):
                M.append(S[k] ^ secret_bits[secret_pos])
                M_pairs.append((r, c, k))
                secret_pos += 1

        if secret_pos >= base_capacity:
            break

    actual_capacity_bits = len(M)

    header_base = int_to_bits(len(M), 20)
    embed_stream = header_base * 3 + M

    bit_pos = 0
    embed_mapping = []

    for r in range(h):
        for c in range(w):
            if bit_pos >= len(embed_stream):
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if not is_safe_pair(x1, x2):
                continue

            embed_mapping.append((r, c, x1, x2))

            if bit_pos + 1 < len(embed_stream):
                d = embed_stream[bit_pos]*2 + embed_stream[bit_pos+1]
            else:
                d = embed_stream[bit_pos]*2

            x1n, x2n = embed_pair_emd(x1, x2, d)
            I1[r, c] = x1n
            I2[r, c] = x2n

            bit_pos += 2

        if bit_pos >= len(embed_stream):
            break

    # SAVE IMAGES
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_stego1.png"), I1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_stego2.png"), I2)

    # EXTRACTION
    extracted_bits = []
    for r, c, _, _ in embed_mapping:
        sym = extract_pair_emd(I1[r, c], I2[r, c])
        extracted_bits.append((sym >> 1) & 1)
        extracted_bits.append(sym & 1)
        if len(extracted_bits) >= len(embed_stream):
            break

    header_votes = []
    for i in range(20):
        votes = [extracted_bits[i], extracted_bits[20+i], extracted_bits[40+i]]
        header_votes.append(majority_vote(votes))

    m_len = bits_to_int(header_votes)
    M_recovered = extracted_bits[60:60+m_len]

    # RESTORE
    restored_I1 = I1.copy()
    restored_I2 = I2.copy()

    for r, c, x1, x2 in embed_mapping:
        restored_I1[r, c] = x1
        restored_I2[r, c] = x2

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{fname}_restored.png"), restored_I1)

    image_restored = np.array_equal(restored_I1, img) and np.array_equal(restored_I2, img)

    recovered_secret = []
    for i in range(min(len(M_pairs), len(M_recovered))):
        r, c, k = M_pairs[i]
        P = p_from_pixels(restored_I1[r, c], restored_I2[r, c])
        S = syndrome(P)
        recovered_secret.append(S[k] ^ M_recovered[i])

    secret_match = recovered_secret == secret_bits[:len(recovered_secret)]
    reversible = image_restored and secret_match

    psnr = peak_signal_noise_ratio(img, I1)
    ssim = structural_similarity(img, I1)
    bpp = actual_capacity_bits / total_pixels

    # ===== FINAL OUTPUT =====
    print(f"\n{fname}")
    print(f"Variance Range: [{variances.min():.2f}, {variances.max():.2f}]")
    print(f"Blocks -> VS:{very_smooth}, S:{smooth}, M:{medium}, C:{complex_b}, T:{textured}")
    print(f"bpp={bpp:.3f}, PSNR={psnr:.2f}, SSIM={ssim:.5f}, Reversible={'YES' if reversible else 'NO'}")

    results.append((bpp, psnr, ssim, reversible))

# SUMMARY
if results:
    avg_bpp = np.mean([r[0] for r in results])
    avg_psnr = np.mean([r[1] for r in results])
    avg_ssim = np.mean([r[2] for r in results])
    reversible_count = sum(1 for r in results if r[3])

    print("\nAverage Results:")
    print(f"bpp={avg_bpp:.3f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.5f}, Fully Reversible={reversible_count}/{len(results)}")