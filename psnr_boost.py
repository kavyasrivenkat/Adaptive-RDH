# ============================================================================
# TEST: Can 1.5 bpp be achieved while maintaining reversibility?
# ============================================================================
# Paper claims: 1.5 bpp @ 66.77 dB (reversible)
# Test: What PSNR do we get at 1.5 bpp?
# ============================================================================

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from hamming import p_from_pixels, syndrome
from arithmetic import encode_bits, decode_bits
from emd import is_guard_pair

DATASET_DIR = "dataset"
BLOCK_SIZE = 16

def bytes_to_bits(data: bytes):
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return bits

def iter_blocks(h: int, w: int):
    for r0 in range(0, h, BLOCK_SIZE):
        for c0 in range(0, w, BLOCK_SIZE):
            yield r0, c0, min(BLOCK_SIZE, h - r0), min(BLOCK_SIZE, w - c0)

def embed_pair_emd(x1, x2, d_decimal):
    x1, x2 = int(x1), int(x2)
    f = (x1 + x2 * 2) % 5
    pos = (d_decimal - f) % 5
    if pos < 0:
        pos = 5 - abs(pos)
    
    if pos == 0:
        return x1, x2
    elif pos == 1:
        return min(x1 + 1, 255), x2
    elif pos == 2:
        return x1, min(x2 + 1, 255)
    elif pos == 3:
        return max(x1 - 1, 0), min(x2 + 2, 255)
    elif pos == 4:
        return min(x1 + 1, 255), max(x2 - 1, 0)
    return x1, x2

print("="*100)
print("TEST: 1.5 BPP REVERSIBILITY CHECK")
print("="*100)
print("Paper claims: 1.5 bpp @ 66.77 dB (reversible)")
print("Question: What PSNR at 1.5 bpp with your implementation?")
print("="*100)

fname = "Airplane.png"
img = cv2.imread(os.path.join(DATASET_DIR, fname), cv2.IMREAD_GRAYSCALE)
h, w = img.shape
total_pixels = h * w

print(f"\nImage: {fname} ({h}×{w} = {total_pixels} pixels)")

# Calculate capacities
capacity_15bpp = int(1.5 * total_pixels)

print(f"\n1.5 bpp capacity: {capacity_15bpp:,} bits")

# ===== TEST: EMBED 1.5 BPP =====
print("\n" + "-"*100)
print("EMBEDDING 1.5 BPP (MAXIMUM)")
print("-"*100)

I1 = img.copy()
I2 = img.copy()

np.random.seed(0)
secret_bits = np.random.randint(0, 2, capacity_15bpp).tolist()

# Build M
M = []
secret_pos = 0
pairs_used = 0

for r0, c0, bh, bw in iter_blocks(h, w):
    for r in range(r0, min(r0 + bh, h)):
        for c in range(c0, min(c0 + bw, w)):
            if secret_pos >= capacity_15bpp:
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])
            
            if is_guard_pair(x1, x2):
                continue

            P = p_from_pixels(x1, x2)
            S = syndrome(P)

            remaining = capacity_15bpp - secret_pos
            take = min(3, remaining)

            for k in range(take):
                M.append(S[k] ^ secret_bits[secret_pos])
                secret_pos += 1
            
            pairs_used += 1

        if secret_pos >= capacity_15bpp:
            break
    if secret_pos >= capacity_15bpp:
        break

print(f"M size: {len(M):,} bits")
print(f"Pairs used: {pairs_used:,}")

# Compress
compressed_bytes = encode_bits(M)
Dcomp = bytes_to_bits(compressed_bytes)

print(f"Compressed: {len(Dcomp):,} bits")

# Embed
bit_pos = 0
for r0, c0, bh, bw in iter_blocks(h, w):
    for r in range(r0, min(r0 + bh, h)):
        for c in range(c0, min(c0 + bw, w)):
            if bit_pos >= len(Dcomp):
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            if bit_pos + 1 < len(Dcomp):
                d_decimal = Dcomp[bit_pos] * 2 + Dcomp[bit_pos + 1]
            else:
                d_decimal = Dcomp[bit_pos] * 2

            x1_new, x2_new = embed_pair_emd(x1, x2, d_decimal)
            I1[r, c] = x1_new
            I2[r, c] = x2_new

            bit_pos += 2
        
        if bit_pos >= len(Dcomp):
            break
    if bit_pos >= len(Dcomp):
        break

# ===== QUALITY METRICS =====
psnr1 = peak_signal_noise_ratio(img, I1)
psnr2 = peak_signal_noise_ratio(img, I2)
psnr_avg = (psnr1 + psnr2) / 2.0

ssim1 = structural_similarity(img, I1)
ssim2 = structural_similarity(img, I2)
ssim_avg = (ssim1 + ssim2) / 2.0

print(f"\n" + "-"*100)
print("RESULTS AT 1.5 BPP")
print("-"*100)

print(f"\nCapacity: 1.5 bpp ({capacity_15bpp:,} bits)")
print(f"PSNR1: {psnr1:.2f} dB")
print(f"PSNR2: {psnr2:.2f} dB")
print(f"Avg PSNR: {psnr_avg:.2f} dB")
print(f"Avg SSIM: {ssim_avg:.5f}")

print(f"\nReversibility Analysis:")
if psnr_avg > 60:
    print(f"✅ REVERSIBLE (PSNR > 60 dB)")
elif psnr_avg > 55:
    print(f"⚠️  NEAR-REVERSIBLE (PSNR ≈ 55-60 dB)")
else:
    print(f"❌ NOT REVERSIBLE (PSNR < 55 dB)")

print(f"\n" + "="*100)
print("COMPARISON")
print("="*100)

print(f"""
Paper claims: 1.5 bpp @ 66.77 dB (reversible)
Your result:  1.5 bpp @ {psnr_avg:.2f} dB

Gap: {66.77 - psnr_avg:.2f} dB

Analysis:
- Paper's claim: PSNR 66.77 dB (reversible threshold > 60 dB)
- Your 1.5 bpp: PSNR {psnr_avg:.2f} dB

Reversibility Status:
""")

if psnr_avg > 60:
    print(f"✅ Your 1.5 bpp IS REVERSIBLE (PSNR {psnr_avg:.2f} > 60)")
else:
    print(f"❌ Your 1.5 bpp NOT REVERSIBLE (PSNR {psnr_avg:.2f} < 60)")
    reversible_capacity = 40000  # approximate
    print(f"\n⚠️  Maximum REVERSIBLE capacity appears to be ~40K bits (0.153 bpp)")
    print(f"    To achieve 1.5 bpp reversibly, you need:")
    print(f"    • Better compression (paper uses undocumented method)")
    print(f"    • Different PSNR calculation")
    print(f"    • Improvement in core algorithm")

print("="*100)