# extract.py
from typing import List, Tuple
import numpy as np
from emd import extract_pair, is_guard_pair

BLOCK = 16
SIDE_PIXELS = 8

def iter_blocks(h: int, w: int):
    for r0 in range(0, h, BLOCK):
        for c0 in range(0, w, BLOCK):
            yield r0, c0, min(BLOCK, h - r0), min(BLOCK, w - c0)


def extract_bits_from_block(
    I1: np.ndarray,
    I2: np.ndarray,
    r0: int,
    c0: int,
    n_payload_bits: int,
    payload_bits: List[int],
    n_side_bits: int,
    side_bits: List[int]
) -> Tuple[int, int]:
    """
    Extract payload bits (forward) and side bits (backward)
    """

    h, w = I1.shape
    r_end = min(r0 + BLOCK, h)
    c_end = min(c0 + BLOCK, w)

    total_pixels = (r_end - r0) * (c_end - c0)
    max_payload_bits = (total_pixels - SIDE_PIXELS) * 2

    got_payload = 0
    got_side = 0

    # --------------------------------------------------
    # 1. Payload extraction (forward scan)
    # --------------------------------------------------
    for r in range(r0, r_end):
        for c in range(c0, c_end):
            if got_payload >= n_payload_bits:
                break
            if got_payload >= max_payload_bits:
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            sym = extract_pair(x1, x2)
            payload_bits.append((sym >> 1) & 1)
            payload_bits.append(sym & 1)

            got_payload += 2

        if got_payload >= n_payload_bits or got_payload >= max_payload_bits:
            break

    # --------------------------------------------------
    # 2. Side information extraction (backward scan)
    # --------------------------------------------------
    side_pixels_used = 0

    for r in range(r_end - 1, r0 - 1, -1):
        for c in range(c_end - 1, c0 - 1, -1):
            if got_side >= n_side_bits:
                break
            if side_pixels_used >= SIDE_PIXELS:
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            sym = extract_pair(x1, x2)
            side_bits.append((sym >> 1) & 1)
            side_bits.append(sym & 1)

            got_side += 2
            side_pixels_used += 1

        if got_side >= n_side_bits or side_pixels_used >= SIDE_PIXELS:
            break

    return got_payload, got_side


def restore_pairs_after_payload(
    I1: np.ndarray,
    I2: np.ndarray,
    r0: int,
    c0: int,
    bh: int,
    bw: int,
    payload_bits_in_block: int
):
    """
    ✅ FIXED: Restore original pixels correctly for reversibility
    Uses: p = (x1 + x2) // 2 (better than x1*x2//2)
    """

    pairs_to_restore = payload_bits_in_block // 2
    if pairs_to_restore <= 0:
        return

    restored = 0

    for r in range(r0, r0 + bh):
        for c in range(c0, c0 + bw):
            if restored >= pairs_to_restore:
                return

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            # ✅ BETTER restoration formula
            p = (x1 + x2) // 2
            p = max(0, min(255, p))

            I1[r, c] = p
            I2[r, c] = p

            restored += 1