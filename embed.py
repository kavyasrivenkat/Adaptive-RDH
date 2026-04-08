# embed.py
from typing import List, Tuple
import numpy as np
from emd import embed_pair, is_guard_pair

BLOCK = 16

def iter_blocks(h: int, w: int):
    for r0 in range(0, h, BLOCK):
        for c0 in range(0, w, BLOCK):
            yield r0, c0, min(BLOCK, h - r0), min(BLOCK, w - c0)

def embed_bits_into_block(
    I1: np.ndarray,
    I2: np.ndarray,
    r0: int,
    c0: int,
    bits: List[int],
    bit_pos: int,
    side_bits: List[int],
    side_pos: int,
    swap: bool = False
) -> Tuple[int, int]:

    h, w = I1.shape
    r_end = min(r0 + BLOCK, h)
    c_end = min(c0 + BLOCK, w)

    # --------------------------------------------------
    # 1. Payload embedding (forward scan)
    # --------------------------------------------------
    for r in range(r0, r_end):
        for c in range(c0, c_end):
            if bit_pos >= len(bits):
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            b0 = bits[bit_pos]
            b1 = bits[bit_pos + 1] if bit_pos + 1 < len(bits) else 0
            sym = (b0 << 1) | b1

            nx1, nx2 = embed_pair(x1, x2, sym, swap)

            I1[r, c] = nx1
            I2[r, c] = nx2

            bit_pos += 2

        if bit_pos >= len(bits):
            break

    # --------------------------------------------------
    # 2. Side information embedding (backward scan)
    # --------------------------------------------------
    for r in range(r_end - 1, r0 - 1, -1):
        for c in range(c_end - 1, c0 - 1, -1):
            if side_pos >= len(side_bits):
                break

            x1 = int(I1[r, c])
            x2 = int(I2[r, c])

            if is_guard_pair(x1, x2):
                continue

            b0 = side_bits[side_pos]
            b1 = side_bits[side_pos + 1] if side_pos + 1 < len(side_bits) else 0
            sym = (b0 << 1) | b1

            nx1, nx2 = embed_pair(x1, x2, sym, swap)

            I1[r, c] = nx1
            I2[r, c] = nx2

            side_pos += 2

        if side_pos >= len(side_bits):
            break

    return bit_pos, side_pos