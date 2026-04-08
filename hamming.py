# hamming.py
# -------------------------------------------
# Implements the P → S mapping from the paper (Eq. 3 & Eq. 1)
# P = [x1[7], x1[6], x1[5], x1[4], x2[7], x2[6], x2[5]]
# S = H * P^T mod 2
# Also provides bit-flip for recovery: P' = P ⊕ e_l (Eq. 11)

from typing import List  # ✅ FIXED: Added this import

# Parity-check matrix H (3x7) used for Hamming(7,4)
# Each row corresponds to s1, s2, s3 parity equations.
H = [
    [0,0,0,1,1,1,1],  # s1 parity over positions 4,5,6,7
    [0,1,1,0,0,1,1],  # s2 parity over positions 2,3,6,7
    [1,0,1,0,1,0,1],  # s3 parity over positions 1,3,5,7
]

def p_from_pixels(x1: int, x2: int) -> List[int]:
    """
    Construct 7-bit vector P from most significant bits:
    P = [x1[7], x1[6], x1[5], x1[4],  x2[7], x2[6], x2[5]]
    """
    return [
        (x1 >> 7) & 1,
        (x1 >> 6) & 1,
        (x1 >> 5) & 1,
        (x1 >> 4) & 1,
        (x2 >> 7) & 1,
        (x2 >> 6) & 1,
        (x2 >> 5) & 1
    ]

def syndrome(P: List[int]) -> List[int]:
    """
    Compute the 3-bit Hamming syndrome S = H * P^T mod 2
    """
    S = []
    for row in H:
        v = 0
        for (bit, weight) in zip(P, row):
            if weight:
                v ^= bit
        S.append(v & 1)
    return S

def flip_bit(P: List[int], l: int) -> List[int]:
    """
    Flip bit at position l (1-based index, 1 ≤ l ≤ 7).
    Used in recovery (Eq. 11).
    """
    assert 1 <= l <= 7
    Q = P[:]  # copy
    Q[l - 1] ^= 1
    return Q