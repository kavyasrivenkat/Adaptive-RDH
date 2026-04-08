from typing import Tuple

def is_guard_pair(x1: int, x2: int) -> bool:
    # ✅ BACK TO ORIGINAL: This is actually correct for the algorithm
    return (x1 == 0 and x2 == 0) or (x1 == 255 and x2 == 255)

def embed_pair(x1: int, x2: int, symbol_2bits: int, swap=False) -> Tuple[int, int]:
    d = symbol_2bits & 3
    if not swap:
        f = (x1 + 2 * x2) % 5
    else:
        f = (2 * x1 + x2) % 5
    pos = (d - f) % 5

    if pos == 0:
        return x1, x2

    elif pos == 1 and x1 < 255:
        return x1 + 1, x2

    elif pos == 2 and x2 < 255:
        return x1, x2 + 1

    elif pos == 3:
        if x1 > 0 and x2 < 254:
            return x1 - 1, x2 + 2
        else:
            return x1, x2

    elif pos == 4 and x1 < 255 and x2 > 0:
        return x1 + 1, x2 - 1

    return x1, x2

def extract_pair(x1: int, x2: int) -> int:
    return (x1 + 2 * x2) % 5