# arithmetic.py
# ----------------------------------------------------
# Adaptive Binary Arithmetic Coding
# Used to compress/decompress the M bitstream.
# Fully deterministic and stable under Python 3.12.

from typing import List
import struct

class AdaptiveBinaryAC:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.out = bytearray()
        self._zero = 1    # initial counts
        self._one  = 1

    def _emit(self, b: int):
        self.out.append(b & 0xFF)

    def _scale_counts(self):
        # Prevent counts from growing unbounded
        if self._zero + self._one >= (1 << 20):
            self._zero = (self._zero + 1) // 2
            self._one  = (self._one  + 1) // 2

    def _split(self):
        total = self._zero + self._one
        rng   = self.high - self.low + 1
        return self.low + (rng * self._zero) // total - 1

    def encode_bit(self, bit: int):
        split = self._split()
        if bit == 0:
            self.high = split
            self._zero += 1
        else:
            self.low = split + 1
            self._one += 1

        while True:
            # Case 1: same top byte -> output a byte
            if (self.high ^ self.low) & 0xFF000000 == 0:
                self._emit((self.high >> 24) & 0xFF)
                self.low  = (self.low  << 8) & 0xFFFFFFFF
                self.high = ((self.high << 8) | 0xFF) & 0xFFFFFFFF

            # Case 2: low in upper half and high in lower half -> "underflow"
            elif (self.low & 0x80000000) and not (self.high & 0x80000000):
                self.low  = (self.low  << 1) & 0xFFFFFFFF
                self.high = ((self.high << 1) | 1) & 0xFFFFFFFF

            else:
                break

        self._scale_counts()

    def finish(self) -> bytes:
        # Flush remaining state
        for _ in range(4):
            self._emit((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF

        # Store starting model counts for decoder consistency
        header = struct.pack(">II", 1, 1)
        return header + bytes(self.out)


class AdaptiveBinaryAD:
    def __init__(self, data: bytes):
        if len(data) < 8:
            raise ValueError("Invalid compressed stream")

        # restore initial model
        self._zero, self._one = struct.unpack(">II", data[:8])
        self.buf = data[8:]

        self.low  = 0
        self.high = 0xFFFFFFFF
        self.pos  = 0
        self.code = 0

        # initialize code with first 4 bytes
        for _ in range(4):
            self.code = ((self.code << 8) | self._read_byte()) & 0xFFFFFFFF

    def _read_byte(self) -> int:
        if self.pos >= len(self.buf):
            return 0
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def _scale_counts(self):
        if self._zero + self._one >= (1 << 20):
            self._zero = (self._zero + 1) // 2
            self._one  = (self._one  + 1) // 2

    def _split(self):
        total = self._zero + self._one
        rng   = (self.high - self.low + 1) & 0xFFFFFFFF
        return self.low + (rng * self._zero) // total - 1

    def decode_bit(self) -> int:
        split = self._split()

        if self.code <= split:
            self.high = split
            self._zero += 1
            bit = 0
        else:
            self.low = (split + 1) & 0xFFFFFFFF
            self._one += 1
            bit = 1

        while True:
            if (self.high ^ self.low) & 0xFF000000 == 0:
                self.high = ((self.high << 8) | 0xFF) & 0xFFFFFFFF
                self.low  = (self.low  << 8) & 0xFFFFFFFF
                self.code = ((self.code << 8) | self._read_byte()) & 0xFFFFFFFF

            elif (self.low & 0x80000000) and not (self.high & 0x80000000):
                self.low  = (self.low  << 1) & 0xFFFFFFFF
                self.high = ((self.high << 1) | 1) & 0xFFFFFFFF
                self.code = ((self.code << 1) | ((self._read_byte() >> 7) & 1)) & 0xFFFFFFFF

            else:
                break

        self._scale_counts()
        return bit


def encode_bits(bits: List[int]) -> bytes:
    enc = AdaptiveBinaryAC()
    for b in bits:
        enc.encode_bit(int(b))
    return enc.finish()


def decode_bits(data: bytes, nbits: int) -> List[int]:
    dec = AdaptiveBinaryAD(data)
    out = []
    for _ in range(nbits):
        out.append(dec.decode_bit())
    return out