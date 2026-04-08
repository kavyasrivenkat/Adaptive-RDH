import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# -----------------------------
# Load grayscale image
# -----------------------------
def load_grayscale_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at path: {path}")
    return img


# -----------------------------
# Save image (optional helper)
# -----------------------------
def save_image(path, image):
    cv2.imwrite(path, image)


# -----------------------------
# PSNR calculation
# -----------------------------
def psnr(original, stego):
    original = original.astype(np.float32)
    stego = stego.astype(np.float32)

    mse = np.mean((original - stego) ** 2)

    if mse == 0:
        return 100  # Perfect match

    return 10 * np.log10((255 ** 2) / mse)


# -----------------------------
# SSIM calculation
# -----------------------------
def compute_ssim(original, stego):
    return ssim(original, stego, data_range=255)


# -----------------------------
# Generate random bit string
# -----------------------------
def generate_random_bits(length):
    return "".join(np.random.choice(['0', '1'], length))