# config.py
# -------------------------------------------------
# Global configuration for Dual RDH Improvement
# -------------------------------------------------

import os

# ===============================
# BASIC EXPERIMENT SETTINGS
# ===============================

# Payload for high-quality experiment
PAYLOAD_BITS = 30000

# Random seed for reproducibility
RANDOM_SEED = 42

# Pixel range
PIXEL_MIN = 0
PIXEL_MAX = 255

# Variance threshold for smooth region embedding
VARIANCE_THRESHOLD = 50   # You can tune this later


# ===============================
# PATH SETTINGS
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "output")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SUPPORTED_FORMATS = ('.png', '.jpg', '.bmp', '.tif')


# ===============================
# IMAGE SETTINGS
# ===============================

EXPECTED_IMAGE_SIZE = (512, 512)