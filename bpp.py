import os
import cv2
import numpy as np

# Folder containing images
DATASET_DIR = "dataset"

def calculate_bpp(embedded_bits, image):
    h, w = image.shape
    total_pixels = h * w
    return embedded_bits / total_pixels

print("=" * 60)
print("📊 EMBEDDING RATE (bpp) CALCULATION")
print("=" * 60)

results = []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
        continue

    path = os.path.join(DATASET_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    h, w = img.shape
    total_pixels = h * w

    # 🔹 CASE 1: Paper scenario (fixed payload)
    paper_bits = 30000
    paper_bpp = calculate_bpp(paper_bits, img)

    # 🔹 CASE 2: Your method (use your actual embedded bits)
    # 👉 Replace this with your actual value from code output
    your_embedded_bits = int(1.366 * total_pixels)   # example from your results
    your_bpp = calculate_bpp(your_embedded_bits, img)

    print(f"\nImage: {fname}")
    print(f"  Total Pixels: {total_pixels}")
    print(f"  Paper (30,000 bits): {paper_bpp:.4f} bpp")
    print(f"  Your Method:         {your_bpp:.4f} bpp")

    results.append((paper_bpp, your_bpp))

# 📊 Average comparison
if results:
    avg_paper = np.mean([r[0] for r in results])
    avg_your = np.mean([r[1] for r in results])

    print("\n" + "=" * 60)
    print("📈 AVERAGE RESULTS")
    print("=" * 60)
    print(f"Paper Avg bpp: {avg_paper:.4f}")
    print(f"Your Avg bpp:  {avg_your:.4f}")

    improvement = ((avg_your - avg_paper) / avg_paper) * 100
    print(f"Improvement:   {improvement:.2f}%")