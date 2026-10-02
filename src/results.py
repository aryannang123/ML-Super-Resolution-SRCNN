import os
import matplotlib.pyplot as plt


# =========================
# Final test results
# =========================

methods = [
    "Nearest Neighbor",
    "Bilinear",
    "Bicubic",
    "SRCNN"
]

psnr_values = [
    24.667,
    24.678,
    25.678,
    26.372
]

mse_values = [
    0.003415,
    0.003406,
    0.002705,
    0.002306
]


OUTPUT_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# PSNR comparison
# =========================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    methods,
    psnr_values
)

plt.ylabel("PSNR (dB)")
plt.xlabel("Method")
plt.title("Test PSNR Comparison")

plt.ylim(
    min(psnr_values) - 1,
    max(psnr_values) + 1
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Add values above bars
for bar, value in zip(bars, psnr_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

psnr_path = os.path.join(
    OUTPUT_DIR,
    "psnr_comparison.png"
)

plt.savefig(
    psnr_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================
# MSE comparison
# =========================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    methods,
    mse_values
)

plt.ylabel("MSE")
plt.xlabel("Method")
plt.title("Test MSE Comparison")

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(bars, mse_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.6f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

mse_path = os.path.join(
    OUTPUT_DIR,
    "mse_comparison.png"
)

plt.savefig(
    mse_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("Results graphs generated successfully.")

print(
    f"PSNR graph: {psnr_path}"
)

print(
    f"MSE graph: {mse_path}"
)