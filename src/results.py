import os
import matplotlib.pyplot as plt


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

methods = [
    "Nearest Neighbor",
    "Bilinear",
    "Bicubic",
    "SRCNN",
    "FSRCNN"
]

mse_values = [
    0.003415,
    0.003406,
    0.002705,
    0.002306,
    0.001658
]

psnr_values = [
    24.667,
    24.678,
    25.678,
    26.372,
    27.805
]


# ============================================================
# PSNR COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    methods,
    psnr_values
)

plt.xlabel("Method")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")

plt.xticks(
    rotation=15,
    ha="right"
)

plt.ylim(
    20,
    max(psnr_values) + 2
)

# Add values above bars
for bar, value in zip(bars, psnr_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "psnr_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# MSE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    methods,
    mse_values
)

plt.xlabel("Method")
plt.ylabel("MSE")
plt.title("MSE Comparison")

plt.xticks(
    rotation=15,
    ha="right"
)

# Add values above bars
for bar, value in zip(bars, mse_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.00003,
        f"{value:.6f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "mse_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

for method, mse, psnr in zip(
    methods,
    mse_values,
    psnr_values
):

    print(
        f"{method:<20}"
        f"MSE: {mse:.6f}   "
        f"PSNR: {psnr:.3f} dB"
    )

print("=" * 60)

print(
    f"\nBest Model: FSRCNN"
)

print(
    f"Best PSNR: {max(psnr_values):.3f} dB"
)

print(
    f"Best MSE: {min(mse_values):.6f}"
)

print(
    "\nGraphs saved to:"
)

print(
    "outputs/plots/psnr_comparison.png"
)

print(
    "outputs/plots/mse_comparison.png"
)