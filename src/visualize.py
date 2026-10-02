import os

import matplotlib.pyplot as plt
from PIL import Image


TEST_IMAGE = "0081.png"

INPUT_PATH = f"data/test/input/{TEST_IMAGE}"
TARGET_PATH = f"data/test/target/{TEST_IMAGE}"

PREDICTION_DIR = "outputs/predictions"
OUTPUT_DIR = "outputs/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(path):
    return Image.open(path).convert("RGB")


def main():

    input_image = load_image(INPUT_PATH)
    target_image = load_image(TARGET_PATH)

    nearest_image = load_image(
        os.path.join(
            PREDICTION_DIR,
            f"nearest_{TEST_IMAGE}"
        )
    )

    bilinear_image = load_image(
        os.path.join(
            PREDICTION_DIR,
            f"bilinear_{TEST_IMAGE}"
        )
    )

    bicubic_image = load_image(
        os.path.join(
            PREDICTION_DIR,
            f"bicubic_{TEST_IMAGE}"
        )
    )

    srcnn_image = load_image(
        os.path.join(
            PREDICTION_DIR,
            f"srcnn_{TEST_IMAGE}"
        )
    )

    # Create figure
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    images = [
        input_image,
        nearest_image,
        bilinear_image,
        bicubic_image,
        srcnn_image,
        target_image
    ]

    titles = [
        "SRCNN Input",
        "Nearest Neighbor",
        "Bilinear",
        "Bicubic",
        "SRCNN",
        "Ground Truth"
    ]

    for ax, image, title in zip(
        axes.flat,
        images,
        titles
    ):

        ax.imshow(image)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "comparison_0081.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Comparison saved to: {output_path}"
    )


if __name__ == "__main__":
    main()