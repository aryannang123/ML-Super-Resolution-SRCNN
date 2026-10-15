import os

import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from dataset import SuperResolutionDataset
from model import SRCNN
from fsrcnn import FSRCNN


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "data/test"

SRCNN_MODEL_PATH = "outputs/models/srcnn_best.pth"
FSRCNN_MODEL_PATH = "outputs/models/fsrcnn_best.pth"

OUTPUT_DIR = "outputs/plots"

IMAGE_NAME = "0081.png"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRANSFORMS
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# LOAD SRCNN
# ============================================================

def load_srcnn():

    model = SRCNN()

    model.load_state_dict(
        torch.load(
            SRCNN_MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD FSRCNN
# ============================================================

def load_fsrcnn():

    model = FSRCNN()

    model.load_state_dict(
        torch.load(
            FSRCNN_MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# CREATE INTERPOLATION OUTPUT
# ============================================================

def interpolate_image(
    target_image,
    method
):

    # Create 112x112 LR image
    lr_image = target_image.resize(
        (112, 112),
        Image.Resampling.BILINEAR
    )

    # Upsample
    if method == "nearest":

        output = lr_image.resize(
            (224, 224),
            Image.Resampling.NEAREST
        )

    elif method == "bilinear":

        output = lr_image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

    elif method == "bicubic":

        output = lr_image.resize(
            (224, 224),
            Image.Resampling.BICUBIC
        )

    else:

        raise ValueError(
            f"Unknown interpolation method: {method}"
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERATING VISUAL COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    input_path = os.path.join(
        TEST_DIR,
        "input",
        IMAGE_NAME
    )

    target_path = os.path.join(
        TEST_DIR,
        "target",
        IMAGE_NAME
    )

    input_image = Image.open(
        input_path
    ).convert("RGB")

    target_image = Image.open(
        target_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    srcnn = load_srcnn()

    fsrcnn = load_fsrcnn()


    # --------------------------------------------------------
    # Prepare input tensor
    # --------------------------------------------------------

    input_tensor = to_tensor(
        input_image
    ).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # SRCNN prediction
    # --------------------------------------------------------

    with torch.no_grad():

        srcnn_output = srcnn(
            input_tensor
        )

        srcnn_output = torch.clamp(
            srcnn_output,
            0,
            1
        )

    srcnn_output = (
        srcnn_output
        .squeeze(0)
        .cpu()
    )


    # --------------------------------------------------------
    # FSRCNN prediction
    # --------------------------------------------------------

    with torch.no_grad():

        fsrcnn_output = fsrcnn(
            input_tensor
        )

        fsrcnn_output = torch.clamp(
            fsrcnn_output,
            0,
            1
        )

    fsrcnn_output = (
        fsrcnn_output
        .squeeze(0)
        .cpu()
    )


    # --------------------------------------------------------
    # Interpolation baselines
    # --------------------------------------------------------

    nearest_image = interpolate_image(
        target_image,
        "nearest"
    )

    bilinear_image = interpolate_image(
        target_image,
        "bilinear"
    )

    bicubic_image = interpolate_image(
        target_image,
        "bicubic"
    )


    # --------------------------------------------------------
    # Plot comparison
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8)
    )


    # Input
    axes[0, 0].imshow(
        input_image
    )

    axes[0, 0].set_title(
        "SRCNN Input"
    )

    axes[0, 0].axis("off")


    # Nearest
    axes[0, 1].imshow(
        nearest_image
    )

    axes[0, 1].set_title(
        "Nearest Neighbor"
    )

    axes[0, 1].axis("off")


    # Bilinear
    axes[0, 2].imshow(
        bilinear_image
    )

    axes[0, 2].set_title(
        "Bilinear"
    )

    axes[0, 2].axis("off")


    # Bicubic
    axes[0, 3].imshow(
        bicubic_image
    )

    axes[0, 3].set_title(
        "Bicubic"
    )

    axes[0, 3].axis("off")


    # SRCNN
    axes[1, 0].imshow(
        srcnn_output.permute(1, 2, 0)
    )

    axes[1, 0].set_title(
        "SRCNN"
    )

    axes[1, 0].axis("off")


    # FSRCNN
    axes[1, 1].imshow(
        fsrcnn_output.permute(1, 2, 0)
    )

    axes[1, 1].set_title(
        "FSRCNN"
    )

    axes[1, 1].axis("off")


    # Ground Truth
    axes[1, 2].imshow(
        target_image
    )

    axes[1, 2].set_title(
        "Ground Truth"
    )

    axes[1, 2].axis("off")


    # Empty final panel
    axes[1, 3].axis("off")


    plt.suptitle(
        f"Super-Resolution Comparison - {IMAGE_NAME}",
        fontsize=16
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{IMAGE_NAME.replace('.png', '')}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"\nComparison saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()