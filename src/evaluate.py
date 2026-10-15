import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

from dataset import SuperResolutionDataset
from model import SRCNN
from fsrcnn import FSRCNN


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="srcnn",
    choices=["srcnn", "fsrcnn"],
    help="Model to evaluate"
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "data/test"

MODEL_PATH = f"outputs/models/{args.model}_best.pth"

OUTPUT_DIR = f"outputs/predictions/{args.model}"

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
to_pil = transforms.ToPILImage()


# ============================================================
# GET TEST FILENAMES
# ============================================================

TEST_FILENAMES = sorted(
    os.listdir(
        os.path.join(TEST_DIR, "input")
    )
)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(mse):

    if mse == 0:
        return float("inf")

    return 10 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if args.model == "srcnn":

        model = SRCNN()

    else:

        model = FSRCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# EVALUATE SRCNN / FSRCNN
# ============================================================

def evaluate_model(model, dataset):

    total_squared_error = 0.0
    total_pixels = 0

    with torch.no_grad():

        for i in range(len(dataset)):

            input_tensor, target_tensor = dataset[i]

            input_batch = (
                input_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )

            # Forward pass
            output = model(
                input_batch
            )

            # Keep pixel values in [0, 1]
            output = torch.clamp(
                output,
                0,
                1
            )

            # Remove batch dimension
            output = output.squeeze(0).cpu()

            # Calculate squared error
            squared_error = (
                (output - target_tensor) ** 2
            )

            total_squared_error += (
                squared_error.sum().item()
            )

            total_pixels += (
                squared_error.numel()
            )

            # Save prediction
            output_image = to_pil(
                output
            )

            filename = TEST_FILENAMES[i]

            output_image.save(
                os.path.join(
                    OUTPUT_DIR,
                    f"{args.model}_{filename}"
                )
            )

    mse = (
        total_squared_error /
        total_pixels
    )

    psnr = calculate_psnr(mse)

    return mse, psnr


# ============================================================
# INTERPOLATION BASELINES
# ============================================================

def evaluate_interpolation(
    dataset,
    method
):

    total_squared_error = 0.0
    total_pixels = 0

    for i in range(len(dataset)):

        filename = TEST_FILENAMES[i]

        # ----------------------------------------------------
        # Load ground truth
        # ----------------------------------------------------

        target_path = os.path.join(
            TEST_DIR,
            "target",
            filename
        )

        target_image = Image.open(
            target_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Create 112x112 low-resolution image
        # ----------------------------------------------------

        lr_image = target_image.resize(
            (112, 112),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Upsample to 224x224
        # ----------------------------------------------------

        if method == "nearest":

            output_image = lr_image.resize(
                (224, 224),
                Image.Resampling.NEAREST
            )

        elif method == "bilinear":

            output_image = lr_image.resize(
                (224, 224),
                Image.Resampling.BILINEAR
            )

        elif method == "bicubic":

            output_image = lr_image.resize(
                (224, 224),
                Image.Resampling.BICUBIC
            )

        else:

            raise ValueError(
                f"Unknown interpolation method: {method}"
            )

        # ----------------------------------------------------
        # Convert images to tensors
        # ----------------------------------------------------

        output_tensor = to_tensor(
            output_image
        )

        target_tensor = to_tensor(
            target_image
        )

        # ----------------------------------------------------
        # Calculate squared error
        # ----------------------------------------------------

        squared_error = (
            (output_tensor - target_tensor) ** 2
        )

        total_squared_error += (
            squared_error.sum().item()
        )

        total_pixels += (
            squared_error.numel()
        )

        # ----------------------------------------------------
        # Save baseline output
        # ----------------------------------------------------

        output_image.save(
            os.path.join(
                OUTPUT_DIR,
                f"{method}_{filename}"
            )
        )

    mse = (
        total_squared_error /
        total_pixels
    )

    psnr = calculate_psnr(mse)

    return mse, psnr


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print(f"{args.model.upper()} EVALUATION")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SuperResolutionDataset(
        TEST_DIR
    )

    print(
        f"Test images: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluate SRCNN / FSRCNN
    # --------------------------------------------------------

    model_mse, model_psnr = evaluate_model(
        model,
        dataset
    )

    # --------------------------------------------------------
    # Evaluate interpolation baselines
    # --------------------------------------------------------

    nearest_mse, nearest_psnr = (
        evaluate_interpolation(
            dataset,
            "nearest"
        )
    )

    bilinear_mse, bilinear_psnr = (
        evaluate_interpolation(
            dataset,
            "bilinear"
        )
    )

    bicubic_mse, bicubic_psnr = (
        evaluate_interpolation(
            dataset,
            "bicubic"
        )
    )

    # --------------------------------------------------------
    # Final Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"{'Method':<20}"
        f"{'MSE':<15}"
        f"{'PSNR (dB)':<15}"
    )

    print("-" * 60)

    print(
        f"{'Nearest Neighbor':<20}"
        f"{nearest_mse:<15.6f}"
        f"{nearest_psnr:<15.3f}"
    )

    print(
        f"{'Bilinear':<20}"
        f"{bilinear_mse:<15.6f}"
        f"{bilinear_psnr:<15.3f}"
    )

    print(
        f"{'Bicubic':<20}"
        f"{bicubic_mse:<15.6f}"
        f"{bicubic_psnr:<15.3f}"
    )

    print(
        f"{args.model.upper():<20}"
        f"{model_mse:<15.6f}"
        f"{model_psnr:<15.3f}"
    )

    print("=" * 60)

    print(
        f"\nPredictions saved to: "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()