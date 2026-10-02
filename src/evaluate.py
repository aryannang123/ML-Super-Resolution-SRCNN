import os

import torch
from PIL import Image
from torchvision import transforms

from dataset import SuperResolutionDataset
from model import SRCNN


# =========================
# Configuration
# =========================

TEST_DIR = "data/test"
MODEL_PATH = "outputs/models/srcnn_best.pth"

OUTPUT_DIR = "outputs/predictions"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# =========================
# PSNR
# =========================

def calculate_psnr(mse):
    """
    Calculate PSNR assuming pixel values
    are in the range [0, 1].
    """

    if mse <= 0:
        return float("inf")

    return 10 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


# =========================
# Load trained SRCNN
# =========================

def load_model():

    model = SRCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# =========================
# Evaluate SRCNN
# =========================

def evaluate_srcnn(model, dataset):

    total_squared_error = 0.0
    total_pixels = 0

    print("\nEvaluating SRCNN...")

    with torch.no_grad():

        for index in range(len(dataset)):

            input_tensor, target_tensor = dataset[index]

            # Add batch dimension
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            # SRCNN prediction
            output = model(input_batch)

            # Keep image values valid
            output = torch.clamp(
                output,
                0.0,
                1.0
            )

            target = target_tensor.unsqueeze(0).to(DEVICE)

            # Calculate squared error
            squared_error = torch.sum(
                (output - target) ** 2
            ).item()

            total_squared_error += squared_error
            total_pixels += target.numel()

            # Save prediction
            output_image = output.squeeze(0).cpu()

            filename = dataset.image_names[index]

            output_path = os.path.join(
                OUTPUT_DIR,
                f"srcnn_{filename}"
            )

            to_pil(output_image).save(output_path)

    mse = total_squared_error / total_pixels

    psnr = calculate_psnr(mse)

    return mse, psnr


# =========================
# Evaluate interpolation
# =========================

def evaluate_interpolation(dataset, method):

    total_squared_error = 0.0
    total_pixels = 0

    print(f"\nEvaluating {method}...")

    for index in range(len(dataset)):

        # We use the ORIGINAL target image
        # to create the common 112x112 LR image.
        _, target_tensor = dataset[index]

        target_image = to_pil(target_tensor)

        # ---------------------------------
        # Create the common low-resolution
        # image used by every baseline
        # ---------------------------------

        low_res = target_image.resize(
            (112, 112),
            resample=Image.Resampling.BILINEAR
        )

        # ---------------------------------
        # Upsample using selected method
        # ---------------------------------

        if method == "nearest":

            reconstructed = low_res.resize(
                (224, 224),
                resample=Image.Resampling.NEAREST
            )

        elif method == "bilinear":

            reconstructed = low_res.resize(
                (224, 224),
                resample=Image.Resampling.BILINEAR
            )

        elif method == "bicubic":

            reconstructed = low_res.resize(
                (224, 224),
                resample=Image.Resampling.BICUBIC
            )

        else:

            raise ValueError(
                f"Unknown interpolation method: {method}"
            )

        # Convert reconstructed image to tensor
        reconstructed_tensor = to_tensor(
            reconstructed
        ).unsqueeze(0)

        target = target_tensor.unsqueeze(0)

        # Calculate squared error
        squared_error = torch.sum(
            (reconstructed_tensor - target) ** 2
        ).item()

        total_squared_error += squared_error
        total_pixels += target.numel()

        # Save baseline prediction
        filename = dataset.image_names[index]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{method}_{filename}"
        )

        reconstructed.save(output_path)

    mse = total_squared_error / total_pixels

    psnr = calculate_psnr(mse)

    return mse, psnr


# =========================
# Main
# =========================

def main():

    print("=" * 55)
    print("SRCNN TEST EVALUATION")
    print("=" * 55)

    print("Device:", DEVICE)

    # =========================
    # Load test dataset
    # =========================

    test_dataset = SuperResolutionDataset(TEST_DIR)

    print(
        "Test samples:",
        len(test_dataset)
    )

    # =========================
    # Load trained SRCNN
    # =========================

    model = load_model()

    print(
        "\nLoaded model:",
        MODEL_PATH
    )

    # =========================
    # SRCNN
    # =========================

    srcnn_mse, srcnn_psnr = evaluate_srcnn(
        model,
        test_dataset
    )

    # =========================
    # Baselines
    # =========================

    nearest_mse, nearest_psnr = evaluate_interpolation(
        test_dataset,
        "nearest"
    )

    bilinear_mse, bilinear_psnr = evaluate_interpolation(
        test_dataset,
        "bilinear"
    )

    bicubic_mse, bicubic_psnr = evaluate_interpolation(
        test_dataset,
        "bicubic"
    )

    # =========================
    # Final results
    # =========================

    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS")
    print("=" * 55)

    print(
        f"Nearest Neighbor : "
        f"{nearest_psnr:.3f} dB"
    )

    print(
        f"Bilinear         : "
        f"{bilinear_psnr:.3f} dB"
    )

    print(
        f"Bicubic          : "
        f"{bicubic_psnr:.3f} dB"
    )

    print(
        f"SRCNN            : "
        f"{srcnn_psnr:.3f} dB"
    )

    print("\nMSE values:")

    print(
        f"Nearest Neighbor : "
        f"{nearest_mse:.6f}"
    )

    print(
        f"Bilinear         : "
        f"{bilinear_mse:.6f}"
    )

    print(
        f"Bicubic          : "
        f"{bicubic_mse:.6f}"
    )

    print(
        f"SRCNN            : "
        f"{srcnn_mse:.6f}"
    )

    print(
        "\nPredictions saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()