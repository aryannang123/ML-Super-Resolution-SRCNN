import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SuperResolutionDataset
from model import SRCNN


# =========================
# Configuration
# =========================

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

MODEL_DIR = "outputs/models"
PLOT_DIR = "outputs/plots"

BATCH_SIZE = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Create directories
# =========================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================
# PSNR
# =========================

def calculate_psnr(mse):

    if mse <= 0:
        return float("inf")

    return 10 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


# =========================
# Validation
# =========================

def validate(model, dataloader, criterion):

    model.eval()

    total_squared_error = 0.0
    total_pixels = 0

    with torch.no_grad():

        for inputs, targets in dataloader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            outputs_clamped = torch.clamp(
                outputs,
                0.0,
                1.0
            )

            squared_error = torch.sum(
                (outputs_clamped - targets) ** 2
            ).item()

            total_squared_error += squared_error
            total_pixels += targets.numel()

    average_mse = (
        total_squared_error / total_pixels
    )

    psnr = calculate_psnr(average_mse)

    return average_mse, psnr


# =========================
# Plot training history
# =========================

def save_plots(
    train_losses,
    val_losses,
    val_psnrs
):

    epochs = range(1, len(train_losses) + 1)

    # -------------------------
    # Loss curve
    # -------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("SRCNN Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(
        PLOT_DIR,
        "loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------
    # PSNR curve
    # -------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        val_psnrs,
        label="Validation PSNR"
    )

    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("SRCNN Validation PSNR")
    plt.legend()
    plt.grid(True)

    psnr_path = os.path.join(
        PLOT_DIR,
        "psnr_curve.png"
    )

    plt.savefig(
        psnr_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPlots saved:")
    print(loss_path)
    print(psnr_path)


# =========================
# Main
# =========================

def main():

    print("=" * 55)
    print("SRCNN TRAINING")
    print("=" * 55)

    print("Device:", DEVICE)

    # =========================
    # Dataset
    # =========================

    train_dataset = SuperResolutionDataset(
        TRAIN_DIR
    )

    val_dataset = SuperResolutionDataset(
        VAL_DIR
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # =========================
    # DataLoaders
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================
    # Model
    # =========================

    model = SRCNN().to(DEVICE)

    # =========================
    # Loss
    # =========================

    criterion = nn.MSELoss()

    # =========================
    # Optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # =========================
    # History
    # =========================

    train_losses = []
    val_losses = []
    val_psnrs = []

    best_val_psnr = float("-inf")

    print("\nStarting training...")
    print("=" * 55)

    # =========================
    # Training loop
    # =========================

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        for inputs, targets in progress_bar:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(
                outputs,
                targets
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.6f}"
            )

        # Average training loss
        train_loss = (
            running_loss /
            len(train_loader)
        )

        # Validation
        val_loss, val_psnr = validate(
            model,
            val_loader,
            criterion
        )

        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        # Print results
        print(
            f"\nEpoch [{epoch + 1:02d}/{NUM_EPOCHS}]"
        )

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Val Loss   : {val_loss:.6f}"
        )

        print(
            f"Val PSNR   : {val_psnr:.3f} dB"
        )

        # =========================
        # Save best model
        # =========================

        if val_psnr > best_val_psnr:

            best_val_psnr = val_psnr

            model_path = os.path.join(
                MODEL_DIR,
                "srcnn_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"✓ New best model saved "
                f"({best_val_psnr:.3f} dB)"
            )

    # =========================
    # Save history
    # =========================

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_psnrs": val_psnrs
    }

    history_path = os.path.join(
        PLOT_DIR,
        "training_history.pth"
    )

    torch.save(
        history,
        history_path
    )

    # =========================
    # Generate plots
    # =========================

    save_plots(
        train_losses,
        val_losses,
        val_psnrs
    )

    # =========================
    # Finished
    # =========================

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)

    print(
        f"Best validation PSNR: "
        f"{best_val_psnr:.3f} dB"
    )

    print(
        f"Best model: "
        f"{MODEL_DIR}/srcnn_best.pth"
    )

    print(
        f"History: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()