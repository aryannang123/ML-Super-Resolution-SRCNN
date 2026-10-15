import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

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
    help="Model to train"
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = "outputs/models"
PLOT_DIR = "outputs/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# MODEL SELECTION
# ============================================================

if args.model == "srcnn":

    model = SRCNN()
    MODEL_NAME = "srcnn"

else:

    model = FSRCNN()
    MODEL_NAME = "fsrcnn"


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
# VALIDATION
# ============================================================

def validate(model, dataloader, criterion):

    model.eval()

    total_loss = 0.0
    total_pixels = 0

    with torch.no_grad():

        for inputs, targets in dataloader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            outputs = torch.clamp(outputs, 0, 1)

            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            total_pixels += inputs.size(0)

    average_loss = total_loss / total_pixels

    psnr = calculate_psnr(average_loss)

    return average_loss, psnr


# ============================================================
# SAVE PLOTS
# ============================================================

def save_plots(
    train_losses,
    val_losses,
    val_psnrs
):

    # Loss curve
    plt.figure()

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        f"{MODEL_NAME.upper()} Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{PLOT_DIR}/{MODEL_NAME}_loss_curve.png"
    )

    plt.close()


    # PSNR curve
    plt.figure()

    plt.plot(
        val_psnrs,
        label="Validation PSNR"
    )

    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")

    plt.title(
        f"{MODEL_NAME.upper()} Validation PSNR"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{PLOT_DIR}/{MODEL_NAME}_psnr_curve.png"
    )

    plt.close()


# ============================================================
# TRAINING
# ============================================================

def main():

    print("\n" + "=" * 50)
    print(f"{MODEL_NAME.upper()} TRAINING")
    print("=" * 50)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = SuperResolutionDataset(
        TRAIN_DIR
    )

    val_dataset = SuperResolutionDataset(
        VAL_DIR
    )

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

    print(
        f"Training images: {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(val_dataset)}"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model.to(DEVICE)

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )


    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    train_losses = []
    val_losses = []
    val_psnrs = []

    best_psnr = -float("inf")


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

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


            running_loss += (
                loss.item() *
                inputs.size(0)
            )


            progress_bar.set_postfix(
                loss=loss.item()
            )


        # Average training loss
        train_loss = (
            running_loss /
            len(train_dataset)
        )


        # Validation
        val_loss, val_psnr = validate(
            model,
            val_loader,
            criterion
        )


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)


        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.3f} dB"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/{MODEL_NAME}_best.pth"
            )

            print(
                f"  ✓ Best {MODEL_NAME.upper()} model saved"
            )


    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_psnrs": val_psnrs
        },
        f"{PLOT_DIR}/{MODEL_NAME}_training_history.pth"
    )


    # --------------------------------------------------------
    # SAVE PLOTS
    # --------------------------------------------------------

    save_plots(
        train_losses,
        val_losses,
        val_psnrs
    )


    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)

    print(
        f"Best Validation PSNR: "
        f"{best_psnr:.3f} dB"
    )

    print(
        f"Model saved to: "
        f"{MODEL_DIR}/{MODEL_NAME}_best.pth"
    )

    print(
        f"Plots saved to: "
        f"{PLOT_DIR}/"
    )


if __name__ == "__main__":
    main() 