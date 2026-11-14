# ML Super-Resolution using SRCNN and FSRCNN-style Model

A PyTorch implementation of single-image super-resolution using SRCNN and an enhanced FSRCNN-style convolutional neural network.

The project reconstructs high-resolution images from degraded low-resolution inputs and compares both learned models against Nearest Neighbor, Bilinear, and Bicubic interpolation.

## Project Overview

Image super-resolution is the process of recovering a high-resolution image from a low-resolution image.

### Objectives

- Implement SRCNN and an enhanced FSRCNN-style model using PyTorch
- Prepare paired low-resolution and high-resolution images
- Train using Mean Squared Error (MSE)
- Evaluate using MSE and PSNR
- Compare SRCNN and the enhanced FSRCNN-style model with traditional interpolation methods
- Visualize reconstruction and training results

## SRCNN Architecture

```text
Input Image
     |
     v
Conv2D: 3 -> 64, 9x9
     |
    ReLU
     |
     v
Conv2D: 64 -> 32, 1x1
     |
    ReLU
     |
     v
Conv2D: 32 -> 3, 5x5
     |
     v
Super-Resolved Image
```

The model contains approximately 20,099 trainable parameters.

| Layer | Configuration |
|---|---|
| Conv1 | 3 -> 64 channels, 9x9 kernel |
| ReLU | Activation |
| Conv2 | 64 -> 32 channels, 1x1 kernel |
| ReLU | Activation |
| Conv3 | 32 -> 3 channels, 5x5 kernel |

## Enhanced FSRCNN-Style Architecture

In addition to SRCNN, the project implements an enhanced FSRCNN-style model. It processes the already-upscaled input at 224 x 224 resolution and uses a residual connection to preserve input information.

```text
Input Image (3 channels)
        |
        v
Feature Extraction: Conv2D 3 -> 64, 5x5 + PReLU
        |
        v
Shrinking: Conv2D 64 -> 16, 1x1 + PReLU
        |
        v
Mapping: 8 x [Conv2D 16 -> 16, 3x3 + PReLU]
        |
        v
Expanding: Conv2D 16 -> 64, 1x1 + PReLU
        |
        v
Reconstruction: Conv2D 64 -> 3, 5x5
        |
        v
Add input through residual connection
        |
        v
Super-Resolved Image
```

| Stage | Configuration | Purpose |
|---|---|---|
| Feature extraction | 3 -> 64 channels, 5x5 + PReLU | Extract image features |
| Shrinking | 64 -> 16 channels, 1x1 + PReLU | Reduce feature channels and computation |
| Mapping | 8 layers, each 16 -> 16 channels, 3x3 + PReLU | Learn feature transformations |
| Expanding | 16 -> 64 channels, 1x1 + PReLU | Expand feature representation |
| Reconstruction | 64 -> 3 channels, 5x5 | Generate RGB output |
| Residual connection | Add original input | Preserve input information |

**Note:** This is an enhanced FSRCNN-style implementation, not an exact reproduction of the canonical FSRCNN architecture.

## Dataset

The project uses a subset of the DIV2K High Resolution Image Dataset.

| Split | Images |
|---|---:|
| Training | 60 |
| Validation | 20 |
| Testing | 20 |
| **Total** | **100** |

The dataset is not included in this repository because of its size.

## Preprocessing

```text
High Resolution Image
        |
        v
   Center Crop
        |
        v
   224 x 224 Target
        |
        v
     Downsample
        |
        v
   112 x 112 Image
        |
        v
Upsample to 224 x 224
        |
        v
    SRCNN Input
```

The SRCNN receives the degraded 224 x 224 image and reconstructs the corresponding 224 x 224 high-resolution target.

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam (both models) |
| Loss Function | Mean Squared Error (both models) |
| Learning Rate | 0.001 (SRCNN) / 0.0001 (FSRCNN-style) |
| Batch Size | 2 (both models) |
| Epochs | 50 (SRCNN) / 100 (FSRCNN-style) |
| Input Size | 224 x 224 |
| Target Size | 224 x 224 |
| Device | CPU / CUDA if available |

The best model is selected using validation PSNR and saved as:

```text
outputs/models/srcnn_best.pth
outputs/models/fsrcnn_best.pth
```

## Evaluation Metrics

### Mean Squared Error (MSE)

MSE measures the average squared difference between the reconstructed image and the ground-truth image. Lower MSE indicates lower reconstruction error.

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures reconstruction quality in decibels (dB). Higher PSNR indicates better reconstruction quality.

## Final Test Results

| Method | MSE | PSNR (dB) |
|---|---:|---:|
| Nearest Neighbor | 0.003415 | 24.667 |
| Bilinear | 0.003406 | 24.678 |
| Bicubic | 0.002705 | 25.678 |
| **SRCNN** | **0.002306** | **26.372** |
| **Enhanced FSRCNN-style** | **0.001658** | **27.805** |

Best validation PSNR: **25.091 dB**

The enhanced FSRCNN-style model achieved the highest test PSNR among the evaluated methods.

Compared with Bicubic:

- Bicubic PSNR: 25.678 dB
- SRCNN PSNR: 26.372 dB (**0.694 dB improvement**)
- Enhanced FSRCNN-style PSNR: 27.805 dB (**2.127 dB improvement**)
- Enhanced FSRCNN-style MSE reduction relative to Bicubic: approximately **38.7%**

Compared with SRCNN, the enhanced FSRCNN-style model improved PSNR by **1.433 dB** and reduced MSE by approximately **28.1%**.

## Results and Visualizations

Generated results are stored in:

```text
outputs/plots/
```

Important outputs include:

```text
loss_curve.png
psnr_curve.png
comparison_0081.png
psnr_comparison.png
mse_comparison.png
```

The visual comparison contains:

- SRCNN Input
- Nearest Neighbor
- Bilinear
- Bicubic
- SRCNN
- Enhanced FSRCNN-style output
- Ground Truth

## Project Structure

```text
ML-Super-Resolution-SRCNN/
|
├── src/
│   ├── dataset.py
│   ├── model.py          # SRCNN
│   ├── fsrcnn.py         # Enhanced FSRCNN-style model
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── results.py
|
├── data/
│   ├── train/
│   ├── val/
│   └── test/
|
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── plots/
|
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/` and `outputs/` directories are excluded from version control because they contain large datasets and generated files.

## Installation

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ML-Super-Resolution-SRCNN
```

### Create virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the DIV2K dataset separately and prepare the required images in the `data/` directory.

Expected structure:

```text
data/
├── train/
├── val/
└── test/
```

The dataset should not be committed to GitHub because of its size.

## Running the Project

### Verify dataset

```bash
python src/dataset.py
```

### Verify model

```bash
python src/model.py
```

### Train SRCNN

```bash
python src/train.py --model srcnn
```

Best checkpoint:

```text
outputs/models/srcnn_best.pth
```

### Train enhanced FSRCNN-style model

```bash
python src/train.py --model fsrcnn
```

Best checkpoint:

```text
outputs/models/fsrcnn_best.pth
```

Training history and plots are saved to:

```text
outputs/plots/
```

### Evaluate SRCNN

```bash
python src/evaluate.py --model srcnn
```

### Evaluate enhanced FSRCNN-style model

```bash
python src/evaluate.py --model fsrcnn
```

Evaluation calculates MSE and PSNR for the selected model and the interpolation baselines.

Predictions are saved to:

```text
outputs/predictions/
```

### Generate visual comparison

```bash
python src/visualize.py
```

### Generate result graphs

```bash
python src/results.py
```

## Conclusion

Both SRCNN and the enhanced FSRCNN-style model learned to reconstruct high-resolution images from degraded inputs.

On the final 20-image test set:

- SRCNN achieved **26.372 dB PSNR** and **0.002306 MSE**.
- The enhanced FSRCNN-style model achieved **27.805 dB PSNR** and **0.001658 MSE**.

Bicubic interpolation achieved 25.678 dB. SRCNN improved PSNR by 0.694 dB over Bicubic, while the enhanced FSRCNN-style model improved it by 2.127 dB in this experiment.

The two neural models used different training configurations, so these results reflect the configurations run rather than a controlled architecture-only comparison.
