# ML Super-Resolution using SRCNN

A PyTorch implementation of Single Image Super-Resolution using a Super-Resolution Convolutional Neural Network (SRCNN).

The project reconstructs high-resolution images from degraded low-resolution inputs and compares SRCNN against Nearest Neighbor, Bilinear, and Bicubic interpolation.

## Project Overview

Image super-resolution is the process of recovering a high-resolution image from a low-resolution image.

### Objectives

- Implement SRCNN using PyTorch
- Prepare paired low-resolution and high-resolution images
- Train using Mean Squared Error (MSE)
- Evaluate using MSE and PSNR
- Compare SRCNN with traditional interpolation methods
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
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Learning Rate | 0.001 |
| Batch Size | 2 |
| Epochs | 50 |
| Input Size | 224 x 224 |
| Target Size | 224 x 224 |
| Device | CPU / CUDA if available |

The best model is selected using validation PSNR and saved as:

```text
outputs/models/srcnn_best.pth
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

Best validation PSNR: **25.091 dB**

SRCNN achieved the highest test PSNR among the evaluated methods.

Compared with Bicubic:

- Bicubic PSNR: 25.678 dB
- SRCNN PSNR: 26.372 dB
- Improvement: **0.694 dB**
- SRCNN MSE reduction relative to Bicubic: approximately **14.75%**

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
- Ground Truth

## Project Structure

```text
ML-Super-Resolution-SRCNN/
|
├── src/
│   ├── dataset.py
│   ├── model.py
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

### Train

```bash
python src/train.py
```

The best model is saved to:

```text
outputs/models/srcnn_best.pth
```

Training history and plots are saved to:

```text
outputs/plots/
```

### Evaluate

```bash
python src/evaluate.py
```

This calculates MSE and PSNR for SRCNN and the interpolation baselines.

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

The implemented SRCNN successfully learned to reconstruct high-resolution images from degraded inputs.

On the final 20-image test set, SRCNN achieved:

- **Test PSNR: 26.372 dB**
- **Test MSE: 0.002306**

Bicubic interpolation achieved 25.678 dB, while SRCNN achieved 26.372 dB, giving a 0.694 dB improvement in PSNR for this experiment.
