import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network.

    Architecture:
        Conv 3 -> 64, 9x9
        ReLU
        Conv 64 -> 32, 1x1
        ReLU
        Conv 32 -> 3, 5x5
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=9,
                padding=4
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=1,
                padding=0
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=5,
                padding=2
            )
        )

    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":
    model = SRCNN()

    print(model)

    # Test with a dummy RGB image
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)

    # Count trainable parameters
    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", parameters)