import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """
    Enhanced FSRCNN-style network.

    Input:  [B, 3, 224, 224]
    Output: [B, 3, 224, 224]
    """

    def __init__(self, num_mapping_layers=8):
        super(FSRCNN, self).__init__()

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.PReLU(64)
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(
                64,
                16,
                kernel_size=1
            ),
            nn.PReLU(16)
        )

        # Non-linear mapping
        mapping_layers = []

        for _ in range(num_mapping_layers):

            mapping_layers.append(
                nn.Conv2d(
                    16,
                    16,
                    kernel_size=3,
                    padding=1
                )
            )

            mapping_layers.append(
                nn.PReLU(16)
            )

        self.mapping = nn.Sequential(
            *mapping_layers
        )

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(
                16,
                64,
                kernel_size=1
            ),
            nn.PReLU(64)
        )

        # Reconstruction
        self.reconstruction = nn.Conv2d(
            64,
            3,
            kernel_size=5,
            padding=2
        )

    def forward(self, x):

        # Keep original input for residual learning
        residual = x

        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.reconstruction(x)

        # Residual reconstruction
        x = x + residual

        return x