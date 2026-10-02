import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SuperResolutionDataset(Dataset):
    """
    Dataset for SRCNN image super-resolution.

    Each sample consists of:
        input  -> degraded/upscaled image
        target -> original high-resolution target
    """

    def __init__(self, root_dir):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")

        self.image_names = sorted(
            [
                filename
                for filename in os.listdir(self.input_dir)
                if filename.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]

        input_path = os.path.join(self.input_dir, image_name)
        target_path = os.path.join(self.target_dir, image_name)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        input_tensor = self.to_tensor(input_image)
        target_tensor = self.to_tensor(target_image)

        return input_tensor, target_tensor


if __name__ == "__main__":
    train_dataset = SuperResolutionDataset("data/train")

    print("Number of training samples:", len(train_dataset))

    input_tensor, target_tensor = train_dataset[0]

    print("Input shape:", input_tensor.shape)
    print("Target shape:", target_tensor.shape)
    print("Input range:", input_tensor.min().item(), "to", input_tensor.max().item())
    print("Target range:", target_tensor.min().item(), "to", target_tensor.max().item())