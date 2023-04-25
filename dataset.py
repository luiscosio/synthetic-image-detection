from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import create_dataset_csv

torch.manual_seed(42)

# Possible image extensions used by Pillow, copied from PyTorch's torchvision.datasets source code
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, label: int, csv_path: Path, transform=None):
        """
        Initialize an image dataset containing either only real or synthesized images.

        Args:
            data_dir:
            label:
            csv_path:
            transform:
        """
        self._dir = data_dir
        self._label = label
        self._transform = transform
        self._csv_path = csv_path
        self._images = [img for img in data_dir.iterdir() if img.suffix in IMG_EXTENSIONS]
        if not self._csv_path.exists():
            create_dataset_csv(self._images, label, csv_path)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img_path = self._images[idx]
        img_name = img_path.name
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)  # (C x H x W)
        #if self._transform:
        #    img = self._transform(img)
        return img, img_name


def get_dataloader(data_dir: Path, label: int, csv_path: Path, transform=None, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    """
    Create a dataloader for the given directory.

    Args:
        data_dir: Directory containing images
        label: Label for the images
        csv_path: Path to the csv file
        transform: Data augmentations
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Dataloader for the given directory
    """
    dataset = ImageDataset(data_dir, label, csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

