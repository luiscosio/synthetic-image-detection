import io
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import create_dataset_csv

torch.manual_seed(42)

# Possible image extensions used by Pillow, copied from PyTorch's torchvision.datasets source code
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, label: int, csv_path: Path, augmentations: Optional[Dict] = None):
        """
        Initialize an image dataset containing either only real or synthesized images.

        Args:
            data_dir: Path to the directory containing the images
            label: Ground truth label, 0 for real image, 1 for synthesized
            csv_path: Path a CSV file to save the image names and labels
            augmentations: Dictionary containing the augmentation parameters
        """
        self._dir = data_dir
        self._label = label
        self._compression = None
        self._csv_path = csv_path

        self._images = [img for img in data_dir.iterdir() if img.suffix in IMG_EXTENSIONS]
        if not self._csv_path.exists():
            create_dataset_csv(self._images, label, csv_path)

        transform_list = []
        if isinstance(augmentations, Dict):
            crop_size = augmentations.get("crop_size")
            compression = augmentations.get("compression")
            if crop_size:
                transform_list.append(transforms.CenterCrop(crop_size))
            if compression:
                transform_list.append(transforms.Lambda(lambda x: jpeg_compression(x, quality=compression)))

        transform_list.append(transforms.ToTensor())  # (C x H x W)
        self._transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img_path = self._images[idx]
        img_name = img_path.name
        img = Image.open(img_path).convert("RGB")
        img = self._transform(img)
        return img, img_name


def jpeg_compression(image: Image, quality: int = 80) -> Image:
    """
    Apply JPEG compression to an image.

    Args:
        image: Image to compress
        quality: Quality of the compression

    Returns:
        Compressed image
    """
    temp_stream = io.BytesIO()
    image.convert("RGB").save(temp_stream, "JPEG", quality=quality, optimize=True)
    temp_stream.seek(0)
    return Image.open(temp_stream)


def get_dataloader(data_dir: Path,
                   label: int,
                   csv_path: Path,
                   augmentations: Optional[Dict] = None,
                   batch_size: int = 32,
                   num_workers: int = 0,
                   ) -> DataLoader:
    """
    Create a dataloader for the given directory.

    Args:
        data_dir: Directory containing images
        label: Label for the images
        csv_path: Path a CSV file to save the image names and labels
        augmentations: Dictionary containing the augmentation parameters
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Dataloader for the given directory
    """
    dataset = ImageDataset(data_dir, label, csv_path, augmentations=augmentations)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataloader

