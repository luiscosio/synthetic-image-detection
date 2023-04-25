from pathlib import Path
from typing import List

import pandas as pd


def create_dataset_csv(images: List, label: int, csv_path: Path, sep: str = ",") -> None:
    """
    Create a csv file for an image dataset. The csv file contains a row for each image,
    with the image name and the ground truth label. The label is the same for all images.

    Args:
        images: List of images
        label: Ground truth label, 0 for real image, 1 for synthesized
        csv_path: Path to the csv file
        sep: Separator for the csv file
    """
    df = pd.DataFrame(columns=["filename", "label"])
    df["filename"] = [img.name for img in images]
    df["label"] = label
    df.to_csv(csv_path, index=False, sep=sep)
    return None


def add_results_to_csv(csv_path: Path, detector_id: str, results: List, sep: str = ",") -> None:
    """
    Add or overwrite the results of a detector to a dataset csv file.

    Args:
        csv_path: Path to the csv file
        detector_id: ID of the detector
        results: List of labels with the same length as the dataset
        sep: Separator for the csv file
    """
    df = pd.read_csv(csv_path, sep=sep)
    df[detector_id] = results
    df.to_csv(csv_path, index=False, sep=sep)
    return None
