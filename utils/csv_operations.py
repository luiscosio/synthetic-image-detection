from pathlib import Path
from typing import List

import pandas as pd


def create_dataset_csv(images: List, label: int, csv_path: Path, sep: str = ",") -> None:
    """
    Create a CSV file for an image dataset. The CSV file contains a row for each image,
    with the image name and the ground truth label. The label is the same for all images.

    Args:
        images: List of images
        label: Ground truth label, 0 for real image, 1 for synthesized
        csv_path: Path to the CSV file
        sep: Separator for the CSV file
    """
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(columns=["filename", "label"])
    df["filename"] = [img.name for img in images]
    df["label"] = label
    df.to_csv(csv_path, index=False, sep=sep)


def add_results_to_csv(csv_path: Path, detector_id: str, results: List, scores: List, sep: str = ",") -> None:
    """
    Add or overwrite the results of a detector to a CSV dataset file.

    Args:
        csv_path: Path to the CSV file
        detector_id: ID of the detector
        results: List of labels with the same length as the dataset
        scores: List of scores with the same length as the dataset
        sep: Separator for the CSV file
    """
    print(f"Saving results to {csv_path.absolute()}...")
    df = pd.read_csv(csv_path, sep=sep)
    df[f"{detector_id}_labels"] = results
    df[f"{detector_id}_scores"] = scores
    df.to_csv(csv_path, index=False, sep=sep)
