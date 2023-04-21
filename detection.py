from pathlib import Path

import pandas as pd
import torch

from dataset import get_dataloader
# from detectors.GAN_image_detection import Detector
# from detectors.CNNDetection import detect_dir


def load_detector(detector_id: str) -> torch.nn.Module:
    """
    Load a detector with the given detector id.

    Args:
        detector_id: Detector id

    Returns:
        Detector
    """
    return None


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector_id = "123"
    detector = load_detector(detector_id)

    dataset_id = "MSCOCO2014"  # "StableDiffusion2"
    dataset_subfolder = "valsubset"  # "samples"
    data_dir = Path("data", dataset_id, dataset_subfolder)
    csv_path = Path("csvs", f"{dataset_id}.csv")

    dataloader = get_dataloader(data_dir, label=0, csv_path=csv_path)
    results = []

    for data in dataloader:
        images = data.contiguous().to(device=device)
        batch_results = [0] * images.shape[0]
        results.extend(batch_results)

    df = pd.read_csv(csv_path)
    df[detector_id] = results
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
