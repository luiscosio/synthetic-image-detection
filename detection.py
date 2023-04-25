from pathlib import Path
from typing import Dict, NamedTuple, Protocol, Type

import pandas as pd
import torch

from dataset import get_dataloader
from detectors.GAN_image_detection import Detector as EnsembleDetector
from detectors.CNNDetection import Detector as CNNDetector

torch.manual_seed(42)


class Detector(Protocol):
    def load_pretrained(self, weights_path: Path) -> None: ...
    def configure(self, device: str, training: bool, **kwargs) -> None: ...


class DetectorTuple(NamedTuple):
    detector: Type[Detector]
    weights_path: Path


WEIGHTS = Path("weights")
DETECTORS: Dict[str, DetectorTuple] = {
    "CNNDetector": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.5.pth")),
    "EnsembleDetector": DetectorTuple(EnsembleDetector, WEIGHTS.joinpath("EnsembleDetector")),
}


def load_detector(detector_id: str, device="cpu") -> torch.nn.Module:
    """
    Initialize a detector with the given detector id and load its weights.

    Args:
        detector_id: Detector id
        device: Device to load the detector on

    Returns:
        Detector
    """
    detector_cls, weights_path = DETECTORS[detector_id]
    detector = detector_cls()
    detector.load_pretrained(weights_path)
    detector.configure(device=device, training=False)
    return detector


def main():
    verbose = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector_id = "CNNDetector"
    detector = load_detector(detector_id, device)

    dataset_id = "StableDiffusion2"  # "MSCOCO2014"
    dataset_subfolder = "samples"  # "valsubset"
    label = 1  # 1 for synthesized dataset, 0 for real

    data_dir = Path("data", dataset_id, dataset_subfolder)
    csv_path = Path("csvs", f"{dataset_id}.csv")

    dataloader = get_dataloader(data_dir, label=label, csv_path=csv_path, batch_size=3)
    results = []

    for (images, names) in dataloader:
        with torch.no_grad():
            images = images.contiguous().to(device=device)
            batch_labels, batch_scores = detector(images)
            results.extend(batch_labels.flatten().tolist())
            print(batch_scores)

    if verbose:
        print(f"Number of images: {len(results)}")
        print(f"Number of fake images: {sum(results)}")
        print(results)

    df = pd.read_csv(csv_path)
    df[detector_id] = results
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
