from pathlib import Path
from typing import Dict, NamedTuple, Optional, Protocol, Type

import numpy as np
import torch
from tqdm import tqdm

from dataset import get_dataloader
from detectors.CNNDetection import Detector as CNNDetector
from detectors.DIRE import Detector as DIRE
from detectors.GAN_image_detection import Detector as EnsembleDetector
from detectors.UniversalFakeDetect import Detector as CLIPDetector
from utils.csv_operations import add_results_to_csv

torch.manual_seed(42)


class Detector(Protocol):
    def load_pretrained(self, weights_path: Path) -> None: ...
    def configure(self, device: str, training: bool, **kwargs) -> None: ...


class DetectorTuple(NamedTuple):
    detector: Type[Detector]
    weights_path: Path
    crop_size: Optional[int]


class DatasetTuple(NamedTuple):
    data_dir: Path
    label: int  # 0 for real image, 1 for synthesized


WEIGHTS = Path("weights")
DETECTORS: Dict[str, DetectorTuple] = {
    "CNNDetector_p0.1_crop": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.1.pth"), 224),
    "CNNDetector_p0.1": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.1.pth"), None),
    "CNNDetector_p0.5_crop": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.5.pth"), 224),
    "CNNDetector_p0.5": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.5.pth"), None),
    "EnsembleDetector": DetectorTuple(EnsembleDetector, WEIGHTS.joinpath("EnsembleDetector"), None),
    "CLIPDetector_crop": DetectorTuple(CLIPDetector, WEIGHTS.joinpath("CLIPDetector", "fc_weights.pth"), 224),
    "DIRE": DetectorTuple(DIRE, WEIGHTS.joinpath("DIRE", "lsun_adm.pth"), None),
}

DATA = Path("data")
DATASETS: Dict[str, DatasetTuple] = {
    "MSCOCO2014_val2014": DatasetTuple(DATA.joinpath("MSCOCO2014", "val2014"), 0),
    "MSCOCO2014_valsubset": DatasetTuple(DATA.joinpath("MSCOCO2014", "valsubset"), 0),
    "MSCOCO2014_filtered_val": DatasetTuple(DATA.joinpath("MSCOCO2014", "filtered_val"), 0),
    "SDR": DatasetTuple(DATA.joinpath("HDR", "filtered_images"), 0),
    "StableDiffusion2": DatasetTuple(DATA.joinpath("StableDiffusion2", "filtered_val2014_ts50"), 1),
    "StableDiffusion2_ts20": DatasetTuple(DATA.joinpath("StableDiffusion2", "filtered_val2014_ts20"), 1),
    "StableDiffusion2_ts80": DatasetTuple(DATA.joinpath("StableDiffusion2", "filtered_val2014_ts80"), 1),
    "LDM": DatasetTuple(DATA.joinpath("LDM", "filtered_val2014_ts50"), 1),
    "Midjourney": DatasetTuple(DATA.joinpath("midjourney_v51_cleaned_data", "filtered_images"), 1),
    "BigGAN": DatasetTuple(DATA.joinpath("BigGAN"), 1),
    "StyleGAN2": DatasetTuple(DATA.joinpath("StyleGAN2", "filtered_images"), 1),
    "StyleGAN2r": DatasetTuple(DATA.joinpath("StyleGAN2", "all_real"), 0),
    "StyleGAN2f": DatasetTuple(DATA.joinpath("StyleGAN2", "all_fake"), 1),
    "VQGAN": DatasetTuple(DATA.joinpath("VQGAN", "filtered_images"), 1),
    "Craiyon": DatasetTuple(DATA.joinpath("Craiyon"), 1),
    "DALLE2": DatasetTuple(DATA.joinpath("DALLE2", "DMimageDetection"), 1),
}


def load_detector(detector_id: str, device="cpu") -> (torch.nn.Module, Optional[int]):
    """
    Initialize a detector with the given detector id and load its weights and crop size.

    Args:
        detector_id: Detector id
        device: Device to load the detector on

    Returns:
        Detector and crop size
    """
    detector_cls, weights_path, crop_size = DETECTORS[detector_id]
    detector = detector_cls()
    detector.load_pretrained(weights_path)
    detector.configure(device=device, training=False)
    return detector, crop_size


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    Only works correctly for nested torch.nn.Modules, not for example Modules with Modules in a list.

    Args:
        model: Model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    verbose = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector_id = "DIRE"
    #dataset_id = "MSCOCO2014_filtered_val"
    dataset_id = "SDR"
    compression = None  # 100 - 10 (most compressed) or None

    print(f"Loading detector {detector_id} on device {device}...")
    detector, crop_size = load_detector(detector_id, device)
    parameter_count = count_parameters(detector)
    parameter_count_str = f"at least {parameter_count}" if parameter_count else "an unknown number of"
    print(f"Loaded detector with {parameter_count_str} parameters")

    augmentations = {
        "crop_size": crop_size,
        "compression": compression,
    }

    data_dir, label = DATASETS[dataset_id]
    csv_subname = f"_compression{compression}" if compression is not None else ""
    csv_path = Path("csvs",  f"{dataset_id}{csv_subname}.csv")
    csv_print = f" and creating a CSV in {csv_path}" if not csv_path.exists() else ""

    print(f"Loading dataset {data_dir}{csv_print}...")
    dataloader = get_dataloader(data_dir, label=label, csv_path=csv_path, batch_size=1, augmentations=augmentations)
    results = []
    scores = []
    name_list = []
    times = []

    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for (images, names) in tqdm(dataloader, desc="Performing detection", unit="batch"):
        name_list += names
        with torch.no_grad():
            images = images.contiguous().to(device=device)
            # starter.record()
            batch_labels, batch_scores = detector(images)
            # ender.record()
            # torch.cuda.synchronize()
            # times.append(starter.elapsed_time(ender))
            results.extend(batch_labels.flatten().tolist())
            scores.extend(batch_scores.flatten().tolist())

    # Print inference speeds, skip first batches from calculations due to GPU warm-up
    # times = np.array(times)
    # nskip = 10
    # print(f"mean: {np.mean(times)}")
    # print(f"min: {np.min(times)}")
    # print(f"first: {times[0]}")
    # print(f"mean without first {nskip}: {np.mean(times[nskip:])}")

    if verbose:
        print(f"Number of images: {len(results)}")
        print(f"Number of alleged fake images: {sum(results)}")

    # print(scores)
    # for score, name in zip(scores, name_list):
    #     print(f"{name} synthetic: {round(score, 4):.6f}")
    print(f"Mean: {np.mean(scores)}")
    print("Saving results to CSV...")
    add_results_to_csv(csv_path, detector_id, results, scores)


if __name__ == "__main__":
    main()
