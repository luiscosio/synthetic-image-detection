import argparse
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Protocol, Tuple, Type, Union

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
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


DETECTOR_DICT: Dict[str, Type[Detector]] = {
    "clipdetector": CLIPDetector,
    "cnndetector": CNNDetector,
    "ensembledetector": EnsembleDetector,
    "dire": DIRE,
}

WEIGHTS = Path("weights")
DETECTORS: Dict[str, DetectorTuple] = {
    "CNNDetector_p0.1_crop": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.1.pth"), 224),
    "CNNDetector_p0.1": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.1.pth"), None),
    "CNNDetector_p0.5_crop": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.5.pth"), 224),
    "CNNDetector_p0.5": DetectorTuple(CNNDetector, WEIGHTS.joinpath("CNNDetector", "blur_jpg_prob0.5.pth"), None),
    "EnsembleDetector": DetectorTuple(EnsembleDetector, WEIGHTS.joinpath("EnsembleDetector"), None),
    "EnsembleDetector_crop": DetectorTuple(EnsembleDetector, WEIGHTS.joinpath("EnsembleDetector"), 224),
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


def load_detector_from_id(detector_id: str, device: str = "cpu") -> (Detector, Optional[int]):
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


def load_detector_from_args(detector_class: str, weights_path: Path, device: str = "cpu") -> Detector:
    """
    Initialize a detector with the given detector id and weights.

    Args:
        detector_class: Name of a detector's class
        weights_path: Path to specific weights
        device: Device to load the detector on

    Returns:
        Detector
    """
    detector_cls = DETECTOR_DICT[detector_class]
    detector = detector_cls()
    detector.load_pretrained(weights_path)
    detector.configure(device=device, training=False)
    return detector


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


def start_detection(detector_id: str,
                    detector_class: str,
                    detector_weights: Path,
                    dataset_id: str,
                    data_dir: Path,
                    label: int,
                    batch_size: int,
                    crop_size: Union[None, int, Tuple[int, ...]],
                    compression: int,
                    resize: Union[None, int, Tuple[int, ...]],
                    resize_method: str,
                    csv_path: Path,
                    verbose: bool,
                    force_cpu: bool,
                    ) -> None:
    # For description, see the Argument Parser below or $ python detection.py --help
    device = "cpu"
    if force_cpu:
        print("Forcing CPU use")

    else:
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"Using {device}")
        else:
            print("No GPU or CUDA drivers found, using CPU")

    # Validate dataset arguments
    if data_dir is None:
        if dataset_id is None and csv_path is None:
            raise ValueError(f"Dataset ID (-ds) needs to be given if dataset directory and CSV output file are not specified")
        if dataset_id in DATASETS.keys():
            data_dir, label = DATASETS[dataset_id]
        else:
            raise ValueError(f"The given dataset ID (-ds) {dataset_id} does not match any hard-coded "
                             f"dataset values. Either input also a dataset directory argument (-dsd) "
                             f"or a hard-coded dataset ID {list(DATASETS.keys())}")
    elif label is None:
        raise ValueError("A custom dataset directory was given, but no dataset label")

    elif dataset_id is None and csv_path is None:
        dataset_id = data_dir.name

    # Validate detector arguments and load the detector
    if detector_class is None:
        if detector_id in DETECTORS.keys():
            print(f"Loading detector {detector_id}...")
            detector, crop_size_coded = load_detector_from_id(detector_id, device)
            if crop_size_coded is not None:
                crop_size = crop_size_coded
        else:
            raise ValueError(f"The given detector ID (-d) {detector_id} does not match any hard-coded "
                             f"detector values. Either input also the detector class (-dc) and weights (-dw) arguments "
                             f"or a hard-coded detector ID {list(DETECTORS.keys())}")
    elif detector_weights is None:
        raise ValueError(f"Detector weights path (-dw) needs to be specified when the detector class is given")
    else:
        detector = load_detector_from_args(detector_class, detector_weights, device)

    parameter_count = count_parameters(detector)
    parameter_count_str = f"at least {parameter_count}" if parameter_count else "an unknown number of"
    print(f"Loaded detector with {parameter_count_str} parameters")

    # Augmentations
    resize_aug = (resize, InterpolationMode(resize_method))  # (size, method), size=None for no resizing
    augmentations = {
        "resize": resize_aug,
        "crop_size": crop_size,
        "compression": compression,
    }
    print(f"Using the following data augmentations: {augmentations}")

    # CSV path naming
    csv_subname = ""
    resize_str = ""
    if not csv_path:
        if isinstance(resize, int):
            resize_str = f"_rs{resize}_{resize_method}"
        elif isinstance(resize, tuple):
            resize_str = f"_rs{resize[0]}{resize[1]}_{resize_method}"
        csv_subname += resize_str if resize_str else ""
        csv_subname += f"_compression{compression}" if compression is not None else ""
        csv_path = Path("csvs",  f"{dataset_id}{csv_subname}.csv")

    if not csv_path.suffix.lower() == ".csv":
        raise ValueError(f"Output file should be a CSV file")

    csv_print = f", creating a CSV in {csv_path.absolute()}" if not csv_path.exists() else ""
    csv_print += f" and outputs will be saved to {csv_path} under {detector_id} columns"

    print(f"Loading dataset {data_dir}{csv_print}...")
    dataloader = get_dataloader(data_dir, label=label, csv_path=csv_path, batch_size=batch_size, augmentations=augmentations)

    results = []
    scores = []
    name_list = []
    times = []

    # Code regarding speed measurements are commented out
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

    # Print inference speeds, skip first batches from calculations due to initialization GPU warm-up
    # times = np.array(times)
    # nskip = 10
    # print(f"mean: {np.mean(times)}")
    # print(f"min: {np.min(times)}")
    # print(f"first: {times[0]}")
    # print(f"mean without first {nskip}: {np.mean(times[nskip:])}")

    if verbose:
        print(f"Number of images: {len(results)}")
        print(f"Number of alleged fake images: {sum(results)}")
        print(f"Mean sidmoid output: {np.mean(scores)}")

    # print(scores)
    # for score, name in zip(scores, name_list):
    #     print(f"{name} synthetic: {round(score, 4):.6f}")
    add_results_to_csv(csv_path, detector_id, results, scores)


def tuple_or_int_type(strings) -> Union[None, int, Tuple[int, ...]]:
    """
    Custom type for argument parsing.
    Example argument inputs include: 1, "None", "1, 2", "1,2", "(1, 2)"

    Args:
        strings: Argument input

    Returns:
        None if the input is "None" or None,
        Tuple of integers if the input is multiple integers (with or without parenthesis),
        integer if the input matches an integer
    """
    if strings is None or strings == "None":
        return None
    strings = strings.replace("(", "").replace(")", "").replace(" ", "")
    mapped_int = map(int, strings.split(","))
    tup_int = tuple(mapped_int)
    if len(tup_int) == 1:
        return tup_int[0]
    return tup_int


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     allow_abbrev=False,
                                     description="Perform synthetic image detection with one of the setup "
                                                 "detectors, on a dataset consisting of either real or fake images. "
                                                 "By default, any data augmentation combination (except cropping) "
                                                 "produces a unique CSV file for the results. If a CSV file has results "
                                                 "matching the given detector ID, even if the configurations are "
                                                 "different, they are overwritten.")

    parser.add_argument("--detector", "-d", type=str, required=True, dest="detector_id",
                        help=f"ID of a synthetic image detector that is used for the detection. "
                             f"The ID is only used for naming and hard-coded detector weight paths. "
                             f"Can be arbitrarily named when not using hard-coded paths, as detector "
                             f"class and weights arguments needs to be given. "
                             f"Hard-coded IDs are: {list(DETECTORS.keys())}")

    parser.add_argument("--detector-class", "-dc", choices=["cnndetector", "clipdetector", "ensembledetector", "dire"],
                        default=None, dest="detector_class",
                        help="Name of the detector class to use. If not given, a hard-coded detector is attempted to be "
                             "loaded using the detector ID")

    parser.add_argument("--detector-weights", "-dw", type=Path, default=None, dest="detector_weights",
                        help="Path to the detector weights. Note that the expected input differs between the "
                             "detectors: CLIP- and CNNDetector expect a path to the weights file (.pth), "
                             "EnsembleDetector a path to its weight directory holding the 5 weights, and DIRE a path "
                             "to the weights file (.pth) but with the assumption that the diffusion weights are "
                             "in the same directory as well")

    parser.add_argument("--dataset", "-ds", type=str, dest="dataset_id",
                        help="ID of an image dataset that is used as input for the detection. "
                             "The ID is only used for naming and hard-coded dataset paths")

    parser.add_argument("--dataset-dir", "-dsd", type=Path, default=None, dest="data_dir",
                        help="Path to the input dataset directory. If not given, the location is assumed to match "
                             "a hard-coded value for the given dataset ID")

    parser.add_argument("--dataset-label", "-dsl", choices=[0, 1], type=int, dest="label",
                        help="Label of the used image dataset, 1 for synthetic, 0 for real. "
                             "Ignored if a hard-coded dataset is used")

    parser.add_argument("--batch_size", "-bs", type=int, default=1, dest="batch_size",
                        help="Number of images input to a detector at the same time. Higher size can be faster "
                             "but requires more memory. Images in a batch need to have the same resolution")

    parser.add_argument("--crop-size", "-cs", type=tuple_or_int_type, default=(224, 224), dest="crop_size",
                        help="Size (h, w) of center-cropping, performed after resizing. A single integer input results "
                             "in a square crop. None to not include cropping. Default crop size is (224, 224) as some of "
                             "the included detectors were trained with data of such size and may require the input to "
                             "be of that size. None to exclude cropping. Hard-coded crop sizes are used with specific "
                             "detector IDs.")

    parser.add_argument("--compression", "-c", type=int, default=None, dest="compression",
                        help="Compression quality factor applied to images (100 - 10, with 10 being most compressed)."
                             "Leave out to not perform compression")

    parser.add_argument("--resize", "-rs", type=tuple_or_int_type, default=None, dest="resize",
                        help="Size (h, w) to resize images, performed before center-cropping")

    parser.add_argument("--resize-method", "-rsm",
                        choices=["nearest", "nearest-exact", "bilinear", "bicubic","box", "hamming", "lanczos"],
                        default="bilinear", dest="resize_method",
                        help="Resizing method used when resize argument is also supplied")

    parser.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                        help="Print more information")

    parser.add_argument("--output", "-o", type=Path, default=None, dest="csv_path",
                        help="Optional output path (.csv) for the produced CSV file with the results. "
                             "By default, the CSV is placed in the CSV folder with the name being "
                             "derived from the dataset and augmentations")

    parser.add_argument("--force-cpu", "-f", action="store_true", dest="force_cpu",
                        help="Force the using of CPU, not recommended")

    args = parser.parse_args()
    start_detection(**vars(args))


if __name__ == "__main__":
    main()
