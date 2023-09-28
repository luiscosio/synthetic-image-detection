from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def get_detector_values_and_truths(csv_paths: List[Path],
                                   detector_id: str,
                                   value_type: Literal["labels", "scores"],
                                   sep: str = ",") -> (List[float], List[int]):
    """
    Get the values and ground truth labels from a CSV file for a detector.
    The value can refer to a predicted label or a float output by the detector.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        value_type: Type of the value, either "labels" or "scores"
        sep: Separator of the CSV file

    Returns:
        List of values and list of ground truth labels
    """
    values = []
    gt = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep=sep)
        gt.extend(df["label"].tolist())
        values.extend(df[f"{detector_id}_{value_type}"].tolist())

    return values, gt


def calculate_detector_accuracy(csv_paths: List[Path], detector_id: str, sep: str = ",") -> float:
    """
    Calculate the total accuracy of a detector from the CSV dataset result files.
    The CSV files are assumed to contain a column with the detector's predictions.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file

    Returns:
        Total accuracy of the detector on the datasets
    """
    preds, gt = get_detector_values_and_truths(csv_paths, detector_id, "labels", sep=sep)
    return accuracy_score(gt, preds)


def calculate_detector_accuracies(csv_path: Union[Path, List[Path]],
                                  detector_id: str,
                                  threshold: Optional[float] = None,
                                  csv_filter: Optional[str] = None,
                                  sep: Union[str, List[str]] = ",") -> Dict[str, float]:
    """
    Calculate the separate accuracies for a detector from the CSV dataset result files.
    The input path can be either a directory with CSV result files, a single CSV file, or a list of them.
    By default, the labels column is used, but a custom threshold can be given to use the scores column.

    Args:
        csv_path: Path to a directory with CSV files, single CSV file, or a list of them
        detector_id: ID of the detector
        threshold: Optional threshold for accuracy, already rounded labels column used if not given
        csv_filter: Only the CSV files with the given pattern in their names are included
        sep: Separator for CSV files, or a list of them if they differ

    Returns:
        A dictionary with datasets as keys and accuracies of the detector as values
    """
    accuracies = {}
    if isinstance(csv_path, Path):
        if csv_path.is_dir():
            csv_paths = list(csv_path.glob("*.csv"))
        else:
            csv_paths = [csv_path]
    else:
        csv_paths = csv_path

    if csv_filter:
        csv_paths = [path for path in csv_paths if csv_filter in path.name]
    if not csv_paths:
        raise ValueError("Did not find any CSV files matching the inputs")

    if isinstance(sep, str):
        sep = [sep] * len(csv_paths)

    assert len(sep) == len(csv_paths)

    for path, separator in zip(csv_paths, sep):
        df = pd.read_csv(path, sep=separator)
        detectors = [k.replace("_labels", "") for k in df.columns if "_labels" in k]
        if detector_id not in detectors:
            print(f"{detector_id} not found in {path} dataset, skipping")
            continue

        labels = np.array(df["label"].to_list())
        values = df[detector_id + "_labels"] if threshold is None else df[detector_id + "_scores"]
        values = np.array(values.to_list())
        if threshold is not None:
            values = values >= threshold
        accuracy = np.sum(values == labels) / len(values)
        accuracies[path.name] = accuracy

    return accuracies


def calculate_dataset_accuracies(csv_path: Path,
                                 threshold: Optional[float] = None,
                                 specific_detector: Optional[str] = None,
                                 sep: str = ",") -> Dict[str, float]:
    """
    Calculate the accuracies from the results of detectors on a CSV file for a dataset.
    By default, the labels column is used, but a custom threshold can be given to use the scores column.
    Given a specific detector, only its accuracy is reported.

    Args:
        csv_path: Path to a dataset CSV file
        threshold: Optional threshold for accuracy, already rounded labels column used if not given
        specific_detector: Optional detector ID to filter only its results
        sep: Separator for the CSV file

    Returns:
        A dictionary with detectors as keys and accuracies as values
    """
    accuracies = {}
    df = pd.read_csv(csv_path, sep=sep)
    detectors = [k.replace("_labels", "") for k in df.columns if "_labels" in k]

    if isinstance(specific_detector, str):
        if specific_detector in detectors:
            detectors = [specific_detector]
        else:
            print(f"{specific_detector} not found in {csv_path.name} dataset, no accuracies calculated")
            return dict()

    labels = np.array(df["label"].to_list())

    for detector in detectors:
        values = df[detector + "_labels"] if threshold is None else df[detector + "_scores"]
        values = np.array(values.to_list())
        if threshold is not None:
            values = values >= threshold
        accuracy = np.sum(values == labels) / len(values)
        accuracies[detector] = accuracy
    return accuracies


def print_dataset_accuracies(csv_path: Path,
                             threshold: Optional[float] = None,
                             specific_detector: Optional[str] = None,
                             sep: str = ",",) -> None:
    """
    Print the accuracies from the results of detectors on a CSV file for a dataset.
    By default, the labels column is used, but a custom threshold can be given to use the scores column.
    Given a specific detector, only its accuracy is reported.

    Args:
        csv_path: Path to a dataset CSV file
        threshold: Optional threshold for accuracy, already rounded labels column used if not given
        specific_detector: Optional detector ID to filter only its results
        sep: Separator for the CSV file
    """
    accuracies = calculate_dataset_accuracies(csv_path, threshold, specific_detector, sep)
    postfix = "using the _labels column" if threshold is None else f"with threshold {threshold}"
    print(f"Accuracies on the {csv_path.name} dataset {postfix}")
    for k, v in accuracies.items():
        print(f"{k}: {v}")


def print_detector_accuracies(csv_path: Union[Path, List[Path]],
                              detector_id: str,
                              threshold: Optional[float] = None,
                              csv_filter: Optional[str] = None,
                              sep: Union[str, List[str]] = ",") -> None:
    """
    Print the separate accuracies for a detector from the CSV dataset result files.
    The input path can be either a directory with CSV result files, a single CSV file, or a list of them.
    By default, the labels column is used, but a custom threshold can be given to use the scores column.

    Args:
        csv_path: Path to a directory with CSV files, single CSV file, or a list of them
        detector_id: ID of the detector
        threshold: Optional threshold for accuracy, already rounded labels column used if not given
        csv_filter: Only the CSV files with the given pattern in their names are included
        sep: Separator for CSV files, or a list of them if they differ
    """
    try:
        accuracies = calculate_detector_accuracies(csv_path, detector_id, threshold, csv_filter, sep)
    except ValueError as e:
        print(e)
        return

    if not accuracies:
        print(f"No results found for {detector_id} on the given datasets")
        return None

    postfix = "using the _labels column" if threshold is None else f"with threshold {threshold}"
    postfix += f' and "{csv_filter}" CSV filter' if csv_filter else ""
    print(f"Accuracies for {detector_id} {postfix}")
    for k, v in accuracies.items():
        print(f"{k}: {v}")


def print_latex_accuracy_table(datasets: Dict[str, Union[Path, List[Path]]],
                               detectors: Dict[str, str],
                               thresholds: Optional[List[float]] = None,
                               sep: str = ",") -> None:
    """
    Print a LaTeX table contents with the accuracies of given detectors on given datasets.
    Each given dataset is expected to have results for each given detector.

    Multiple variations of a dataset can be given as a list, causing the use of multicolumns.
    The variations are in the order they are input, their distinct variation property is not shown.
    Each given dataset should have the same number of variations.

    Args:
        datasets: Dictionary of printed dataset names as keys and (lists of) Paths as values
        detectors: Dictionary of printed detector names as keys and IDs as values
        thresholds: Optional thresholds for accuracy, in the same order as the matching detectors
        sep: Separator for the CSV files, expected to be the same for all
    """
    if thresholds is None:
        thresholds = [None] * len(detectors)

    # Check how many columns for each detector
    multicolumn = 1
    for _, data_paths in datasets.items():
        if isinstance(data_paths, list):
            multicolumn = len(data_paths)
        break

    # Set the header row
    table = "\hline\n"
    for detector_name in detectors.keys():
        if multicolumn > 1:
            table += f" & \multicolumn{{{multicolumn}}}{{c|}}{{{detector_name}}}"
        else:
            table += f" & {{{detector_name}}}"
    table += "\\\\\n\hline"

    # Set a dataset's accuracies on one row, with each detector making up a column for each dataset variation
    for data_name, data_paths in datasets.items():
        table += f"\n{data_name.ljust(10)} "
        for idx, (detector_name, detector_id) in enumerate(detectors.items()):
            if isinstance(data_paths, Path):
                data_paths = [data_paths]
            for data_path in data_paths:
                accs = calculate_dataset_accuracies(data_path, thresholds[idx], detector_id, sep)
                table += f"& {accs[detector_id]}".ljust(10)
        table += "\\\\"

    print(table)


def calculate_balanced_threshold_from_roc(csv_paths: List[Path], detector_id: str, sep: str = ",") -> float:
    """
    Calculate a threshold for accuracy from a receiver operating characteristics curve.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file:

    Returns:
        Balanced threshold from the ROC curve
    """
    scores, gt = get_detector_values_and_truths(csv_paths, detector_id, "scores", sep=sep)
    fpr, tpr, thresholds = roc_curve(gt, scores)
    return thresholds[np.argmax(tpr - fpr)]


def calculate_detector_auc(csv_paths: List[Path], detector_id: str, sep: str = ",") -> float:
    """
    Calculate the total AUC of a detector on given datasets. The CSV files are assumed to contain
    a column with the detector's scores.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file

    Returns:
        Total AUC of the detector on the datasets
    """
    scores, gt = get_detector_values_and_truths(csv_paths, detector_id, "scores", sep=sep)
    return roc_auc_score(gt, scores)


def calculate_detector_average_precision(csv_paths: List[Path], detector_id: str, sep: str = ",") -> float:
    """
    Calculate the total average precision of a detector on given datasets. The CSV files are assumed to contain
    a column with the detector's scores.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file

    Returns:
        Total average precision of the detector on the datasets
    """
    scores, gt = get_detector_values_and_truths(csv_paths, detector_id, "scores", sep=sep)
    return average_precision_score(gt, scores)


def plot_auc_and_ap(csv_paths: List[Path], detector_id: str, sep: str = ",") -> None:
    """
    Plot the ROC and RP curves and their areas under the curves for a detector on given datasets.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file
    """
    scores, gt = get_detector_values_and_truths(csv_paths, detector_id, "scores", sep=sep)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6))
    plt.rcParams.update({'font.size': 12})

    # ROC curve
    fpr, tpr, _ = roc_curve(gt, scores)
    auc = roc_auc_score(gt, scores)
    ax1.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("AUC")
    ax1.fill_between(fpr, tpr, alpha=0.2)
    ax1.legend(loc="lower right")

    # RP curve
    precision, recall, _ = precision_recall_curve(gt, scores)
    ap = average_precision_score(gt, scores)
    ax2.plot(recall, precision, label=f"AP={ap:.4f}")
    ax2.set_xlabel("True Positive Rate", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Average Precision")
    ax2.fill_between(recall, precision, alpha=0.2)
    ax2.legend(loc="lower right")

    # Adjust layout
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plt.tight_layout()

    plt.show()


def calculate_clip_score(prompts: List[str], image_paths: List[Path]) -> float:
    """
    Calculate the CLIP score for a list of prompt and image pairs.
    CLIP score attempts to measure how well the image matches the prompt.
    It uses a CLIP model to embed the prompt and image, and then calculates their similarity.

    Args:
        prompts: List of prompts
        image_paths: List of image paths

    Returns:
        Average CLIP score for the prompt and image pairs
    """
    # Weights are downloaded automatically on the first run
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    skips = 0
    for prompt, image_path in tqdm(zip(prompts, image_paths), total=len(prompts)):
        if len(prompt) > 77:  # CLIP has a limit of 77 words
            skips += 1
            continue
        image = Image.open(image_path)
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        scores.append(float(outputs.logits_per_image))

    if skips > 0:
        print(f"Skipped {skips} prompts because they were too long")
    return sum(scores) / len(scores)


def calculate_clip_score_from_csv(csv_path: Path, image_dir: Path, column: str = "prompt", sep: str = ",") -> float:
    """
    Calculate the CLIP score for a dataset, where prompts are in a CSV and images in a directory.

    Args:
        csv_path: Path to a CSV file containing prompts
        image_dir: Path to a directory with images
        column: Name of the column containing the prompts
        sep: Separator for the CSV file

    Returns:
        Average CLIP score for the prompt and image pairs
    """
    df = pd.read_csv(csv_path, sep=sep)
    prompts = df[column].tolist()
    image_paths = list(image_dir.iterdir())
    return calculate_clip_score(prompts, image_paths)


def main():
    csv_dir = Path("csvs")
    csv_path1 = Path("csvs", "StyleGAN2.csv")
    csv_path2 = Path("csvs", "MSCOCO2014_filtered_val.csv")
    csv_path3 = Path("csvs", "SDR.csv")
    csv_path4 = Path("csvs", "VQGAN.csv")
    csv_paths = [csv_path1, csv_path2]

    detector_id = "CLIPDetector"

    # print(f"acc: {calculate_detector_accuracy(csv_paths, detector_id)}")
    # plot_auc_and_ap(csv_paths, detector_id)
    # print(f"aucroc: {calculate_detector_auc(csv_paths, detector_id)}")
    # print(f"ap: {calculate_detector_average_precision(csv_paths, detector_id)}")
    print_detector_accuracies(csv_dir, detector_id, csv_filter="rs224_bilinear")
    # print_dataset_accuracies(csv_path1)
    # print_dataset_accuracies(csv_path1, None, detector_id)
    # print_dataset_accuracies(csv_path2, None, detector_id)
    # th = calculate_balanced_threshold_from_roc(csv_paths, detector_id)
    # print_dataset_accuracies(csv_path1, th, detector_id)
    # print_dataset_accuracies(csv_path2, th, detector_id)

    datasets = {
        # "COCO": csv_dir.joinpath("MSCOCO2014_filtered_val.csv"),
        # "SDR": csv_dir.joinpath("SDR.csv"),
        "SDR": [csv_dir.joinpath("SDR_rs224_bilinear.csv"), csv_dir.joinpath("SDR_rs224_bicubic.csv")],
        # "BigGAN": csv_dir.joinpath("BigGAN.csv"),
        # "StyleGAN2": csv_dir.joinpath("StyleGAN2.csv"),
        # "VQGAN": csv_dir.joinpath("VQGAN.csv"),
        "VQGAN": [csv_dir.joinpath("VQGAN_rs224_bilinear.csv"), csv_dir.joinpath("VQGAN_rs224_bicubic.csv")],
        # "Craiyon": csv_dir.joinpath("Craiyon.csv"),
        # "SD2": csv_dir.joinpath("StableDiffusion2.csv"),
        # "DALL·E 2": csv_dir.joinpath("DALLE2.csv"),
        # "Midjourney": csv_dir.joinpath("Midjourney.csv"),
    }

    detectors = {
        "CNNDet\(_{0.1}\)": "CNNDetector_p0.1_crop",
        "CNNDet\(_{0.5}\)": "CNNDetector_p0.5_crop",
        "EnsembleDet": "EnsembleDetector",
        "CLIPDet": "CLIPDetector_crop",
    }
    # print_latex_accuracy_table(datasets, detectors)


if __name__ == "__main__":
    main()
