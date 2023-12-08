import argparse
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def get_detector_values_and_truths(
    csv_paths: List[Path],
    detector_id: str,
    value_type: Literal["labels", "scores"],
    sep: str = ",",
) -> (List[float], List[int]):
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


def calculate_detector_accuracies(
    csv_path: Union[Path, List[Path]],
    detector_id: str,
    threshold: Optional[float] = None,
    csv_filter: Optional[str] = None,
    sep: Union[str, List[str]] = ",",
) -> Dict[str, float]:
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
        values = (
            df[detector_id + "_labels"]
            if threshold is None
            else df[detector_id + "_scores"]
        )
        values = np.array(values.to_list())
        if threshold is not None:
            values = values >= threshold
        accuracy = np.sum(values == labels) / len(values)
        accuracies[path.name] = accuracy

    return accuracies


def calculate_dataset_accuracies(
    csv_path: Path,
    threshold: Optional[float] = None,
    specific_detector: Optional[str] = None,
    sep: str = ",",
) -> Dict[str, float]:
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
            print(
                f"{specific_detector} not found in {csv_path.name} dataset, no accuracies calculated"
            )
            return dict()

    labels = np.array(df["label"].to_list())

    for detector in detectors:
        values = (
            df[detector + "_labels"] if threshold is None else df[detector + "_scores"]
        )
        values = np.array(values.to_list())
        if threshold is not None:
            values = values >= threshold
        accuracy = np.sum(values == labels) / len(values)
        accuracies[detector] = accuracy
    return accuracies


def print_dataset_accuracies(
    csv_path: Path,
    threshold: Optional[float] = None,
    specific_detector: Optional[str] = None,
    sep: str = ",",
) -> None:
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
    accuracies = calculate_dataset_accuracies(
        csv_path, threshold, specific_detector, sep
    )
    postfix = (
        "using the _labels column"
        if threshold is None
        else f"with threshold {threshold}"
    )
    print(f"\nAccuracies on the {csv_path.name} dataset {postfix}")
    for k, v in accuracies.items():
        print(f"{k}: {v}")


def print_detector_accuracies(
    csv_path: Union[Path, List[Path]],
    detector_id: str,
    threshold: Optional[float] = None,
    csv_filter: Optional[str] = None,
    sep: Union[str, List[str]] = ",",
) -> None:
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
        accuracies = calculate_detector_accuracies(
            csv_path, detector_id, threshold, csv_filter, sep
        )
    except ValueError as e:
        print(e)
        return

    if not accuracies:
        print(f"No results found for {detector_id} on the given datasets")
        return

    postfix = (
        "using the _labels column"
        if threshold is None
        else f"with threshold {threshold}"
    )
    postfix += f' and "{csv_filter}" CSV filter' if csv_filter else ""
    print(f"\nAccuracies for {detector_id} {postfix}")
    for k, v in accuracies.items():
        print(f"{k}: {v}")


def print_latex_accuracy_table(
    datasets: Dict[str, Path],
    detectors: Dict[str, str],
    thresholds: Optional[List[float]] = None,
    variations: Optional[List[str]] = None,
    sep: str = ",",
) -> None:
    """
    Print a LaTeX table contents with the accuracies of given detectors on given datasets.
    Each given dataset is expected to have results for each given detector.

    Multiple variations of datasets can be given through the variations argument, which adds the corresponding
    postfix(es) to the dataset Paths. Given variations, even if only one, the original dataset results are not included,
    unless one of the variations is an empty string.
    The variations in the table are in the order they are input, their distinct variation property is not shown.
    Each given dataset should have the same variations.

    Args:
        datasets: Dictionary of printed dataset names as keys and Paths as values
        detectors: Dictionary of printed detector names as keys and IDs as values
        thresholds: Optional thresholds for accuracy, in the same order as the matching detectors
        variations: Optional list of variations, matching the augmentations in CSV filenames
        sep: Separator for the CSV files, expected to be the same for all
    """
    if not thresholds:
        thresholds = [None] * len(detectors)
        th_str = ""
    else:
        th_str = f"\nThreshold & {' & '.join([f'{{{str(th)}}}' for th in thresholds])} \\\\\n\hline"

    # Check how many columns for each detector and apply variations
    multicolumn = 1
    datasets_varied = {}
    if isinstance(variations, list) and variations:
        multicolumn = len(variations)
        for k, v in datasets.items():
            paths = []
            for variation in variations:
                path_variation = v.with_name(f"{v.stem}{variation}{v.suffix}")
                paths.append(path_variation)
            datasets_varied[k] = paths
    else:
        datasets_varied = datasets

    # Set the header row
    table = "\hline\n"
    for detector_name in detectors.keys():
        if multicolumn > 1:
            table += f" & \multicolumn{{{multicolumn}}}{{c|}}{{{detector_name}}}"
        else:
            table += f" & {{{detector_name}}}"

    table += "\\\\\n\hline"
    table += th_str

    # Set a dataset's accuracies on one row, with each detector making up a column for each dataset variation
    for data_name, data_paths in datasets_varied.items():
        table += f"\n{data_name.ljust(10)} "
        for idx, (detector_name, detector_id) in enumerate(detectors.items()):
            if isinstance(data_paths, Path):
                data_paths = [data_paths]
            for data_path in data_paths:
                accs = calculate_dataset_accuracies(
                    data_path, thresholds[idx], detector_id, sep
                )
                table += f"& {accs[detector_id]}".ljust(10)
        table += "\\\\"

    print(table)


def calculate_balanced_threshold_from_roc(
    csv_paths: List[Path], detector_id: str, sep: str = ","
) -> float:
    """
    Calculate a threshold for accuracy from a receiver operating characteristics curve.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file:

    Returns:
        Balanced threshold from the ROC curve
    """
    scores, gt = get_detector_values_and_truths(
        csv_paths, detector_id, "scores", sep=sep
    )
    fpr, tpr, thresholds = roc_curve(gt, scores)
    return thresholds[np.argmax(tpr - fpr)]


def get_balanced_thresholds(
    csv_paths: List[Path], detectors: List[str], verbose: bool = False
):
    """
    Get the balanced thresholds for each detector on the given CSV datasets.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detectors: List of detector IDs
        verbose: Whether to print the thresholds

    Returns:
        List of thresholds in the same order as detectors
    """
    thresholds = []
    for detector_id in detectors:
        th = calculate_balanced_threshold_from_roc(csv_paths, detector_id)
        thresholds.append(th)
        if verbose:
            print(f"{detector_id}: {th}")

    return thresholds


def calculate_detector_auc(
    csv_paths: List[Path], detector_id: str, sep: str = ","
) -> float:
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
    scores, gt = get_detector_values_and_truths(
        csv_paths, detector_id, "scores", sep=sep
    )
    return roc_auc_score(gt, scores)


def calculate_detector_average_precision(
    csv_paths: List[Path], detector_id: str, sep: str = ","
) -> float:
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
    scores, gt = get_detector_values_and_truths(
        csv_paths, detector_id, "scores", sep=sep
    )
    return average_precision_score(gt, scores)


def plot_auc_and_ap(
    csv_paths: List[Path], detector_id: str, sep: str = ",", **kwargs
) -> None:
    """
    Plot the ROC and RP curves and their areas under the curves for a detector on given datasets.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file
    """
    scores, gt = get_detector_values_and_truths(
        csv_paths, detector_id, "scores", sep=sep
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6))
    plt.rcParams.update({"font.size": 12})

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


def calculate_clip_score_from_csv(
    csv_path: Path, image_dir: Path, column: str = "prompt", sep: str = ","
) -> float:
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


def print_score_differences(
    csv_path: Path, detector_1: str, detector_2: str, sep: str = ","
) -> None:
    """
    Print statistics for the difference in scores between two different detectors for the same CSV dataset.

    Args:
        csv_path: Path to a dataset CSV file
        detector_1: ID of a detector
        detector_2: ID of a detector
        sep: Separator for the CSV file
    """
    df = pd.read_csv(csv_path, sep=sep)
    scores_1 = np.array(df[f"{detector_1}_scores"].to_list())
    scores_2 = np.array(df[f"{detector_2}_scores"].to_list())
    diff = np.abs(scores_1 - scores_2)
    print(
        f"Differences between {detector_1} and {detector_2} scores for {csv_path.name}"
        f"\nmean: {np.mean(diff)}\nstd: {np.std(diff)}\nmax: {np.max(diff)}"
    )


def start_aucap(**kwargs):
    plot_auc_and_ap(**kwargs)


def start_acc(
    csv_paths,
    detector_ids,
    thresholds,
    balanced_paths,
    variations,
    csv_filter,
    sep,
    **kwargs,
):
    if len(csv_paths) == 1 and csv_paths[0].is_dir():
        csv_paths = list(csv_paths[0].glob("*.csv"))

    if variations:
        csv_paths2 = []
        for variation in variations:
            for c in csv_paths:
                path_variation = c.with_name(f"{c.stem}{variation}{c.suffix}")
                csv_paths2.append(path_variation)

        if csv_paths2:
            csv_paths = csv_paths2

    if csv_filter:
        csv_paths = [path for path in csv_paths if csv_filter in path.name]
    if not csv_paths:
        raise ValueError("Did not find any CSV files matching the inputs")

    if not detector_ids:
        for csv_path in csv_paths:
            print_dataset_accuracies(
                csv_path=csv_path, threshold=None, specific_detector=None, sep=sep
            )

    else:
        if balanced_paths:
            thresholds = get_balanced_thresholds(
                balanced_paths, detector_ids, verbose=False
            )
        if thresholds is None:
            thresholds = [None] * len(detector_ids)
        elif len(thresholds) == 1:
            thresholds = thresholds * len(detector_ids)

        for detector_id, th in zip(detector_ids, thresholds):
            print_detector_accuracies(
                csv_path=csv_paths,
                detector_id=detector_id,
                threshold=th,
                csv_filter=None,
                sep=sep,
            )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        description="",
    )
    subparsers = parser.add_subparsers(title="subcommands")
    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument(
        "--separator",
        "-s",
        type=str,
        default=",",
        nargs="+",
        dest="sep",
        help="Separator for the CSV file(s). If only one is given, each CSV file is "
        "expected to use the same separator. Otherwise, the number of separators should "
        "match the number of given CSV files.",
    )

    # Parser for plotting AUC and AP
    parser_aucap = subparsers.add_parser(
        "aucap",
        parents=[common_parser],
        help="Plot the AUC and AP of a detector for multiple datasets. "
        "Both synthetic and real image labels need to be present in the "
        "final input data.",
    )
    parser_aucap.set_defaults(func=start_aucap)

    parser_aucap.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        dest="csv_paths",
        help="Paths to CSV result files. Both synthetic and real image labels need to be "
        "present",
    )

    parser_aucap.add_argument(
        "--detector",
        "-d",
        type=str,
        required=True,
        dest="detector_id",
        help="ID of the evaluate detector",
    )

    # Parser for printing accuracies
    parser_acc = subparsers.add_parser(
        "acc",
        parents=[common_parser],
        help="Print accuracies depending on the input arguments."
        "If no detector IDs are given, all the detectors found in "
        "the given files are used."
        "The input path(s) for the results can be a list of files, or "
        "their shared directory.",
    )
    parser_acc.set_defaults(func=start_acc)

    parser_acc.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        dest="csv_paths",
        help="Path(s) to CSV result file(s) or a directory containing them",
    )

    parser_acc.add_argument(
        "--detectors",
        "-d",
        type=str,
        nargs="*",
        dest="detector_ids",
        help="IDs of the evaluatable detectors."
        "If none are given, all the ones found in the given files are used",
    )

    parser_acc.add_argument(
        "--thresholds",
        "-th",
        type=float,
        nargs="*",
        dest="thresholds",
        help="Custom decision thresholds for each given detector."
        "If not given, the label column is used (usually th of 0.5) instead of the score "
        "column. If only one is given, it is used for all the detectors."
        "Ignored if balanced-paths argument is given or detector IDs are not given",
    )

    parser_acc.add_argument(
        "--balanced-paths",
        "-bp",
        type=Path,
        nargs="*",
        dest="balanced_paths",
        help="Use dataset(s) with both real and synthetic images to calculate balanced "
        "thresholds for each detector by using the ROC curve. Each given detector "
        "needs to have results for the datasets given in this argument.",
    )

    parser_acc.add_argument(
        "--variations",
        "-v",
        type=str,
        nargs="*",
        dest="variations",
        help="Postfix(es) for the input result files. "
        "Giving variations: compression60 compression90 "
        "results in each input file having the form filename_compression60.csv "
        "and filename_compression60.csv, for which the evaluations are then performed. "
        "Used to reduce the amount of repetitive writing for data augmented results",
    )

    parser_acc.add_argument(
        "--csv_filter",
        "-cf",
        type=str,
        dest="csv_filter",
        help="Only the CSV files with the given pattern in their names are included",
    )

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == "__main__":
    main()
