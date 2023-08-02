from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
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
    The value can refer to a predicdted label or a float output by the detector.

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
    Calculate the total accuracy of a detector on given datasets. The CSV files are assumed to contain
    a column with the detector's predictions.

    Args:
        csv_paths: List of paths to the dataset CSV files
        detector_id: ID of the detector
        sep: Separator for the CSV file

    Returns:
        Total accuracy of the detector on the datasets
    """
    preds, gt = get_detector_values_and_truths(csv_paths, detector_id, "labels", sep=sep)
    return accuracy_score(gt, preds)


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
    csv_path = Path("csvs", "PartiPrompts.csv")
    image_dir = Path("data", "StableDiffusion2", "PartiPrompts")
    # score = calculate_clip_score_from_csv(csv_path, image_dir, column="Prompt", sep=";")
    # print(score)

    csv_paths = [Path("csvs", "MSCOCO2014_filtered_val.csv"), Path("csvs", "StyleGAN2.csv")]
    csv_paths = [Path("csvs", "StyleGAN2r.csv"), Path("csvs", "StyleGAN2f.csv")]
    # detector_id = "EnsembleDetector"
    detector_id = "CNNDetector"
    # print(f"acc: {calculate_detector_accuracy(csv_paths, detector_id)}")
    plot_auc_and_ap(csv_paths, detector_id)
    print(f"aucroc: {calculate_detector_auc(csv_paths, detector_id)}")
    print(f"ap: {calculate_detector_average_precision(csv_paths, detector_id)}")


if __name__ == "__main__":
    main()
