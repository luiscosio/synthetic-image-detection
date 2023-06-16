from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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
    gt = []
    preds = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep=sep)
        gt.extend(df["label"].tolist())
        preds.extend(df[f"{detector_id}_labels"].tolist())

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
    gt = []
    scores = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep=sep)
        gt.extend(df["label"].tolist())
        scores.extend(df[f"{detector_id}_scores"].tolist())

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
    gt = []
    scores = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep=sep)
        gt.extend(df["label"].tolist())
        scores.extend(df[f"{detector_id}_scores"].tolist())

    return average_precision_score(gt, scores)


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
    score = calculate_clip_score_from_csv(csv_path, image_dir, column="Prompt", sep=";")
    print(score)

    # csv_paths = [Path("csvs", "MSCOCO2014_valsubset.csv"), Path("csvs", "StableDiffusion2.csv")]
    # detector_id = "EnsembleDetector"
    # print(f"acc: {calculate_detector_accuracy(csv_paths, detector_id)}")
    # print(f"aucroc: {calculate_detector_auc(csv_paths, detector_id)}")
    # print(f"ap: {calculate_detector_average_precision(csv_paths, detector_id)}")


if __name__ == "__main__":
    main()
