from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def calculate_detector_accuracy(csv_paths: List[Path], detector_id: str, sep: str = ",") -> float:
    """
    Calculate the total accuracy of a detector on given datasets. The csv files are assumed to contain
    a column with the detector's predictions.

    Args:
        csv_paths: List of paths to the dataset csv files
        detector_id: ID of the detector
        sep: Separator for the csv file

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
    Calculate the total AUC of a detector on given datasets. The csv files are assumed to contain
    a column with the detector's scores.

    Args:
        csv_paths: List of paths to the dataset csv files
        detector_id: ID of the detector
        sep: Separator for the csv file

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


def main():
    csv_paths = [Path("csvs", "MSCOCO2014_valsubset.csv"), Path("csvs", "StableDiffusion2.csv")]
    detector_id = "EnsembleDetector"
    print(f"acc: {calculate_detector_accuracy(csv_paths, detector_id)}")
    print(f"aucroc: {calculate_detector_auc(csv_paths, detector_id)}")


if __name__ == "__main__":
    main()
