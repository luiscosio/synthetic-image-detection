import csv
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from dataset import IMG_EXTENSIONS

np.random.seed(42)
random.seed(42)


def download_midjourney_data(csv_path: Path,
                             dir_out: Path,
                             apply_filtering: bool = True,
                             csv_out: Optional[Path] = None,
                             desired_size: Optional[int] = None,
                             validate_data: bool = True) -> None:
    """
    Download the Midjourney v5.1 Cleaned Dataset with filters.
    Download the CSV file from https://www.kaggle.com/datasets/iraklip/modjourney-v51-cleaned-data.
    Code adapted from https://www.kaggle.com/code/iraklip/downloading-midjourney-v5-1-images.

    The CSV file contains metadata used to download the images from the internet.
    A filtered CSV is optionally created from the original one before downloading.

    Args:
        csv_path: Path to the Midjourney CSV file, or an already filtered one
        dir_out: Output directory for the images
        apply_filtering: Whether to filter the data
        csv_out: Path for a CSV file to save the filtered data
        desired_size: Desired number of images
        validate_data: Whether to validate the data, only used if apply_filtering and desired_size are not None
    """
    dir_out.mkdir(parents=True, exist_ok=True)

    # Read the CSV file and filter
    data = pd.read_csv(csv_path, sep=",")
    if apply_filtering:
        data = filter_midjourney_data(data, csv_out, desired_size, validate_data)

    # Loop through the rows of the DataFrame
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Downloading images", unit="img"):
        # Get the image URL and prompt
        image_url = row["Attachments"]
        og_idx = row["original_index"]

        # Get the file extension from the image URL
        file_extension = Path(image_url).suffix

        image_file_path = dir_out.joinpath(f"{og_idx}{file_extension}")
        if image_file_path.exists():
            continue

        # Download and save the image
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url)
                if response.status_code != 200:
                    print(f"Error downloading image at row {index} (idx {og_idx}): {response.status_code}")
                    continue
                with open(image_file_path, 'wb') as f:
                    f.write(response.content)
                    break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Error downloading image at row {index} (idx {og_idx}) (attempt {attempt + 1}): {str(e)}")
                    time.sleep(3)  # Wait before retrying
                else:
                    print(f"Failed to download image at row {index} (idx {og_idx}) after {max_retries} attempts: {str(e)}")
                    continue


def test_midjourney_filters(midjourney_csv: Path,
                            csv_out: Optional[Path] = None,
                            desired_size: Optional[int] = None,
                            validate_data: bool = True) -> None:
    """
    Test the output size of the current Midjourney filters, printing the original and final shape.

    Args:
        midjourney_csv: Path to the Midjourney CSV file
        csv_out: Path for a CSV file to save the filtered data
        desired_size: Desired number of samples (rows) in the output data
        validate_data: Whether to validate the data during optional size filtering
    """
    og_data = pd.read_csv(midjourney_csv, sep=",")
    filtered_data = filter_midjourney_data(og_data, csv_out=csv_out,
                                           desired_size=desired_size, validate_data=validate_data)
    print(f"Original data: {og_data.shape}")
    print(f"Filtered data: {filtered_data.shape} rows")


def filter_midjourney_data(data: pd.DataFrame,
                           csv_out: Optional[Path] = None,
                           desired_size: Optional[int] = None,
                           validate_data: bool = True) -> pd.DataFrame:
    """
    Filter the Midjourney data to keep only the rows that pass the filters.
    A CSV file with the filtered data is saved if a path is supplied.
    The output size can be further reduced by specifying a desired size.

    Args:
        data: DataFrame with the Midjourney data
        csv_out: Path for a CSV file to save the filtered data
        desired_size: Desired number of samples in the output data
        validate_data: Whether to validate the data during optional size filtering

    Returns:
        DataFrame with filtered rows and a reduced number of columns
    """
    # Adjustable filters
    # allowed_versions = ["4", "4.0", "5", "5.0", "5.1"]
    allowed_versions = ["5.1"]
    required_words = ["photo"]  # Prompt having ANY of these subwords is kept (unless blocked)
    # Prompt having ANY of these subwords is removed
    blocked_words = ["cartoon", "comic", "painting", "drawing", "animation", "sprite", "drawn", "sketch", "anime"]
    allowed_ratios = ["1:1"]  # Multiples are discarded, such as 2:2 and 3:3
    max_characters = 1600  # Prompts + parameters with more characters are removed

    data = data[["Unnamed: 0", "Attachments", "version", "aspect", "clean_prompts", "Content"]]
    data.rename(columns={"Unnamed: 0": "original_index"}, inplace=True)
    data_length = len(data)
    og_length = data_length
    print("Filtering Midjourney CSV data...")
    print(f"Original data: {data_length} rows")

    # Character filter
    # Too long prompts are truncated with "..." with no access to parameters, causing false assumptions
    data = data[data["Content"].str.len() <= max_characters]
    print(f"Character filter removed {data_length - len(data)} rows")
    data_length = len(data)

    # Version filter
    # Default version is 5.1
    data["version"] = data["version"].fillna("5.1").astype(str)
    data = data[data["version"].isin(allowed_versions)]
    print(f"Version filter removed {data_length - len(data)} rows")
    data_length = len(data)

    # Aspect ratio filter
    # In the original CSV, only --ar has been considered, not --aspect
    # Replace an empty aspect ratio with a possible found one
    reg = r"\s--aspect\s(\d+:\d+)(?:\s|\*\*)"
    data.loc[data["aspect"].isnull(), "aspect"] = data.loc[data["aspect"].isnull(), "Content"].str.extract(reg, expand=False)
    # Default aspect ratio is 1:1
    data["aspect"] = data["aspect"].fillna("1:1").astype(str)
    data = data[data["aspect"].isin(allowed_ratios)]
    print(f"Aspect ratio filter removed {data_length - len(data)} rows")
    data_length = len(data)

    # Prompt filter
    data["clean_prompts"] = data["clean_prompts"].fillna("None").astype(str)
    data = data[data["clean_prompts"].str.contains("|".join(required_words), case=False)]
    data = data[~data["clean_prompts"].str.contains("|".join(blocked_words), case=False)]
    print(f"Prompt filter removed {data_length - len(data)} rows")
    data_length = len(data)

    # Image extensions filter, 0 removed in early testing
    data = data[data["Attachments"].str.contains("|".join(IMG_EXTENSIONS), case=False)]
    print(f"Extension filter removed {data_length - len(data)} rows")
    data_length = len(data)

    # Desired size filter
    if isinstance(desired_size, int):
        if desired_size < data_length:
            if not validate_data:
                data = data.sample(n=desired_size, random_state=42)
                data.sort_values(by="original_index", inplace=True)  # Sort by original index
            else:
                data_new = data.copy()
                rng = np.random.default_rng(42)
                running_idx = 0
                pbar = tqdm(total=desired_size, desc="Validating data", unit="rows")
                for idx in rng.choice(data_length, data_length, replace=False):
                    row = data.iloc[idx]
                    image_url = row["Attachments"]
                    response = requests.head(image_url)
                    if response.status_code == 200:  # Check that the image can be downloaded
                        data_new.iloc[running_idx] = row
                        running_idx += 1
                        pbar.update(1)
                        if running_idx == desired_size:
                            break
                pbar.close()
                data_new = data_new.iloc[:running_idx]  # Remove unused rows from the copy
                data = data_new.sort_values(by="original_index", inplace=False)  # Sort by original index
                if len(data) < desired_size:
                    print(f"Warning: not enough valid data {len(data)} to reach desired size, consider changing filters")

            print(f"Desired size filter removed {data_length - len(data)} rows")
            data_length = len(data)

        elif desired_size == data_length:
            print(f"Desired size filter removed 0 rows")
        elif desired_size > og_length:
            print(f"Desired size is larger than the original data size, no rows removed")
        else:
            print(f"Desired size is larger than the filtered data size, no rows removed, consider changing filters")

    print(f"Final data size: {data_length} rows")

    if csv_out is not None:
        print(f"Saving filtered CSV data to {csv_out.absolute()}")
        data.to_csv(csv_out, index=False, sep=",")

    return data


def create_coco_subset(captions_json: Path,
                       dir_in: Path,
                       dir_out: Path,
                       csv_out: Path,
                       desired_size: int) -> None:
    """
    Create a subset of the COCO (2014 validation) dataset using the official captions JSON file.
    Filters and desired size are used to reduce the size of dataset.

    A CSV file is created with the filtered data, each image matched with one caption.
    The images are copied to a new directory.
    The images and the JSON need to be downloaded beforehand from https://cocodataset.org/#download,
    (http://images.cocodataset.org/zips/val2014.zip and
    http://images.cocodataset.org/annotations/annotations_trainval2014.zip).

    Args:
        captions_json: Path to the official COCO captions JSON file
        dir_in: Directory where the images are located
        dir_out: Directory where the images will be copied
        csv_out: Path to the CSV file where the filtered data will be saved
        desired_size: The desired size of the dataset
    """
    dir_out.mkdir(parents=True, exist_ok=True)

    with open(captions_json) as f:
        data = json.load(f)

    # Change the image list to a dictionary with the image id as key
    image_list = data["images"]
    images = {image["id"]: image for image in image_list}

    # Create a dict with image id as a key and each matching annotation in a list as value
    annotations = data["annotations"]
    image_annotations = defaultdict(list)
    for annotation in annotations:
        image_annotations[annotation["image_id"]].append(annotation)

    data_rows = filter_coco_data(images, image_annotations, dir_in, desired_size)

    # Write the data to a CSV file
    csv_fields = ["image_id", "filename", "height", "width", "caption_id", "caption"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(csv_fields)
        for row in tqdm(data_rows, desc="Writing CSV", unit="row"):
            writer.writerows(row)

    # Copy images to the new directory
    for row in tqdm(data_rows[0], desc="Copying images", unit="img"):
        image_name = row[1]
        image_path = dir_in.joinpath(image_name)
        shutil.copy(image_path, dir_out.joinpath(image_name))


def filter_coco_data(images: Dict[int, Dict[str, Union[int, str]]],
                     annotations: Dict[int, List[Dict[str, Union[int, str]]]],
                     dir_in: Path,
                     desired_size: int) -> List[List[Union[int, str]]]:
    """
    Filter the COCO dataset to match the desired size and filters.
    The images are filtered to be square, above a minimum size, have at least one caption and not be grayscale.

    Args:
        images: Dictionary with the image id as key and the image data as value
        annotations: Dictionary with the image id as key and a list of annotations as value
        dir_in: Directory where the images are located
        desired_size: The desired size of the dataset

    Returns:
        Nested list with each row containing the image id, filename, height, width, caption id and caption
    """
    min_size = 300
    data_rows = [[]]
    og_length = len(images)
    for idx, values in tqdm(images.items(), desc="Filtering images", unit="img"):
        height = values["height"]
        width = values["width"]

        # Above minimum size and square
        if values["height"] < min_size or values["width"] < min_size or height != width:
            continue

        # At least one caption
        if len(annotations[idx]) < 1:
            continue

        # Not grayscale (1 channel nor 3 channel)
        img = Image.open(dir_in.joinpath(values["file_name"]))
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) < 3:
            continue
        if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 0], img[:, :, 2]):
            continue

        # Choose the longest text caption in a list of dictionaries under "caption" key, longest in COCO is ~50 words
        _, cap_id, caption = max(annotations[idx], key=lambda d: len(d["caption"])).values()

        data_rows[0].append([idx, values["file_name"], height, width, cap_id, caption])

    # Desired size filter
    data_length = len(data_rows[0])
    if isinstance(desired_size, int):
        if desired_size < data_length:
            data_rows[0] = random.sample(data_rows[0], desired_size)
            data_rows[0].sort(key=lambda x: x[0])  # Sort by image id
        elif desired_size > og_length:
            print(f"Desired size is larger than the original data size")
        else:
            print(f"Desired size is larger than the filtered data size, consider changing filters")

    print(f"Final data size: {len(data_rows[0])} rows")

    return data_rows


def create_subset_from_structure(data_dir: Path, dir_out: Path, desired_size: int) -> None:
    """
    Create a subset of a dataset with the following folder structure:
    Dataset
    ├── class_1
    │   ├── real (not used)
    │   ├── fake
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   │   └── ...
    ├── class_2
    │   ├── ...
    ├── ...

    Images are copied to the new directory without keeping the original folder structure.

    Args:
        data_dir: Path to the dataset with the specified structure
        dir_out: Path to the directory where the subset will be created
        desired_size: The desired size of the dataset
    """
    class_dirs = list(data_dir.glob("*/*fake*"))
    if len(class_dirs) == 0:
        print("No fake image folders found, check the folder structure")
        return
    class_sizes = [len(list(class_dir.glob("*"))) for class_dir in class_dirs]
    balanced_size, remainder = divmod(desired_size, len(class_dirs))
    small_classes = np.array(class_sizes) < balanced_size
    image_paths = []
    dir_out.mkdir(parents=True, exist_ok=True)

    if sum(class_sizes) <= desired_size:
        print(f"Desired size is larger than the original size of the dataset, copying the whole dataset")
        for class_dir in class_dirs:
            image_paths.extend(list(class_dir.glob("*")))

    elif small_classes.any():
        print(f"The subset will be unbalanced (use desired_size {min(class_sizes)*len(class_dirs)} for a balanced set)")

        def add_images_recursively(remaining_class_dirs: List[Path],
                                   remaining_class_sizes: List[int],
                                   current_best_size: int,
                                   needed_images: int) -> None:
            """
            Recursively add images to the parent function's list, going from smallest to largest class.
            The bigger the class, the more images it will have, while keeping classes as balanced as possible.

            Args:
                remaining_class_dirs: Class directories that have not been added to the subset
                remaining_class_sizes: Sizes of the remaining classes
                current_best_size: Current optimal size for every remaining class
                needed_images: Number of images left to add to the subset
            """
            smallest_pos = np.argmin(remaining_class_sizes)
            class_dir = remaining_class_dirs[smallest_pos]
            class_size = remaining_class_sizes[smallest_pos]

            if class_size < current_best_size:
                image_paths.extend(list(class_dir.glob("*")))
                needed_images -= class_size

            else:
                image_paths.extend(random.sample(list(class_dir.glob("*")), current_best_size))
                needed_images -= current_best_size

            remaining_class_dirs.pop(smallest_pos)
            remaining_class_sizes.pop(smallest_pos)
            if len(remaining_class_dirs) == 0:
                return
            current_best_size = needed_images // len(remaining_class_dirs)  # Biggest class has no remainder
            add_images_recursively(remaining_class_dirs, remaining_class_sizes, current_best_size, needed_images)

        add_images_recursively(class_dirs, class_sizes, balanced_size, desired_size)

    elif remainder != 0:
        print(f"Desired size is not perfectly divisible by the number of classes, creating a slightly unbalanced subset")
        counter = 0
        for class_dir, class_size in zip(class_dirs, class_sizes):
            if counter != remainder and class_size > balanced_size:
                addition = 1
                counter += 1
            else:
                addition = 0
            image_paths.extend(random.sample(list(class_dir.glob("*")), balanced_size+addition))

    else:
        print(f"Creating a balanced subset of size {desired_size}")
        for class_dir in class_dirs:
            image_paths.extend(random.sample(list(class_dir.glob("*")), balanced_size))

    print(f"Final data size: {len(image_paths)} images")

    # Copy images to the new directory
    zfill_size = len(str(len(image_paths)))
    for idx, image_path in tqdm(enumerate(image_paths), desc="Copying images", unit="img"):
        name = str(idx).zfill(zfill_size) + image_path.suffix
        shutil.copy(image_path, dir_out.joinpath(name))


def main():
    desired_size = 1000
    midjourney_csv = Path("..", "data", "midjourney_v51_cleaned_data", "upscaled_prompts_df.csv")
    dir_out = Path("..", "data", "midjourney_v51_cleaned_data", "filtered_images")
    csv_out = midjourney_csv.parent.joinpath("filtered_prompts.csv")
    # test_midjourney_filters(midjourney_csv, csv_out)
    #download_midjourney_data(csv_path=midjourney_csv, dir_out=dir_out,
    #                         apply_filtering=True, csv_out=csv_out, desired_size=desired_size, validate_data=True)

    coco_dir = Path("..", "data", "MSCOCO2014")
    coco_data_dir = coco_dir.joinpath("val2014")
    coco_json = coco_dir.joinpath("annotations", "captions_val2014.json")
    coco_data_out = coco_dir.joinpath("filtered_val2014")
    coco_csv_out = coco_dir.joinpath("filtered_val2014.csv")
    # create_coco_subset(coco_json, coco_data_dir, coco_data_out, coco_csv_out, desired_size=desired_size)

    stylegan_dir = Path("..", "data", "easy_to_spot_dataset", "stylegan2")
    stylegan_data_out = Path("..", "data", "StyleGAN2", "filtered_images")
    create_subset_from_structure(stylegan_dir, stylegan_data_out, desired_size=desired_size)


if __name__ == "__main__":
    main()
