import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from dataset import IMG_EXTENSIONS

np.random.seed(42)


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
    for index, row in tqdm(data.iterrows(), total=data.shape[0], unit="img"):
        # Get the image URL and prompt
        image_url = row['Attachments']
        prompt = row['clean_prompts']

        # Clean the prompt to create a valid file name
        file_name = clean_filename(prompt)

        if file_name is None:
            print(f"Skipping row {index}: invalid or missing prompt")
            continue

        # Get the file extension from the image URL
        file_extension = Path(image_url).suffix

        # Combine the folder name, file name, and extension to form the image file path
        image_file_path = dir_out.joinpath(f"{file_name}{file_extension}")

        # Download and save the image
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url)
                if response.status_code != 200:
                    print(f"Error downloading image at row {index}: {response.status_code}")
                    continue
                with open(image_file_path, 'wb') as f:
                    f.write(response.content)
                    break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Error downloading image at row {index} (attempt {attempt + 1}): {str(e)}")
                    time.sleep(3)  # Wait before retrying
                else:
                    print(f"Failed to download image at row {index} after {max_retries} attempts: {str(e)}")
                    continue


def clean_filename(file_name, max_length=100):
    # Remove illegal characters from file names
    if not isinstance(file_name, str):
        return None
    cleaned_name = re.sub(r'[\\/*?:"<>|]', "", file_name)
    return cleaned_name[:max_length]


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
    blocked_words = ["cartoon", "comic", "painting", "drawing", "animation", "sprite"]  # Prompt having ANY of these subwords is removed
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
            else:
                print("Validating data...")
                data_new = data.copy()
                rng = np.random.default_rng(42)
                running_idx = 0
                pbar = tqdm(total=desired_size, unit="rows")
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
                data = data_new
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


def main():
    midjourney_csv = Path("..", "data", "midjourney_v51_cleaned_data", "upscaled_prompts_df.csv")
    dir_out = Path("..", "data", "midjourney_v51_cleaned_data", "filtered_images")
    csv_out = midjourney_csv.parent.joinpath("filtered_prompts.csv")
    # test_midjourney_filters(midjourney_csv, csv_out)
    download_midjourney_data(csv_path=midjourney_csv, dir_out=dir_out,
                             apply_filtering=True, csv_out=csv_out, desired_size=1000, validate_data=True)


if __name__ == "__main__":
    main()
