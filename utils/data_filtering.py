import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from dataset import IMG_EXTENSIONS
from tqdm import tqdm


def download_midjourney_data(csv_path: Path,
                             dir_out: Path,
                             apply_filtering: bool = True,
                             csv_out: Optional[Path] = None,
                             desired_size: Optional[int] = None) -> None:
    """
    Download the MidJourney v5.1 Cleaned Dataset with filters.
    Download the CSV from https://www.kaggle.com/datasets/iraklip/modjourney-v51-cleaned-data.
    Code adapted from https://www.kaggle.com/code/iraklip/downloading-midjourney-v5-1-images.

    Args:
        csv_path: Path to the MidJourney CSV file, or an already filtered one
        dir_out: Output directory for the images
        apply_filtering: Whether to filter the data
        csv_out: Path for a CSV file to save the filtered data
        desired_size: Desired number of images
    """
    dir_out.mkdir(parents=True, exist_ok=True)

    # Read the CSV file and filter
    data = pd.read_csv(csv_path, sep=",")
    if apply_filtering:
        data = filter_midjourney_data(data, csv_out, desired_size)

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


def test_midjourney_filters(midjourney_csv: Path, csv_out: Optional[Path] = None, desired_size: Optional[int] = None):
    """
    Test the output size of the current MidJourney filters, printing the original and final shape.

    Args:
        midjourney_csv: Path to the MidJourney CSV file
        csv_out: Path for a CSV file to save the filtered data
        desired_size: Desired number of samples (rows) in the output data
    """
    og_data = pd.read_csv(midjourney_csv, sep=",")
    filtered_data = filter_midjourney_data(og_data, csv_out=csv_out, desired_size=desired_size)
    print(f"Original data: {og_data.shape}")
    print(f"Filtered data: {filtered_data.shape} rows")


def filter_midjourney_data(data: pd.DataFrame,
                           csv_out: Optional[Path] = None,
                           desired_size: Optional[int] = None) -> pd.DataFrame:
    """
    Filter the MidJourney data to keep only the rows that pass the filters.
    A CSV file with the filtered data is saved if a path is supplied.
    The output size can be further reduced by specifying a desired size.

    Args:
        data: DataFrame with the MidJourney data
        csv_out: Path for a CSV file to save the filtered data
        desired_size: Desired number of samples in the output data

    Returns:
        DataFrame with filtered rows and a reduced number of columns
    """
    # Adjustable filters
    # allowed_versions = ["4", "4.0", "5", "5.0", "5.1"]
    allowed_versions = ["5.1"]
    required_words = ["photo"]  # Prompt having ANY of these subwords is kept
    blocked_words = ["cartoon", "painting"]  # Prompt having ANY of these subwords is removed
    allowed_ratios = ["1:1"]  # Multiples are discarded, such as 2:2 and 3:3

    data = data[["Unnamed: 0", "Attachments", "version", "aspect", "clean_prompts"]]
    data.rename(columns={"Unnamed: 0": "original_index"}, inplace=True)
    data_length = len(data)
    og_length = data_length
    print("Filtering MidJourney CSV data...")

    # Version filter
    data["version"] = data["version"].fillna("5.1").astype(str)
    data = data[data["version"].isin(allowed_versions)]
    print(f"Version filter removed {data_length - len(data)} rows")
    data_length = len(data)
    # MS Excel shows "5.1" as "5.01.00", similar irrelevant issue with aspect ratio

    # Aspect ratio filter
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
            # Todo: Perform HEAD requests to see that files exist and act accordingly
            data = data.sample(n=desired_size, random_state=42)
            print(f"Desired size filter removed {data_length - len(data)} rows")
            data_length = len(data)
        elif desired_size == data_length:
            print(f"Desired size filter removed 0 rows")
        elif desired_size > og_length:
            print(f"Desired size is larger than the original data size, no rows removed")
        else:
            print(f"Desired size is larger than the filtered data size, no rows removed, consider changing filters,")

    print(f"Final data size: {data_length} rows")

    if csv_out is not None:
        print(f"Saving filtered CSV data to {csv_out.absolute()}")
        data.to_csv(csv_out, index=False, sep=",")

    return data


def main():
    midjourney_csv = Path("..", "data", "midjourney_v51_cleaned_data", "upscaled_prompts_df.csv")
    dir_out = Path("..", "data", "midjourney_v51_cleaned_data", "filtered_images")
    csv_out = midjourney_csv.parent.joinpath("filtered_prompts.csv")
    # test_midjourney_filters(midjourney_csv, csv_out, 1000)
    download_midjourney_data(csv_path=midjourney_csv, dir_out=dir_out,
                             apply_filtering=True, csv_out=csv_out, desired_size=1000)


if __name__ == "__main__":
    main()
