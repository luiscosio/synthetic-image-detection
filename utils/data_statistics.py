import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def print_coco_stats(captions_json: Path) -> None:
    # Print and plot some statistics about the COCO dataset using the official captions json file
    with open(captions_json) as f:
        data = json.load(f)

    word_counts = []
    heights = []
    widths = []
    for el in data["annotations"]:
        count = len(el["caption"].split())
        word_counts.append(count)
        #if count > 48:
        #    print(el["caption"])

    for el in data["images"]:
        heights.append(el["height"])
        widths.append(el["width"])

    word_counts_np = np.array(word_counts)
    unique, counts = np.unique(word_counts_np, return_counts=True)
    heights_np = np.array(heights)
    widths_np = np.array(widths)
    print(f"avg word count: {word_counts_np.mean()} +- {word_counts_np.std()}")
    print(f"longest caption length {max(word_counts_np)}")
    print(f"avg image height: {heights_np.mean()} +- {heights_np.std()}")
    print(f"avg image width: {widths_np.mean()} +- {widths_np.std()}")
    plt.bar(unique, counts)
    plt.show()


def plot_resolutions(data_dir: Path) -> None:
    # Plot a histogram of the resolutions of the images in the given directory
    resolutions = []
    for img_path in tqdm(list(data_dir.iterdir())):
        img = plt.imread(img_path)
        imgs = img.shape
        resolutions.append(f"{imgs[0]}x{imgs[1]}")

    # plot a histogram of the resolutions
    resolutions_np = np.array(resolutions)
    unique, counts = np.unique(resolutions_np, return_counts=True)

    # lower figure font size on the x-axis
    plt.rcParams.update({'font.size': 6})
    # rotate the x-axis labels
    plt.xticks(rotation=90)

    plt.bar(unique, counts)
    plt.show()


def plot_spectra(data_dirs: List[Path], crop_sizes: Optional[List[int]] = None, show_example: bool = True) -> None:
    # Plot average spectras of each directory side by side, with example image on top of them
    if isinstance(crop_sizes, List):
        assert len(data_dirs) == len(crop_sizes)
    rows = 2 if show_example else 1
    fig, axs = plt.subplots(rows, len(data_dirs), squeeze=False)
    for i, data_dir in enumerate(data_dirs):
        avg_spectrum, img = get_mean_spectrum(data_dir, crop_size=crop_sizes[i])
        if show_example:
            axs[0, i].imshow(img, cmap="gray")
            axs[1, i].imshow(avg_spectrum, cmap="gray")
        else:
            axs[0, i].imshow(avg_spectrum, cmap="gray")

    fig.tight_layout()
    plt.show()


def get_mean_spectrum(data_dir: Path, crop_size: Optional[int] = None) -> (np.ndarray, np.ndarray):
    # Get the mean spectrum of all images in the given directory and optionally crop them to a square
    all_ffts = []
    example_img = None
    for img_path in tqdm(list(data_dir.iterdir()), desc="Calculating FFTs", unit="img"):
        img = Image.open(img_path).convert("L")

        if isinstance(crop_size, int):
            # Center crop the image
            w, h = img.size
            if w < crop_size or h < crop_size:
                print(f"Image {img_path} is too small to crop, skipping")
                continue
            img = img.crop((w // 2 - crop_size // 2, h // 2 - crop_size // 2, w // 2 + crop_size // 2, h // 2 + crop_size // 2))

        img = np.array(img)
        img_fft = fourier_transform(img, plot=False)
        all_ffts.append(img_fft)
        if example_img is None:
            example_img = img

    if len(all_ffts) == 0:
        print("No valid images found")
        return None, None

    all_ffts_np = np.array(all_ffts)
    avg_fft = np.mean(all_ffts_np, axis=0)
    return avg_fft, example_img


def fourier_transform(img: np.ndarray, plot: bool = False) -> np.ndarray:
    # Calculate the fourier transform of the given image and optionally plot it, return log of abs
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_fft = np.log(np.abs(img_fft) + np.finfo(float).eps)
    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img_fft, cmap="gray")
        plt.show()
    return img_fft


def main():
    coco_dir = Path("..", "data", "MSCOCO2014")
    coco_data_dir = coco_dir.joinpath("filtered_val")
    coco_json = coco_dir.joinpath("annotations", "captions_val2014.json")
    mid_dir = Path("..", "data", "midjourney_v51_cleaned_data", "filtered_images")
    stylegan_dir = Path("..", "data", "StyleGAN2", "filtered_images")

    # print_coco_stats(coco_json)
    # plot_resolutions(mid_dir)
    plot_spectra([stylegan_dir, coco_data_dir], crop_sizes=[256, 256], show_example=True)


if __name__ == "__main__":
    main()
