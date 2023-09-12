import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
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
        # if count > 48:
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


def plot_spectra(data_dict: Dict[str, Path],
                 crop_size: Optional[Union[List[int], int]] = None,
                 apply_filter: bool = False,
                 jpeg_quality: Optional[int] = None,
                 show_example: bool = False,
                 same_intensity: bool = False,
                 fig_size: Tuple[float, float] = (10, 5)) -> None:
    """
    Plot average Fourier spectras of each directory side by side.
    Optionally, apply center-cropping and filter by subtracting median denoised versions of the images, and plot an
    example image on top of each spectrum.

    Args:
        data_dict: Dictionary containing the dataset name as key and the path to the directory as value
        crop_size: Size of the center-crop to apply on each dataset, List for different sizes, None for no cropping
        apply_filter: Whether to filter the images
        jpeg_quality: Optional JPEG quality to apply to the images
        show_example: Whether to show example images on top of the spectra
        same_intensity: Whether to normalize the spectra to the same colormap intensity range
        fig_size: Optional size of the figure (width, height)
    """
    # Deduce the number of plots and the crop sizes
    plot_count = len(data_dict)
    if isinstance(crop_size, List):
        if len(crop_size) == 1:
            crop_sizes = crop_size * plot_count
        else:
            assert plot_count == len(crop_size)
            crop_sizes = crop_size
    elif isinstance(crop_size, int):
        crop_sizes = [crop_size] * plot_count
    else:
        crop_sizes = [None] * plot_count

    # Configure the plot
    rows = 2 if show_example else 1
    fig, axs = plt.subplots(rows, plot_count, squeeze=False, figsize=fig_size)
    plt.setp(axs, xticks=[], yticks=[], xlabel="", ylabel="")

    # Obtain average spectra and example images
    avg_spectra = []
    example_images = []
    for i, data_dir in enumerate(data_dict.values()):
        avg_spectrum, img = get_mean_spectrum(data_dir, crop_size=crop_sizes[i],
                                              apply_filter=apply_filter, jpeg_quality=jpeg_quality)
        avg_spectra.append(avg_spectrum)
        example_images.append(img)

    # Choose the intensity range for all plots
    if same_intensity:
        max_spectrum = max([np.max(s) for s in avg_spectra])
        min_spectrum = min([np.min(s) for s in avg_spectra])
    else:
        max_spectrum = None
        min_spectrum = None

    # Plot
    for i, data_name in enumerate(data_dict.keys()):
        axs[0, i].set_title(data_name, fontsize=22)
        avg_spectrum = avg_spectra[i]
        if show_example:
            img = example_images[i]
            axs[0, i].imshow(img, cmap="gray")
            axs[1, i].imshow(avg_spectrum, cmap="turbo", vmin=min_spectrum, vmax=max_spectrum)
        else:
            axs[0, i].imshow(avg_spectrum, cmap="turbo", vmin=min_spectrum, vmax=max_spectrum)

    fig.tight_layout()
    plt.show()


def get_mean_spectrum(data_dir: Path,
                      crop_size: Optional[int] = None,
                      apply_filter: bool = False,
                      jpeg_quality: Optional[int] = None) -> (np.ndarray, np.ndarray):
    """
    Get the mean Fourier spectrum of all images in the given directory and an example preprocessed image.
    All images are preprocessed by grayscaling, and optionally center-cropping them to a square and
    filtering by subtracting median denoised versions.

    Args:
        data_dir: Directory containing the images
        crop_size: Optional square center-crop size in pixels, if None, no cropping is done
        apply_filter: Boolean indicating whether to filter the images
        jpeg_quality: Optional JPEG quality to use for compressing the images, if None, no compression is done

    Returns:
        The mean Fourier spectrum of all images in the directory and an example preprocessed image
    """
    all_ffts = []
    example_img = None
    for img_path in tqdm(list(data_dir.iterdir()), desc="Calculating FFTs", unit="img"):
        img = Image.open(img_path).convert("L")

        # JPEG compression
        if jpeg_quality is not None:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=jpeg_quality)
            img_bytes.seek(0)
            img = Image.open(img_bytes)

        # Center-cropping
        if isinstance(crop_size, int):
            w, h = img.size
            if w < crop_size or h < crop_size:
                print(f"Image {img_path} is too small to crop, skipping")
                continue
            img = img.crop((w // 2 - crop_size // 2, h // 2 - crop_size // 2, w // 2 + crop_size // 2, h // 2 + crop_size // 2))

        # High-pass filtering
        if apply_filter:
            img_med = img.filter(ImageFilter.MedianFilter(3))
            img = np.abs(np.array(img) - np.array(img_med))
        else:
            img = np.array(img)

        img_fft = fourier_transform_log(img)
        all_ffts.append(img_fft)
        if example_img is None:
            example_img = img

    if len(all_ffts) == 0:
        print("No valid images found")
        return None, None

    all_ffts_np = np.array(all_ffts)
    avg_fft = np.mean(all_ffts_np, axis=0)
    return avg_fft, example_img


def fourier_transform_log(img: np.ndarray) -> np.ndarray:
    """
    Calculate the fourier transform of the given image and take the logarithm of the absolute values.

    Args:
        img: Numpy array of the image to transform

    Returns:
        Numpy array of the transformed image with logarithm applied on absolute values
    """
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_fft = np.log(np.abs(img_fft) + np.finfo(float).eps)
    return img_fft


def main():
    coco_dir = Path("..", "data", "MSCOCO2014")
    coco_data_dir = coco_dir.joinpath("filtered_val")
    coco_json = coco_dir.joinpath("annotations", "captions_val2014.json")
    sd2_dir = Path("..", "data", "StableDiffusion2", "filtered_val2014_ts50")
    dalle2_dir = Path("..", "data", "DALLE2", "DMimageDetection")
    stylegan_dir = Path("..", "data", "StyleGAN2", "filtered_images")
    sdr_dir = Path("..", "data", "HDR", "filtered_images")
    vqgan_dir = Path("..", "data", "VQGAN", "filtered_images")

    # print_coco_stats(coco_json)
    # plot_resolutions(sdr_dir)
    data_dict = {
        "COCO": coco_data_dir,
        "SDR": sdr_dir,
        "StyleGAN2": stylegan_dir,
        "VQGAN": vqgan_dir,
        "DALL·E 2": dalle2_dir,
    }
    plot_spectra(data_dict, crop_size=[256], apply_filter=True, show_example=False,
                 same_intensity=True, fig_size=(18, 4.3), jpeg_quality=None)


if __name__ == "__main__":
    main()
