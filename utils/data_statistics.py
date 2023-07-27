import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def main():
    coco_dir = Path("..", "data", "MSCOCO2014")
    coco_data_dir = coco_dir.joinpath("val2014")
    coco_json = coco_dir.joinpath("annotations", "captions_val2014.json")
    # print_coco_stats(coco_json)
    plot_resolutions(coco_data_dir)


if __name__ == "__main__":
    main()
