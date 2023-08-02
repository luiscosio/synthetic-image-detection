# About
A work-in-progress collection of synthetic image detectors.

# Disclaimer
This repository contains other repositories as submodules.
The original repository of a submodule may have been edited slightly to make it compatible,
therefore their performance or results may have been altered, but their key ideas should remain the same.
<details close>
<summary>Licenses</summary>

| Method                                                                    | License                                                                                |
|:--------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| [CNNDetection](https://github.com/PeterWang512/CNNDetection)              | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License |
| [GAN-image-detection](https://github.com/polimi-ispl/GAN-image-detection) | GNU GENERAL PUBLIC LICENSE                                                             |

</details>

# Installation
## Cloning
Clone the repo with
``$ git clone --recurse-submodules https://github.com/tunasoup/synthetic-image-detection.git``
to install the repo with all the submodules.

Alternatively, if the repository was already cloned without submodules, use 
``$ git submodule update --init --recursive`` to install the submodules.

## Weights
Download weights for the specified models and unpack them into the ``weights`` folder.

TODO: specify models and link weights

## Environments
TODO

# Usage
TODO

Stable Diffusion 2.1 and LDM can be used to create synthetic images from prompts in [generation.py](generation.py).

[data_filtering.py](utils/data_filtering.py) can be used to filter specific, downloaded datasets.

<details close>
<summary>Example datasets</summary>

| Name & download location                                                                           | Class | 
|:---------------------------------------------------------------------------------------------------|:------|
| [COCO 2014 validation](https://cocodataset.org/#download)                                          | Real  |
| [Midjourney v5.1](https://www.kaggle.com/datasets/iraklip/modjourney-v51-cleaned-data)             | Fake  | 
| [StyleGAN2](https://github.com/peterwang512/CNNDetection) (CNNDetection)                           | Fake  |

</details>
