# About
Code for evaluating specific synthetic image detectors, which have been re-implemented for this 
evaluation framework.

# Disclaimer
This repository contains other repositories as submodules.
The original repository of a submodule may have been edited slightly to make it compatible,
therefore their performance or results may have been altered, but their key ideas should remain the same.
The models are renamed in this repository according to their prominent detection method for easier distinction.

<details close>
<summary>Licenses</summary>

| Method                                                                                     | License                                                                                                                                                                            |
|:-------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [CLIPDetector/UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)       | [Undefined](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository#choosing-the-right-license) |
| [CNNDetector/CNNDetection](https://github.com/PeterWang512/CNNDetection)                   | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License                                                                                             |
| [DIRE](https://github.com/ZhendongWang6/DIRE)                                              | [Undefined](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository#choosing-the-right-license) |
| [EnsembleDetector/GAN-image-detection](https://github.com/polimi-ispl/GAN-image-detection) | GNU GENERAL PUBLIC LICENSE                                                                                                                                                         |

</details>

# Installation
## Cloning
Clone the repository with
``$ git clone --recurse-submodules https://github.com/tunasoup/synthetic-image-detection.git``
to install the repository with all the submodules.

Alternatively, if the repository was already cloned without submodules, use 
``$ git submodule update --init --recursive`` to install the submodules.

## Environment
[Download](https://www.python.org/downloads/) and install Python 3.8 (or later).

The required python packages can be installed via the command line by moving to the project
folder, creating a new virtual environment, and downloading the packages marked in the
[requirements](requirements.txt) file. The example virtual environment activation script is for Bash.
```
$ cd ./synthetic-image-detection
$ python -m venv venv
$ source venv/Scripts/activate
$ pip install -r requirements.txt
```
Note that the PyTorch machine learning library uses a GPU for calculations (CUDA 11.8),
relying on a CPU is not recommended.

## Weights
The detectors require their pretrained weights, or training them from scratch using the original repositories.

Create a ``weights`` directory in the repository. Create a subdirectory for each detector, to which
all the weights used by a detector is placed directly. The final configuration can be seen below.

<details close>
<summary>weights directory</summary>

```
weights
├── CLIPDetector
│   ├── fc_weights.pth
├── CNNDetector
│   ├── blur_jpg_prob0.1.pth
│   ├── blur_jpg_prob0.5.pth
├── DIRE
│   ├── 256x256_diffusion_uncond.pth
│   ├── lsun_adm.pth
├── EnsembleDetector
│   ├── method_A.pth
│   ├── method_B.pth
│   ├── method_C.pth
│   ├── method_D.pth
└── └── method_E.pth
```

</details>

For downloading the weights, refer to the original repositories, or use the provided links/instructions:
- CLIPDetector: Copy [fc_weights](https://github.com/Yuheng-Li/UniversalFakeDetect/blob/main/pretrained_weights/fc_weights.pth)
from the original or submodule repository, CLIP weights are downloaded automatically when first run
- CNNDetector: [blur_jpg_prob0.1](https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth) &
[blur_jpg_prob0.5](https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth) (Dropbox)
- DIRE: [lsun_adm](https://mailustceducn-my.sharepoint.com/personal/zhendongwang_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fzhendongwang%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fdatasets%2FDiffusionForensics%2Fcheckpoints%2Flsun%5Fadm%2Epth&parent=%2Fpersonal%2Fzhendongwang%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fdatasets%2FDiffusionForensics%2Fcheckpoints)
(OneDrive) & [256x256_diffusion_uncond](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) (direct)
- EnsembleDetector: [methods](https://www.dropbox.com/s/n1boisish8m6aoj/weights.zip) (Dropbox), unzip and place each weight file
directly under the detector's weight directory

For evaluating other detectors that are not included in this repository, their PyTorch implementations are required to 
be added as a submodule and implementing the Detector class. Alternatively, a detector's results could be saved in a 
similar CSV file to enable evaluation.

# Usage
The specific detectors are tested for whole datasets in [detection.py](detection.py), 
and their results are saved to a CSV file.

The results from the CSV files are printed in [evaluations.py](evaluations.py).

[data_filtering.py](utils/data_filtering.py) can be used to filter downloaded datasets that are in a specific format.
The filters contain hard-coded values, which can be adjusted in the code.

Stable Diffusion 2.1 and LDM can be used to create synthetic images from prompts in [generation.py](generation.py).
Otherwise, datasets should be downloaded from elsewhere.

Place each downloaded dataset to a ``data`` directory. Each dataset or their subsets should have their
all their images in the same directory, without mixing any synthetic and real images. The paths for the used datasets
are hard-coded in [detection.py](detection.py) and paired with a correct label, 1 for synthetic, 0 for real.

<details close>
<summary>Example datasets</summary>

| Name & download location                                                                                                                                                         | Class | 
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------|
| [COCO 2014 validation](https://cocodataset.org/#download)                                                                                                                        | Real  |
| [HDR/SDR](https://lesc.dinfo.unifi.it/materials/datasets_en.html)                                                                                                                | Real  |
| [Midjourney v5.1](https://www.kaggle.com/datasets/iraklip/modjourney-v51-cleaned-data) ([data_filtering.py](data_filtering.py) requires the CSV file for downloading the images) | Fake  | 
| [StyleGAN2](https://github.com/peterwang512/CNNDetection) (CNNDetection)                                                                                                         | Fake  |
| [VQGAN](https://github.com/CompVis/taming-transformers) (Taming Transformers)                                                                                                    | Fake  |
| [GANs and DMs](https://github.com/grip-unina/DMimageDetection) (DMimageDetection)                                                                                                | Fake  |
| [Dalle-3](https://huggingface.co/datasets/laion/dalle-3-dataset) (Dall-3 dataset)                                                                                                | Fake  |
</details>


<details close>
<summary>Example command-line commands</summary>

```
Detection using hard-coded configurations:
$ python detection.py -d CNNDetector_p0.1_crop -ds StableDiffusion2 -bs 50 --verbose

Detection using custom configurations:
$ python detection.py -d CNNDetector_p0.1_heavy_compression -dc cnndetector -dw weights/cnndetector/blur_jpg_prob0.1.pth -dsd data/StableDiffusion2/text -dsl 1 -bs 50 -c 40 -cs "None" -rs "(500, 500)" -v -o csvs/myresults.csv

Evaluating the results on multiple resize-augmented datasets of multiple detectors with balanced thresholds
$ python evaluation.py acc -i csvs -cf bilinear -d CLIPDetector_crop CNNDetector_p0.1 -bp csvs/SDR.csv csvs/StableDiffusion2.csv 

Plotting the Area Under the ROC Curve and average precision:
$ python evaluation.py aucap -i csvs/SDR.csv csvs/StableDiffusion2.csv -d CLIPDetector_crop

Generating images with Stable Diffusion 2 from a text prompt:
$ python generation.py -i "Hello, World!" -g StableDiffusion2 -n 2
```

</details>
