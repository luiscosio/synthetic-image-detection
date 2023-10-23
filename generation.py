import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Type

import pandas as pd
import torch
from diffusers import DiffusionPipeline, LDMTextToImagePipeline, StableDiffusionPipeline
from tqdm import tqdm

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


class GeneratorHugging(NamedTuple):
    weights: str
    pipeline: Type[DiffusionPipeline]
    dtype: torch.dtype


# Huggingface Diffusers weights are downloaded automatically on first run,
# cached in user/cache/huggingface by default on Windows
GENERATORS: Dict[str, GeneratorHugging] = {
    "StableDiffusion2": GeneratorHugging("stabilityai/stable-diffusion-2-1-base",
                                         StableDiffusionPipeline,
                                         torch.float16),
    "LDM": GeneratorHugging("CompVis/ldm-text2im-large-256",
                            LDMTextToImagePipeline,
                            torch.float32),
}


def load_pipe(generator_id: str, device: str):
    weights, pipeline, dtype = GENERATORS[generator_id]
    pipe = pipeline.from_pretrained(weights, torch_dtype=dtype)

    # Use the Euler scheduler
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    # pipe.enable_xformers_memory_efficient_attention()  # Linux only
    return pipe


def generate_from_csv(generator_id: str,
                      csv_path: Path,
                      device: str,
                      sep: str = ";",
                      prompt_column: str = "prompt",
                      index_column: Optional[str] = None,
                      timesteps: int = 50,
                      prompt_prefix: str = "",
                      path_out: Path = None) -> None:
    """
    Generate images from prompts in a CSV file using a diffusion model with the given ID.
    By default, the images are saved in the data directory, under the generator and CSV filename.

    Args:
        generator_id: ID of the generator to use
        csv_path: Path to the CSV file with prompts
        device: Device to run the generator on
        sep: Separator of the CSV file
        prompt_column: Case-sensitive name of the column with prompts
        index_column: Optional case-sensitive name of the column with indices used for filenaming
        timesteps: Number of iterative denoising steps
        prompt_prefix: Optional prefix to prepend to each prompt
        path_out: Optional output directory for the images
    """
    pipe = load_pipe(generator_id, device)

    # List prompts and indices
    df = pd.read_csv(csv_path, sep=sep)
    prompts = df[prompt_column].tolist()
    zfill_size = 0  # Pad names with zeros only with automatic indices
    if index_column in df.columns:
        indices = df[index_column].tolist()
    else:
        indices = list(range(len(prompts)))
        zfill_size = len(str(len(prompts)))
    indexed_prompts = zip(indices, prompts)

    if path_out is None:
        path_out = Path("data", generator_id, csv_path.stem + f"_ts{timesteps}")
    path_out.mkdir(parents=True, exist_ok=True)
    print(f"Images will be saved to {path_out.absolute()}")

    for idx, prompt in tqdm(list(indexed_prompts), desc="Generating images", unit="img"):
        image = pipe(prompt_prefix + prompt, num_inference_steps=timesteps).images[0]
        image.save(path_out.joinpath(str(idx).zfill(zfill_size) + ".png"))


def generate_from_text(generator_id: str,
                       text: str,
                       device: str,
                       num_images: int = 1,
                       timesteps: int = 50,
                       path_out: Path = None) -> None:
    """
    Generate images from a text prompt using a diffusion model with the given ID.
    By default, the images are saved in the data directory, under the generator's text subdirectory.

    Args:
        generator_id: ID of the generator to use
        text: Text prompt
        device: Device to run the generator on
        num_images: Number of images to generate from the prompt
        timesteps: Number of iterative denoising steps
        path_out: Optional output directory for the images
    """
    pipe = load_pipe(generator_id, device)
    text_out = text[:100]  # Limit filename length

    if path_out is None:
        path_out = Path("data", generator_id, "text")
    path_out.mkdir(parents=True, exist_ok=True)
    print(f"Images will be saved to {path_out.absolute()}")

    for _ in tqdm(range(num_images), desc="Generating images", unit="img"):
        image = pipe(text, num_inference_steps=timesteps).images[0]
        image.save(path_out.joinpath(str(datetime.now()).replace(":", ".") + f" {text_out}" + ".png"))


def handle_args(generator_id: str,
                prompt_input: str,
                prompt_column: str,
                index_column: str,
                timesteps: int,
                num_images: int,
                sep: str,
                path_out: Path,
                force_cpu: bool,
                ) -> None:
    device = "cpu"
    if force_cpu:
        print("Forcing CPU use")

    else:
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"Using {device}")
        else:
            print("No GPU or CUDA drivers found, using CPU")

    if prompt_input.lower().endswith(".csv"):
        prompt_input = Path(prompt_input)
        if prompt_column is None:
            raise ValueError(f"prompt_column needs to be defined when supplying a prompt CSV file")
        generate_from_csv(generator_id, prompt_input, device=device, sep=sep, prompt_column=prompt_column,
                          index_column=index_column, timesteps=timesteps, path_out=path_out)
    else:
        generate_from_text(generator_id, prompt_input, device=device,
                           num_images=num_images, timesteps=timesteps, path_out=path_out)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     allow_abbrev=False,
                                     description="Generate images from text either with Stable Diffusion 2 "
                                                 "or LDM. The required input is either a text prompt, or "
                                                 "a path to a CSV file containing a prompt on every row. "
                                                 "The column is specified by a prompt-column parameter. "
                                                 "The images are saved in the data directory, under the "
                                                 "used generator's name by default.")

    parser.add_argument("--input", "-i", type=str, required=True, dest="prompt_input",
                        help="Text prompt or a path to a CSV file (.csv extension) with a prompt on every row")

    parser.add_argument("--generator", "-g", choices=["StableDiffusion2", "LDM"],
                        default="StableDiffusion2", dest="generator_id",
                        help="Name of the image generator")

    parser.add_argument("--timesteps", "-t", type=int, default=50, dest="timesteps",
                        help="Number of iterative denoising steps")

    parser.add_argument("--output", "-o", type=Path, default=None, dest="path_out",
                        help="Optional output directory for the images")

    parser.add_argument("--force-cpu", "-f", action="store_true", dest="force_cpu",
                        help="Force the using of CPU, not recommended")

    parser.add_argument("--prompt-column", "-pc", type=str, dest="prompt_column",
                        help="CSV column name for prompts, required if the input is a CSV file")

    parser.add_argument("--index-column", "-ic", type=str, dest="index_column",
                        help="Optinonal CSV column name for indices, running indexing used if not given")

    parser.add_argument("--separator", "-s", type=str, default=";", dest="sep",
                        help="Separator of the CSV file")

    parser.add_argument("--number-of-images", "-n", type=int, default=3, dest="num_images",
                        help="Number of images to generate from the text prompt (no CSV support)")

    args = parser.parse_args()
    # args = parser.parse_args(["This is a test prompt", "-n 5"])
    handle_args(**vars(args))


if __name__ == "__main__":
    main()
