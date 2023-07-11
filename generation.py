from pathlib import Path
from typing import Dict, NamedTuple, Type
from datetime import datetime

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
                      index_column: str = "index",
                      timesteps: int = 50,
                      prompt_prefix: str = ""):
    pipe = load_pipe(generator_id, device)

    # List prompts and indices
    df = pd.read_csv(csv_path, sep=sep)
    prompts = df[prompt_column].tolist()
    if index_column in df.columns:
        indices = df[index_column].tolist()
    else:
        indices = list(range(len(prompts)))
    indexed_prompts = zip(indices, prompts)

    path_out = Path("data", generator_id, csv_path.stem)
    path_out.mkdir(parents=True, exist_ok=True)

    for idx, prompt in tqdm(list(indexed_prompts), unit="img"):
        image = pipe(prompt_prefix + prompt, num_inference_steps=timesteps).images[0]
        image.save(path_out.joinpath(str(idx).zfill(4) + ".png"))


def generate_from_text(generator_id: str, text: str, device: str, num_images: int = 1, timesteps: int = 50):
    pipe = load_pipe(generator_id, device)
    path_out = Path("data", generator_id, "text")
    path_out.mkdir(parents=True, exist_ok=True)
    text_out = text[:200]  # Limit filename length
    for _ in tqdm(range(num_images)):
        image = pipe(text, num_inference_steps=timesteps).images[0]
        image.save(path_out.joinpath(str(datetime.now()).replace(":", ".") + f" {text_out}" + ".png"))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p2_path = Path("csvs", "PartiPrompts.csv")
    generator_id = "StableDiffusion2"
    # generator_id = "LDM"
    # generate_from_csv(generator_id, p2_path, device, sep=";", prompt_column="Prompt", index_column="Index")
    generate_from_text(generator_id, "a newspaper with a headline about a new species of fish",
                       device, num_images=3, timesteps=150)


if __name__ == "__main__":
    main()
