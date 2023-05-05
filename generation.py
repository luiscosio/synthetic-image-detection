from pathlib import Path
from typing import Dict, NamedTuple, Type, Union

import pandas as pd
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, LDMTextToImagePipeline, StableDiffusionPipeline

torch.manual_seed(42)


class GeneratorHugging(NamedTuple):
    weights: str
    pipeline: Type[DiffusionPipeline]
    dtype: torch.dtype


class GeneratorCustom(NamedTuple):
    weights_path: Path


class GeneratorTuple(NamedTuple):
    gen_type: str
    tuple_type: Union[GeneratorHugging, GeneratorCustom]


# Huggingface Diffusers weights are downloaded automatically on first run,
# cached in user/cache/huggingface by default on Windows
GENERATORS: Dict[str, GeneratorTuple] = {
    "StableDiffusion2": GeneratorTuple("huggingface",
                                       GeneratorHugging("stabilityai/stable-diffusion-2",
                                                        StableDiffusionPipeline,
                                                        torch.float16)),
    "LDM": GeneratorTuple("huggingface",
                          GeneratorHugging("CompVis/ldm-text2im-large-256",
                                           LDMTextToImagePipeline,
                                           torch.float32)),
}


def generate_from_huggingface(generator_id: str, huggingface_tuple: GeneratorHugging, device):
    weights, pipeline, dtype = huggingface_tuple
    pipe = pipeline.from_pretrained(weights, torch_dtype=dtype)

    # Use the Euler scheduler
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    # pipe.enable_xformers_memory_efficient_attention()  # Linux only

    csv_path = Path("csvs", "PartiPrompts.csv")
    df = pd.read_csv(csv_path, sep=";")
    prompts = df["Prompt"].tolist()
    indices = df["Index"].tolist()
    indexed_prompts = zip(indices, prompts)
    path_out = Path("data", generator_id, "PartiPrompts")
    path_out.mkdir(parents=True, exist_ok=True)

    for idx, prompt in list(indexed_prompts):
        image = pipe(prompt).images[0]
        image.save(path_out.joinpath(str(idx).zfill(4) + ".png"))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # generator_id = "StableDiffusion2"
    generator_id = "LDM"
    gen_type, sub_tuple = GENERATORS[generator_id]
    if gen_type == "huggingface":
        generate_from_huggingface(generator_id, sub_tuple, device)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
