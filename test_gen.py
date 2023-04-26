import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, LDMTextToImagePipeline, UnCLIPPipeline

torch.manual_seed(42)

# Diffusers weights are downloaded automatically on first run,
# cached in user/cache/huggingface by default on Windows
sd2 = "stabilityai/stable-diffusion-2"
ldm = "CompVis/ldm-text2im-large-256"
dalle2 = "nousr/conditioned-prior"

# pipe = StableDiffusionPipeline.from_pretrained(sd2, torch_dtype=torch.float16)
# pipe = LDMTextToImagePipeline.from_pretrained(ldm)
pipe = UnCLIPPipeline.from_pretrained(dalle2)

# Use the Euler scheduler
# pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

prompts = [
"a photo of an astronaut riding a horse on mars, photorealistic",
"a photo taken on top of snowy Austrian alps, cloudy weather, overlooking at a large town with a river going across it, photorealistic",
"a photo taken on top of snowy Austrian alps, cloudy weather, overlooking at a large town with a river going across it, photorealistic",
"a photo taken on top of snowy Austrian alps, cloudy weather, overlooking at a large town with a river going across it, photorealistic",
]

for idx, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f"samples/{idx+8}.png")
