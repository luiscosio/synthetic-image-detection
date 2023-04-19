import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

torch.manual_seed(42)

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
#pipe.enable_xformers_memory_efficient_attention()

prompts = [
"a photo of an astronaut riding a horse on mars, photorealistic",
"a photo taken on top of snowy Austrian alps, cloudy weather, overlooking at a large town with a river going across it, photorealistic",
"a photo taken on top of snowy Austrian alps, cloudy weather, overlooking at a large town with a river going across it, photorealistic",
"a photo taken on top of snowy Austrian alps, cloudy weather, overlooking at a large town with a river going across it, photorealistic",
]

for idx, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f"samples/{idx+4}.png")
