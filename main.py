import torch
from diffusers import DiffusionPipeline
import random


metadata = {
    "prompt": "Hyperrealistic art 1boy, Madara Uchiha \(Naruto\), Naruto, black dress, outdoors, close up face, masterpiece, best quality, very aesthetic, absurdres . Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike",
    "negative_prompt": "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
    "resolution": "896 x 1152",
    "guidance_scale": 7,
    "num_inference_steps": 28,
    "seed": 833127558,
    "sampler": "Euler a",
    "sdxl_style": "Hyperrealism",
    "add_quality_tags": True,
    "quality_tags": "Standard v3.1",
    "use_upscaler": None,
    "Model": {
        "Model": "Stable Diffusion Anime Style",
        "Model hash": "e3c47aedb0"
    }
}


width, height = map(int, metadata["resolution"].split(" x "))

pipe = DiffusionPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.to('cuda')

pipe.enable_model_cpu_offload()  
pipe.enable_vae_slicing()  


prompt = metadata["prompt"]
if "sdxl_style" in metadata:
    prompt = f"{metadata['sdxl_style']}, {prompt}"

negative_prompt = metadata["negative_prompt"]

random_seed = random.randint(0, 2147483647)
metadata["seed"] = random_seed
print(f"Используется seed: {random_seed}")


generator = torch.manual_seed(random_seed)  

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    guidance_scale=metadata["guidance_scale"],
    num_inference_steps=metadata["num_inference_steps"],
    generator=generator
).images[0]

torch.cuda.empty_cache()

image.save("./output/madara_test.png")
