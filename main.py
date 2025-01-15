import torch
from diffusers import DiffusionPipeline
import random
import json

def load_styles(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Загружаем стили
styles_data = load_styles('styles.json')

styles = {
    style_name: style_info 
    for category in styles_data.values() 
    for style_name, style_info in category.items()
}

# Сюда вписывать промт
base_prompt = "1boy, male focus, gojou satoru, jujutsu kaisen, beach t-shirt, beach shorts, white hair, beach background"

selected_style = "Fashion"

if selected_style in styles:
    style_data = styles[selected_style]
    formatted_prompt = style_data['positive'].replace('{prompt}', base_prompt)
    formatted_negative = style_data['negative'] + ", nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
else:
    formatted_prompt = base_prompt
    formatted_negative = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"

random_seed = random.randint(0, 2147483647)
print(f"Используется seed: {random_seed}")

metadata = {
    "prompt": formatted_prompt,
    "negative_prompt": formatted_negative,
    "resolution": "896 x 1152",
    "guidance_scale": 7,
    "num_inference_steps": 50,
    "seed": random_seed,
    "sampler": "Euler a",
    "sdxl_style": selected_style,
    "add_quality_tags": True,
    "quality_tags": "Heavily detailed v3.1",
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
