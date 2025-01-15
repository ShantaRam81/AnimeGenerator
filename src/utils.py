import random
from typing import Dict

def generate_seed() -> int:
    return random.randint(0, 2147483647)

def create_metadata(prompt: str, negative_prompt: str, style: str, seed: int) -> Dict:
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": "896 x 1152",
        "guidance_scale": 7,
        "num_inference_steps": 50,
        "seed": seed,
        "sampler": "Euler a",
        "sdxl_style": style,
        "add_quality_tags": True,
        "quality_tags": "Heavily detailed v3.1",
        "use_upscaler": None,
        "Model": {
            "Model": "Stable Diffusion Anime Style",
            "Model hash": "e3c47aedb0"
        }
    } 