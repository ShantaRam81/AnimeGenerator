import torch
from diffusers import DiffusionPipeline
from typing import Dict

class ImageGenerator:
    def __init__(self, model_path: str):
        self.pipe = self._setup_pipeline(model_path)
        
    def _setup_pipeline(self, model_path: str) -> DiffusionPipeline:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.to('cuda')
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        return pipe
    
    def generate(self, metadata: Dict) -> torch.Tensor:
        width, height = map(int, metadata["resolution"].split(" x "))
        
        prompt = metadata["prompt"]
        if "sdxl_style" in metadata:
            prompt = f"{metadata['sdxl_style']}, {prompt}"
            
        generator = torch.manual_seed(metadata["seed"])
        
        image = self.pipe(
            prompt,
            negative_prompt=metadata["negative_prompt"],
            width=width,
            height=height,
            guidance_scale=metadata["guidance_scale"],
            num_inference_steps=metadata["num_inference_steps"],
            generator=generator
        ).images[0]
        
        torch.cuda.empty_cache()
        return image 