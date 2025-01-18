from src.style_manager import StyleManager
from src.generator import ImageGenerator
from src.utils import generate_seed, create_metadata
import os

def main():
    style_manager = StyleManager('config/styles.json')
    
    base_prompt = "1boy, Roronoa Zoro \(One piece\), One piece, solo, face close up"
    base_negative = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    
    selected_style = "Anime"
    
    formatted_prompt, formatted_negative = style_manager.get_style(
        selected_style, base_prompt, base_negative
    )
    
    # Генерация seed
    seed = generate_seed()
    print(f"Используется seed: {seed}")
    
    metadata = create_metadata(
        formatted_prompt, 
        formatted_negative, 
        selected_style,
        seed
    )
    

    generator = ImageGenerator("cagliostrolab/animagine-xl-3.1")
    

    image = generator.generate(metadata)
    

    os.makedirs("output", exist_ok=True)
    
    output_file = metadata["output_file"]
    image.save(output_file)
    print(f"Изображение сохранено как: {output_file}")

if __name__ == "__main__":
    main()
