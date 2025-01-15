from src.style_manager import StyleManager
from src.generator import ImageGenerator
from src.utils import generate_seed, create_metadata

def main():
    style_manager = StyleManager('config/styles.json')
    
    base_prompt = "1girl, Uta \(One piece\), One piece, solo, white sleeveless dress, bright purple eyes, long hair, white hair on her right side head, red hair on her left side head, right eye is shrouded by her white hair, outdoors, looking at viewer, masterpiece, best quality, very aesthetic"
    base_negative = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    
    # Выбор стиля
    selected_style = "AnimeRetro"
    
    # Получение форматированных промптов
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
    
    # Инициализация генератора
    generator = ImageGenerator("cagliostrolab/animagine-xl-3.1")
    
    # Генерация изображения
    image = generator.generate(metadata)
    
    # Сохранение результата
    image.save("./output/result.png")

if __name__ == "__main__":
    main()
