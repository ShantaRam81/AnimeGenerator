from src.style_manager import StyleManager
from src.generator import ImageGenerator
from src.utils import generate_seed, create_metadata
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import logging
import threading

logging.basicConfig(
    filename='generation_history.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class StableDiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stable Diffusion GUI")
        self.root.geometry("1200x800")
        
        # Инициализация менеджеров
        self.style_manager = StyleManager('config/styles.json')
        self.generator = ImageGenerator("cagliostrolab/animagine-xl-3.1")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Создаем фреймы
        left_frame = ttk.Frame(self.root, padding="10")
        right_frame = ttk.Frame(self.root, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Промпт
        ttk.Label(left_frame, text="Prompt:").pack(anchor=tk.W)
        self.prompt_text = scrolledtext.ScrolledText(left_frame, height=5, width=50)
        self.prompt_text.pack(fill=tk.X, pady=5)
        self.prompt_text.insert(tk.END, "")
        
        # Негативный промпт
        ttk.Label(left_frame, text="Negative Prompt:").pack(anchor=tk.W)
        self.negative_prompt_text = scrolledtext.ScrolledText(left_frame, height=3, width=50)
        self.negative_prompt_text.pack(fill=tk.X, pady=5)
        self.negative_prompt_text.insert(tk.END, "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]")
        
        # Выпадающий список стилей
        ttk.Label(left_frame, text="Style:").pack(anchor=tk.W)
        self.style_var = tk.StringVar()
        self.style_combo = ttk.Combobox(left_frame, textvariable=self.style_var)
        self.style_combo['values'] = list(self.style_manager.styles.keys())
        self.style_combo.set("AnimeRetro")
        self.style_combo.pack(fill=tk.X, pady=5)
        
        # Seed
        seed_frame = ttk.Frame(left_frame)
        seed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT)
        self.seed_var = tk.StringVar(value=str(generate_seed()))
        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var)
        self.seed_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        ttk.Button(seed_frame, text="🎲", width=3, 
                  command=lambda: self.seed_var.set(str(generate_seed()))
        ).pack(side=tk.RIGHT)
        
        # Кнопка генерации
        self.generate_btn = ttk.Button(left_frame, text="Generate", command=self.generate_image)
        self.generate_btn.pack(fill=tk.X, pady=10)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Область предпросмотра
        self.preview_label = ttk.Label(right_frame, text="Generated image will appear here")
        self.preview_label.pack(pady=20)
        
        # Статус
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
    def generate_image(self):
        # Отключаем кнопку на время генерации
        self.generate_btn.state(['disabled'])
        self.progress.start()
        self.status_var.set("Generating...")
        
        # Запускаем генерацию в отдельном потоке
        thread = threading.Thread(target=self._generate_image_thread)
        thread.start()
        
    def _generate_image_thread(self):
        try:
            # Получаем значения из интерфейса
            prompt = self.prompt_text.get("1.0", tk.END).strip()
            negative_prompt = self.negative_prompt_text.get("1.0", tk.END).strip()
            style = self.style_var.get()
            seed = int(self.seed_var.get())
            
            # Логируем параметры
            logging.info(f"Generating image with: style={style}, seed={seed}")
            
            # Получаем форматированные промпты
            formatted_prompt, formatted_negative = self.style_manager.get_style(
                style, prompt, negative_prompt
            )
            
            # Создаем метаданные
            metadata = create_metadata(
                formatted_prompt,
                formatted_negative,
                style,
                seed
            )
            
            # Генерируем изображение
            image = self.generator.generate(metadata)
            
            # Сохраняем результат
            image.save("./output/result.png")
            
            # Обновляем предпросмотр
            self.root.after(0, self.update_preview, image)
            
            # Обновляем статус
            self.root.after(0, self.status_var.set, "Generation complete!")
            
        except Exception as e:
            # Логируем ошибку
            logging.error(f"Generation failed: {str(e)}")
            self.root.after(0, self.status_var.set, f"Error: {str(e)}")
            
        finally:
            # Включаем кнопку обратно
            self.root.after(0, self.generate_btn.state, ['!disabled'])
            self.root.after(0, self.progress.stop)
            
    def update_preview(self, image):
        # Изменяем размер изображения для предпросмотра
        preview_size = (400, 512)
        image.thumbnail(preview_size, Image.Resampling.LANCZOS)
        
        # Конвертируем в формат tkinter
        photo = ImageTk.PhotoImage(image)
        
        # Обновляем метку с изображением
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # Сохраняем ссылку

def main():
    root = tk.Tk()
    app = StableDiffusionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
