import json
from typing import Dict, Optional

class StyleManager:
    def __init__(self, styles_path: str):
        self.styles = self._load_styles(styles_path)
        
    def _load_styles(self, filename: str) -> Dict:
        with open(filename, 'r', encoding='utf-8') as file:
            styles_data = json.load(file)
            return {
                style_name: style_info 
                for category in styles_data.values() 
                for style_name, style_info in category.items()
            }
    
    def get_style(self, style_name: str, base_prompt: str, base_negative: str) -> tuple[str, str]:
        if style_name in self.styles:
            style_data = self.styles[style_name]
            formatted_prompt = style_data['positive'].replace('{prompt}', base_prompt)
            formatted_negative = style_data['negative'] + ", " + base_negative
        else:
            formatted_prompt = base_prompt
            formatted_negative = base_negative
            
        return formatted_prompt, formatted_negative 