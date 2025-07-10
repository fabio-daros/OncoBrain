import os
from PIL import Image, ImageDraw, ImageFont
import random

# Primeiro dataset gerado para testar se o modelo estava funcionando.
# Parâmetros
classes = ['benign', 'malignant', 'normal', 'carcinoma']
base_dir = 'dataset'
num_images_per_class = 200  # Aumentar volume
img_size = (224, 224)

# Garantir que as pastas existem
for cls in classes:
    os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

# Lista de cores e fontes aleatórias
background_colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (255, 255, 0), (128, 0, 128)]
text_colors = [(255, 255, 255), (0, 0, 0), (255, 255, 0)]

# Fonte padrão (evita erro)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Gerar imagens
for cls in classes:
    for i in range(num_images_per_class):
        bg_color = random.choice(background_colors)
        txt_color = random.choice(text_colors)
        x = random.randint(20, 100)
        y = random.randint(80, 180)

        img = Image.new('RGB', img_size, color=bg_color)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), cls, fill=txt_color, font=font)

        img.save(os.path.join(base_dir, cls, f'{cls}_{i}.png'))

print("Synthetic dataset expanded successfully!")
