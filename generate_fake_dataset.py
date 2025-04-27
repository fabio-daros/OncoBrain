import os
from PIL import Image, ImageDraw
import random

# Pastas
classes = ['benign', 'malignant', 'normal', 'carcinoma']
base_dir = 'dataset'

# Garantir que as pastas existem
for cls in classes:
    os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

# Gerar imagens
for cls in classes:
    for i in range(2):  # 2 imagens por classe
        img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), cls, fill=(255, 255, 255))  # Escreve o nome da classe

        img.save(os.path.join(base_dir, cls, f'{cls}_{i}.png'))

print("Mini dataset gerado com sucesso!")
