import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms
from model.faster_rcnn_model import create_faster_rcnn_model

# Configurações
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 7  # 6 classes + background
checkpoint_path = "./saved_models/faster_rcnn_model_checkpoints/faster_rcnn_epoch_0.pth"
image_path = "./dataset/cric_detection/images/0a2a5a681410054941cc56f51eb8fbda.png"  # Altere para o caminho da imagem que quiser testar

# Mapeamento das categorias (id → nome da classe)
CATEGORY_MAPPING = {
    1: "Negative",
    2: "ASC-US",
    3: "LSIL",
    4: "ASC-H",
    5: "HSIL",
    6: "SCC"
}

# Carregar modelo
model = create_faster_rcnn_model(num_classes=num_classes)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Carregar e transformar imagem
image = Image.open(image_path).convert("RGB")
transform = transforms.ToTensor()
img_tensor = transform(image).unsqueeze(0).to(device)

# Fazer predição
with torch.no_grad():
    outputs = model(img_tensor)

output = outputs[0]

# Filtrar por threshold de score
score_threshold = 0.7
filtered_boxes = []
filtered_labels = []
filtered_scores = []

for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    if score >= score_threshold:
        filtered_boxes.append(box.cpu())
        filtered_labels.append(label.cpu().item())
        filtered_scores.append(score.cpu().item())

# Desenhar as caixas na imagem
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
    x_min, y_min, x_max, y_max = box.tolist()
    label_name = CATEGORY_MAPPING.get(label, f"Class {label}")
    text = f"{label_name}: {score:.2f}"

    draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
    draw.text((x_min, y_min - 10), text, fill="red", font=font)

# Mostrar resultado
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()
