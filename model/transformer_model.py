# model/transformer_model.py

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

def create_vit_model(num_classes):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def load_model():
    model = create_vit_model(num_classes=4)  # ou usar config.num_classes futuramente
    model.load_state_dict(torch.load('saved_models/onco_vit_model.pth', map_location='cpu'))
    model.eval()
    print("✅ Modelo carregado com sucesso!")
    return model

def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    return predicted.item()
