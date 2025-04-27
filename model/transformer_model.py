import os

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms


def create_vit_model(num_classes):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def load_model():
    model = create_vit_model(num_classes=4)  # <- Aqui corrigido
    model_path = 'saved_models/onco_vit_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded successfully from {model_path}.")
    else:
        print(f"Model file {model_path} not found. Proceeding without loading weights.")
    return model


def build_transformer_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, 512),  # Supondo imagens 224x224 RGB
        nn.ReLU(),
        nn.Linear(512, 4)  # 4 classes: benign, malignant, carcinoma, normal
    )
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
