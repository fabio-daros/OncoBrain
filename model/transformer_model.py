import os
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

BETHESDA_CLASSES = ["NILM", "LSIL", "HSIL", "INVALID"]


def create_vit_model(num_classes, freeze_base=True):
    # Modelo do transformer
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # Congela as camadas da base se solicitado
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Substitui o head para a quantidade de classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def load_model():
    model_path = 'saved_models/onco_vit_model.pth'
    model = create_vit_model(num_classes=4)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Model partially loaded from {model_path}")
        except Exception as e:
            print(f"Could not load saved model due to mismatch: {e}")
    else:
        print(f"No saved model found at {model_path}.")
    return model


def predict_image(model, image, confidence_threshold=50):
    classes = BETHESDA_CLASSES

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[int(predicted.item())]
    confidence_score = float(confidence.item() * 100)

    if predicted_class == "INVALID" or confidence_score < confidence_threshold:
        return {
            "class": None,
            "confidence": round(confidence_score, 2),
            "message": "Unable to determine diagnosis. Possibly not a valid slide."
        }

    return {
        "class": predicted_class,
        "confidence": round(confidence_score, 2),
        "message": f"Predicted as {predicted_class} with {round(confidence_score, 2)}% confidence"
    }
