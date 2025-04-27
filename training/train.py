# training/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.transformer_model import create_vit_model
from training.dataset import TumorDataset
from training.utils import save_model
from training import config

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformações nas imagens
transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
])

# Dataset e Dataloader
train_dataset = TumorDataset(root_dir=config.train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# Modelo
model = create_vit_model(num_classes=config.num_classes)
model = model.to(device)

# Otimizador e Perda
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Loop de treino
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{config.epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Salvar modelo treinado
save_model(model, output_dir=config.model_output_dir)
