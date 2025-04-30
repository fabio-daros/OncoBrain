import os
import shutil
import matplotlib.pyplot as plt

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

best_acc = 0.0  # Variável para guardar a melhor accuracy

train_losses = []
train_accuracies = []

# Loop de treino
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    print(f"Epoch [{epoch + 1}/{config.epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Salvar o modelo se a accuracy melhorar
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        save_model(model, output_dir=config.model_output_dir)
        print(f"New best model saved with accuracy: {best_acc:.2f}%")

# Plotar Loss e Accuracy
epochs_range = range(1, config.epochs + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config.model_output_dir, "training_metrics.png"))

# Caminho de destino no OncoPixel
destination_path = "../OncoPixel/media/training/training_metrics.png"

try:
    shutil.copy(os.path.join(config.model_output_dir, "training_metrics.png"), destination_path)
    print(f"Training chart copied to: {destination_path}")
except Exception as e:
    print(f"Failed to copy training chart: {e}")


plt.show()
