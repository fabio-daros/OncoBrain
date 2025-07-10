# Classes baseadas na classificacao Bethesda:
# 0: NILM  - Normal (Negative for Intraepithelial Lesion or Malignancy)
# 1: LSIL  - Low-grade intraepithelial scams (low-grade)
# 2: HSIL  - High-grade intraepithelial scams (high-grade, pre-cancer)
# 3: INVALID - Images not related to blades

import os
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model.transformer_model import create_vit_model
from training.dataset import TumorDataset
from training.utils import save_model
from training import config

# Caminho absoluto do diretório do script atual (training/)
ONCOTRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho absoluto até a pasta OncoPixel/media/training
ONCOPIXEL_MEDIA_DIR = os.path.abspath(os.path.join(ONCOTRAIN_DIR, "..", "OncoPixel", "media", "training"))

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# Dataset and loaders
full_dataset = TumorDataset(root_dir=config.train_data_dir, transform=transform)
train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=config.batch_size, shuffle=False)

# Model with fine-tuning freeze_base=False.
model = create_vit_model(num_classes=config.num_classes, freeze_base=False).to(device)
print("Model output features:", model.heads.head.out_features)
print("Expected output classes:", config.num_classes)

# Loss and optimizer
weights = torch.tensor([1.0, 1.5, 1.0, 0.5], device=device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# TensorBoard - ferramenta para acompanhar o treino em tempo real
log_dir = os.path.join(config.model_output_dir, "runs")
writer = SummaryWriter(log_dir=log_dir)
try:
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
except OSError:
    print("TensorBoard already running or port 6006 busy.")

# Training loop - zerados para resetar as metricas a cada epoch
best_val_acc = 0.0
train_losses, train_accuracies, val_accuracies = [], [], []

for epoch in range(config.epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    val_accuracies.append(val_acc)

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    if len(val_accuracies) > 5 and all(val_accuracies[-i] <= val_accuracies[-i - 1] for i in range(1, 6)):
        print(f"Early stopping: no improvement for 5 epochs.")
        break

    print(
        f"Epoch [{epoch + 1}/{config.epochs}] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(model, output_dir=config.model_output_dir)
        print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

# Plot metrics
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Loss")
plt.xlabel("Epoch");
plt.ylabel("Loss");
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Train Acc")
plt.plot(epochs_range, val_accuracies, label="Val Acc", color="orange")
plt.xlabel("Epoch");
plt.ylabel("Accuracy (%)");
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config.model_output_dir, "training_metrics.png"))
plt.show()

try:
    shutil.copy(os.path.join(config.model_output_dir, "training_metrics.png"),
                "../OncoPixel/media/training/training_metrics.png")
except Exception as e:
    print(f"Failed to copy training chart: {e}")

# Classification report
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

class_names = ["NILM (normal)", "LSIL (low grade)", "HSIL (high grade)", "Invalid"]
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
matrix = confusion_matrix(y_true, y_pred)

print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Heatmap)')
plt.tight_layout()
plt.savefig(os.path.join(config.model_output_dir, "confusion_matrix_heatmap.png"))
plt.show()

# Copiar para o OncoPixel localmente
try:
    shutil.copy(os.path.join(config.model_output_dir, "confusion_matrix_heatmap.png"),
                "../OncoPixel/media/training/confusion_matrix_heatmap.png")
except Exception as e:
    print(f"Failed to copy heatmap: {e}")

# Enviar heatmap para o OncoPixel via API
heatmap_path = os.path.join(config.model_output_dir, "confusion_matrix_heatmap.png")

# Salvar classification report
with open(os.path.join(config.model_output_dir, "classification_report.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(matrix))

# Caminhos dos arquivos
training_chart_path = os.path.join(config.model_output_dir, "training_metrics.png")
heatmap_path = os.path.join(config.model_output_dir, "confusion_matrix_heatmap.png")
report_path = os.path.join(config.model_output_dir, "classification_report.txt")

# Copiar para pasta do OncoPixel localmente. (validar se ainda precisa)
try:
    shutil.copy(training_chart_path, os.path.join(ONCOPIXEL_MEDIA_DIR, "training_metrics.png"))
    shutil.copy(heatmap_path, os.path.join(ONCOPIXEL_MEDIA_DIR, "confusion_matrix_heatmap.png"))
    shutil.copy(report_path, os.path.join(ONCOPIXEL_MEDIA_DIR, "classification_report.txt"))

except Exception as e:
    print(f"Failed by copying files locally: {e}")

# Enviar todos para o OncoPixel via upload-training-summary
try:
    files = {}
    if os.path.exists(training_chart_path):
        files['chart'] = ('training_metrics.png', open(training_chart_path, 'rb'), 'image/png')
    if os.path.exists(heatmap_path):
        files['heatmap'] = ('confusion_matrix_heatmap.png', open(heatmap_path, 'rb'), 'image/png')
    if os.path.exists(report_path):
        files['report'] = ('classification_report.txt', open(report_path, 'rb'), 'text/plain')

    response = requests.post('http://localhost:8001/upload-training-summary/', files=files)

    if response.status_code == 200:
        print("Training files successfully sent to Oncopixel:", response.json())
    else:
        print("Failure when sending training files:", response.text)

except Exception as e:
    print(f"Error trying to send training files: {e}")

# Finalizar TensorBoard
writer.close()
