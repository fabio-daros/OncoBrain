import os
import random
import subprocess
import argparse
import pickle
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model.transformer_model import create_vit_model
from training.dataset import TumorDataset
from training.utils import save_model
from training import config

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Train ViT model with K-Fold cross-validation")
parser.add_argument('--folds', type=int, default=int(os.getenv('KFOLDS', 5)), help='Number of folds')
parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of epochs per fold')
parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='Learning rate')
args = parser.parse_args()

k_folds = args.folds
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

# Fixar seeds
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

# Dataset completo
full_dataset = TumorDataset(root_dirs=config.train_data_dirs, transform=transform)
all_indices = list(range(len(full_dataset)))

# TensorBoard geral
log_dir = os.path.join(config.model_output_dir, "runs_kfold")
writer = SummaryWriter(log_dir=log_dir)
try:
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6007"])
except OSError:
    print("TensorBoard já rodando ou porta 6007 ocupada.")

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []
class_names = ["NILM (normal)", "LSIL (low grade)", "HSIL (high grade)", "Invalid"]

for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
    print(f"\n=== Fold {fold + 1}/{k_folds} ===")

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = create_vit_model(num_classes=config.num_classes, freeze_base=False).to(device)

    weights = torch.tensor([1.0, 1.5, 1.0, 0.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(epochs):
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
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)

        writer.add_scalar(f"Fold_{fold + 1}/Loss/train", epoch_loss, epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Accuracy/train", epoch_acc, epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Accuracy/val", val_acc, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                output_dir=f"{config.model_output_dir}/fold_{fold + 1}",
                model_name=f"onco_vit_fold_{fold + 1}.pth"
            )
            print(f"New best model saved for fold {fold + 1} with validation accuracy: {best_val_acc:.2f}%")

        if len(val_accuracies) > 5 and all(val_accuracies[-i] <= val_accuracies[-i - 1] for i in range(1, 6)):
            print(f"Early stopping for fold {fold + 1}: no improvement for 5 epochs.")
            break

    # Salvar y_true e y_pred desse fold
    output_pickle_path = os.path.join(config.model_output_dir, f"fold_{fold + 1}", f"y_true_y_pred_fold_{fold + 1}.pkl")
    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred}, f)
    print(f"Saved y_true and y_pred arrays for Fold {fold + 1}.")

    print(f"Best Validation Accuracy for fold {fold + 1}: {best_val_acc:.2f}%")
    fold_results.append(best_val_acc)

# Resultado final
mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)

print(f"\n=== K-Fold Cross Validation Results ===")
print(f"Mean Val Accuracy: {mean_acc:.2f}%")
print(f"Standard Deviation: {std_acc:.2f}%")

writer.close()

# Caminhos dos arquivos por fold
files = {}
for fold_num in range(1, k_folds + 1):
    heatmap_path = os.path.join(config.model_output_dir, f"fold_{fold_num}", f"confusion_matrix_fold_{fold_num}.png")
    report_path = os.path.join(config.model_output_dir, f"fold_{fold_num}",
                               f"classification_report_fold_{fold_num}.txt")

    if os.path.exists(heatmap_path):
        files[f'heatmap_{fold_num}'] = (f'confusion_matrix_fold_{fold_num}.png', open(heatmap_path, 'rb'), 'image/png')
    if os.path.exists(report_path):
        files[f'report_{fold_num}'] = (f'classification_report_fold_{fold_num}.txt', open(report_path, 'rb'),
                                       'text/plain')

# Chart geral de métricas (exemplo: a média das validações ao longo das épocas, se tiver gerado)
general_chart_path = os.path.join(config.model_output_dir, "runs_kfold", "training_metrics.png")
if os.path.exists(general_chart_path):
    files['chart'] = ('training_metrics.png', open(general_chart_path, 'rb'), 'image/png')

# Número de folds
data = {'num_folds': str(k_folds)}

# Enviar
try:
    response = requests.post('http://localhost:8001/upload-kfold-training-summary/', data=data, files=files)
    if response.status_code == 200:
        print("K-Fold training results successfully sent to OncoPixel!")
    else:
        print(f"Failed to send K-Fold results: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error sending K-Fold results: {e}")
