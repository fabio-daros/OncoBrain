import argparse
import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from model.transformer_model import create_vit_model
from training.dataset import TumorDataset
from training import config

# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate saved OncoBrain K-Fold models")
parser.add_argument('--folds', type=int, default=5, help='Number of folds')
parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size for evaluation')
args = parser.parse_args()

k_folds = args.folds
batch_size = args.batch_size

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
])

# Dataset completo
full_dataset = TumorDataset(root_dir=config.train_data_dir, transform=transform)
all_indices = list(range(len(full_dataset)))

class_names = ["NILM (normal)", "LSIL (low grade)", "HSIL (high grade)", "Invalid"]

for fold in range(1, k_folds + 1):
    print(f"\n=== Evaluating model from Fold {fold} ===")

    model = create_vit_model(num_classes=config.num_classes, freeze_base=False).to(device)
    model_path = os.path.join(config.model_output_dir, f"fold_{fold}", f"onco_vit_fold_{fold}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    fold_size = len(all_indices) // k_folds
    start = (fold - 1) * fold_size
    end = start + fold_size if fold != k_folds else len(all_indices)
    val_indices = all_indices[start:end]
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    labels = list(range(len(class_names)))

    report = classification_report(y_true, y_pred, target_names=class_names, labels=labels, digits=3, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\nClassification Report - Fold {fold}:\n", report)
    print(f"Confusion Matrix - Fold {fold}:\n", matrix)

    report_path = os.path.join(config.model_output_dir, f"fold_{fold}", f"classification_report_fold_{fold}.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for Fold {fold}:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(matrix))

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.tight_layout()
    heatmap_path = os.path.join(config.model_output_dir, f"fold_{fold}", f"confusion_matrix_fold_{fold}.png")
    plt.savefig(heatmap_path)
    plt.close()

    print(f"Saved report and heatmap for Fold {fold}.")
