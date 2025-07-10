import os
import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from training import config

parser = argparse.ArgumentParser(description="Average K-Fold Results")
parser.add_argument('--folds', type=int, default=5, help='Number of folds')
args = parser.parse_args()
k_folds = args.folds

all_y_true = []
all_y_pred = []

class_names = ["NILM (normal)", "LSIL (low grade)", "HSIL (high grade)", "Invalid"]

for fold in range(1, k_folds + 1):
    pkl_path = os.path.join(config.model_output_dir, f"fold_{fold}", f"y_true_y_pred_fold_{fold}.pkl")
    if not os.path.exists(pkl_path):
        print(f"Skip fold {fold}: pickle file not found at {pkl_path}")
        continue

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        all_y_true.extend(data['y_true'])
        all_y_pred.extend(data['y_pred'])

print("\n=== Overall Classification Report ===")
report = classification_report(all_y_true, all_y_pred, target_names=class_names, digits=3)
print(report)

print("\n=== Overall Confusion Matrix ===")
matrix = confusion_matrix(all_y_true, all_y_pred)
print(matrix)

# Salvar relatório
output_dir = os.path.join(config.model_output_dir, "kfold_summary")
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, "overall_classification_report.txt")
with open(report_path, "w") as f:
    f.write("Overall Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(matrix))

# Salvar heatmap da matriz de confusão geral
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Overall Confusion Matrix - KFold Summary')
plt.tight_layout()
heatmap_path = os.path.join(output_dir, "overall_confusion_matrix.png")
plt.savefig(heatmap_path)
plt.close()

print(f"\nSaved overall classification report and heatmap to: {output_dir}")
