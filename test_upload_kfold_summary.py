import os
import requests

# Diretório base onde estão os modelos e os resultados dos folds
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_models"))
NUM_FOLDS = 3  # Ajuste conforme o número de folds que você treinou

# Dados para o POST
files = {}
data = {'num_folds': str(NUM_FOLDS)}

# Loop para pegar os arquivos de cada fold
for fold_num in range(1, NUM_FOLDS + 1):
    fold_dir = os.path.join(BASE_DIR, f"fold_{fold_num}")

    heatmap_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold_num}.png")
    report_path = os.path.join(fold_dir, f"classification_report_fold_{fold_num}.txt")

    if os.path.exists(heatmap_path):
        files[f'heatmap_{fold_num}'] = (f'confusion_matrix_fold_{fold_num}.png', open(heatmap_path, 'rb'), 'image/png')
    else:
        print(f"Warning: Heatmap for fold {fold_num} not found at {heatmap_path}")

    if os.path.exists(report_path):
        files[f'report_{fold_num}'] = (f'classification_report_fold_{fold_num}.txt', open(report_path, 'rb'), 'text/plain')
    else:
        print(f"Warning: Report for fold {fold_num} not found at {report_path}")

# Adicional: chart geral (se tiver gerado manualmente ou por script)
general_chart_path = os.path.join(BASE_DIR, "runs_kfold", "training_metrics.png")
if os.path.exists(general_chart_path):
    files['chart'] = ('training_metrics.png', open(general_chart_path, 'rb'), 'image/png')
else:
    print(f"Warning: General chart not found at {general_chart_path}")

# Endpoint correto (usando underscores como está no Django)
endpoint = "http://localhost:8001/upload_kfold_training_summary/"

# Envio
try:
    response = requests.post(endpoint, data=data, files=files)
    if response.status_code == 200:
        print("✅ K-Fold training results successfully sent to OncoPixel!")
        print(response.json())
    else:
        print(f"Failed to send K-Fold results: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error sending K-Fold results: {e}")
