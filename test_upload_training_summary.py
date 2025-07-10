import os
import requests

# Caminho dos arquivos já gerados
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_models"))
chart_path = os.path.join(BASE_DIR, "training_metrics.png")
heatmap_path = os.path.join(BASE_DIR, "confusion_matrix_heatmap.png")
report_path = os.path.join(BASE_DIR, "classification_report.txt")

# Endpoint do OncoPixel
endpoint = "http://localhost:8001/upload-training-summary/"

files = {}
if os.path.exists(chart_path):
    files['chart'] = ('training_metrics.png', open(chart_path, 'rb'), 'image/png')

if os.path.exists(heatmap_path):
    files['heatmap'] = ('confusion_matrix_heatmap.png', open(heatmap_path, 'rb'), 'image/png')

if os.path.exists(report_path):
    files['report'] = ('classification_report.txt', open(report_path, 'rb'), 'text/plain')

response = requests.post(endpoint, files=files)

if response.status_code == 200:
    print("Files Subjected Successfully:", response.json())
else:
    print("Failed to send files:", response.status_code, response.text)
