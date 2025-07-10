from dotenv import load_dotenv
import os

# Carregar variáveis do .env
load_dotenv()

# Diretórios
train_data_dir = os.getenv('DATASET_DIR', 'dataset/bmt')  # Para classificação
model_output_dir = os.getenv('MODEL_DIR', 'saved_models/')

# Detecção - dataset detection (caso queira diferenciar depois)
detection_images_dir = os.getenv('DETECTION_IMAGES_DIR', 'dataset/cric_detection/images')
detection_annotations_path = os.getenv('DETECTION_ANNOTATIONS', 'dataset/cric_detection/annotations/annotations_coco.json')

# Parâmetros do modelo
num_classes = int(os.getenv('NUM_CLASSES', 4))  # Mude para 7 se for detecção

# Treinamento geral
batch_size = int(os.getenv('BATCH_SIZE', 32))
epochs = int(os.getenv('EPOCHS', 30))
learning_rate = float(os.getenv('LEARNING_RATE', 1e-4))

# Tamanho das imagens (caso precise para ViT)
img_size = (int(os.getenv('IMG_SIZE', 224)), int(os.getenv('IMG_SIZE', 224)))
