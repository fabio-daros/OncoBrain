# training/config.py

from dotenv import load_dotenv
import os

# Carregar variáveis do .env
load_dotenv()

# Diretórios
train_data_dir = os.getenv('DATASET_DIR', 'dataset/')
model_output_dir = os.getenv('MODEL_DIR', 'saved_models/')

# Modelo
num_classes = int(os.getenv('NUM_CLASSES', 4))

# Treinamento
batch_size = int(os.getenv('BATCH_SIZE', 32))
epochs = int(os.getenv('EPOCHS', 10))
learning_rate = float(os.getenv('LEARNING_RATE', 1e-4))

# Tamanho da imagem
img_size = (int(os.getenv('IMG_SIZE', 224)), int(os.getenv('IMG_SIZE', 224)))
