# OncoBrain

OncoBrain is an Artificial Intelligence project focused on **tumor image analysis** using Vision Transformers (ViT) adapted for medical applications.

Developed to serve as the inference API for OncoPixel, OncoBrain performs:
- Upload of tumor images
- Analysis through a trained Transformer model
- Multiclass classification of tumor types

---

## Technologies Used
- **Python 3.12**
- **FastAPI** (API server)
- **PyTorch** (Transformer model and training)
- **Torchvision** (datasets and augmentations)
- **PIL** (image processing)

---

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/your-username/OncoBrain.git
cd OncoBrain
uvicorn main:app --relo


2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. ## How to run
```bash
git clone https://github.com/seu-usuario/OncoBrain.git
cd OncoBrain
docker-compose up --build

4. Access the interactive Swagger documentation:
```
http://127.0.0.1:8000/docs
```

5. Test uploading a tumor/cell image!

---

## 🏋️️ Project Structure

```bash
OncoBrain/
├── main.py                # Main FastAPI app (endpoints)
├── model/
│   ├― transformer_model.py   # ViT model for inference
├── training/
│   ├― train.py             # Training script
│   ├― dataset.py           # Custom dataset loader
│   ├― utils.py             # Helper functions (saving, etc.)
│   └― config.py            # General configurations
├── saved_models/           # Saved trained models (.pth)
├── requirements.txt        # Dependencies
└── README.md               # Documentation (this file)
```

---

## 🔧 How to Train a New Model

1. Put the images in folder structure:
```
path/to/your/train/
   └── normal/
   └── benign/
   └── malignant/
   └── carcinoma/
```

2. Update the training/config.py file with the correct path:
```python
train_data_dir = "path/to/your/train"
```

3. Start the training:
```bash
python training/train.py
```

4. The trained model will be automatically saved in /saved_models/.

---

## 🧠 Técnicas Computacionais e Algoritmos Utilizados no OncoBrain

### ✅ Pré-processamento das Imagens

- Redimensionamento das imagens para **224×224 pixels** (dimensão nativa do ViT-B/16).
- Aplicação de **Data Augmentation leve**, incluindo:
  - Rotação aleatória de até **10 graus**.
  - Variação de **brilho** e **contraste**.
  - Conversão para tensor normalizado.

---

### ✅ Modelo de IA Aplicado

- **Vision Transformer (ViT-B/16)** pré-treinado no **ImageNet-1K**, disponibilizado pela **TorchVision**:
  - Arquitetura baseada em Transformer com **patch size 16**.
  - Entrada de imagens de **224×224 pixels**.

- **Técnica de Aprendizado**:
  - **Transfer Learning com Fine-Tuning parcial**:
    - Base do modelo congelada (opcional).
    - Head ajustado para **4 classes personalizadas**:
      - Benign
      - Carcinoma
      - Malignant
      - Normal

---

### ✅ Linguagens e Bibliotecas Utilizadas

- **Python 3.11**
- **torch**
- **torchvision**
- **numpy**
- **matplotlib**
- **scikit-learn**

---

### ✅ Treinamento

- Conjunto de dados com imagens organizadas em 4 categorias:
  - `benign`, `carcinoma`, `malignant`, `normal`.
- Divisão dos dados:
  - **80%** para treino
  - **20%** para validação
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Épocas**: 50
- **Otimizador**: Adam
- **Função de perda**: CrossEntropyLoss
- Execução em **CPU ou GPU (CUDA)**, se disponível.

---

### ✅ Avaliação dos Casos

- Monitoramento da **acurácia** em treino e validação a cada época.
- Salvamento do **melhor modelo** com base na melhor acurácia de validação.
- Geração de gráficos de **perda e acurácia**, salvos em **PNG**.
- As classes utilizadas neste estudo foram baseadas na classificação Bethesda para citologia cervical:
- NILM: ausência de lesões pré-malignas (normal)
- LSIL: lesão escamosa de baixo grau, associada ao HPV, geralmente autolimitada

HSIL: lesão escamosa de alto grau, potencialmente precursora do câncer cervical
---

### ✅ Infraestrutura e Deploy

- Ambiente containerizado com **Docker** e **Docker Compose**.
- Parâmetros de configuração definidos via **variáveis de ambiente (.env)**.
- Suporte para deploy **local ou em nuvem**.

---

### 🚀 Oportunidades Futuras

1. Divisão em **tiles** de WSIs para análise em alta resolução.
2. Filtragem por densidade celular em tiles.
3. Aplicação de **Contrastive Learning** com **CLIP** ou modelos similares.
4. Fine-Tuning completo do modelo, não apenas do head.
5. Execução em **FP16** em **GPUs de alta performance** (ex: RTX A6000).
6. Organização dos casos em **bags** ou agrupamentos clínicos.
7. Cálculo de **Anomaly Score** baseado na distribuição das predições.
8. Uso de modelos maiores, como **ViT-L/14@336px**.

---

> Projeto em desenvolvimento.


## 🙏 Acknowledgments
Developed by Fabio Daros
