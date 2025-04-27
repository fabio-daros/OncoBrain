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


2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API:
```bash
uvicorn main:app --reload
```

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

1. Coloque as imagens em estrutura de pastas:
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

## 🙏 Acknowledgments
Developed by Fabio Daros
