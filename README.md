# OncoBrain

OncoBrain é um projeto de Inteligência Artificial focado em **análise de imagens tumorais** usando Vision Transformers (ViT) adaptados para aplicações médicas.

Desenvolvido para ser a API de inferência do OncoPixel, o OncoBrain realiza:
- Upload de imagens tumorais
- Análise através de um modelo Transformer treinado
- Classificação multiclasse de tipos tumorais

---

## Tecnologias Utilizadas
- **Python 3.12**
- **FastAPI** (servidor de API)
- **PyTorch** (modelo Transformer e treino)
- **Torchvision** (datasets e augmentations)
- **PIL** (manipulação de imagens)

---

## 🚀 Como rodar localmente

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/OncoBrain.git
cd OncoBrain
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Rode a API:
```bash
uvicorn main:app --reload
```

4. Acesse a documentação interativa Swagger:
```
http://127.0.0.1:8000/docs
```

5. Teste o upload de uma imagem tumor/célula!

---

## 🏋️️ Estrutura do Projeto

```bash
OncoBrain/
├── main.py                # FastAPI principal (endpoints)
├── model/
│   ├― transformer_model.py   # Modelo ViT para inferência
├── training/
│   ├― train.py             # Script de treinamento
│   ├― dataset.py           # Dataset customizado
│   ├― utils.py             # Funções auxiliares (salvamento)
│   └― config.py            # Configurações gerais
├── saved_models/           # Modelos treinados salvos (.pth)
├── requirements.txt        # Dependências
└── README.md               # Documentação (este arquivo)
```

---

## 🔧 Como treinar um novo modelo

1. Coloque as imagens em estrutura de pastas:
```
path/to/your/train/
   └── normal/
   └── benign/
   └── malignant/
   └── carcinoma/
```

2. Ajuste `training/config.py` com o caminho correto:
```python
train_data_dir = "path/to/your/train"
```

3. Rode o treino:
```bash
python training/train.py
```

4. O modelo treinado será salvo automaticamente em `/saved_models/`.

---

## 📈 Roadmap futuro
- [ ] Treinar com BreakHis ou DDSM datasets reais
- [ ] Implementar Augmentations avançadas
- [ ] Adicionar inferência de "confidence score"
- [ ] Deploy online em servidor cloud (AWS / GCP / Render)

---

## 🙏 Agradecimentos
Desenvolvido por Fabio Daros com suporte da IA do ChatGPT. 🚀
