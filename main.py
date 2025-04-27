from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model.transformer_model import load_model, predict_image
from PIL import Image
import io

app = FastAPI(
    title="OncoBrain API",
    description="API responsável por analisar imagens tumorais para o OncoPixel",
    version="1.0.0"
)

# Carregar o modelo ao iniciar a aplicação
model = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to OncoBrain!"}

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Ler o conteúdo da imagem
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Fazer a predição
        prediction = predict_image(model, image)

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
