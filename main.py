from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model.transformer_model import load_model, predict_image
from PIL import Image
import io

app = FastAPI(
    title="OncoBrain API",
    description="API responsible for analyzing tumor images for Oncopixel",
    version="1.0.0"
)

# Carregar o modelo ao iniciar a aplicação
model = load_model()

@app.get("/")
def read_root():
    return {"message": "Welcome to OncoBrain! API responsible for analyzing tumor images for Oncopixel."}

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        result = predict_image(model, image)

        print("Predicted:", result)

        return JSONResponse(content={"prediction": result})

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(status_code=500, content={"error": f"Failed to analyze image: {str(e)}"})


