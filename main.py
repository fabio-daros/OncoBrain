import os
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from model.faster_rcnn_model import create_faster_rcnn_model
from torchvision import transforms
import torch
from model.transformer_model import load_model, predict_image
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="OncoBrain API",
    description="API responsible for analyzing tumor images for Oncopixel",
    version="1.0.0"
)

# Carregar modelo
model = load_model()

# Faster R-CNN (detecção)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Primeiro define o diretório
model_dir = "./saved_models/faster_rcnn_model_checkpoints"

# Depois usa
checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
if not checkpoints:
    raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")

# Ordena os checkpoints por número de epoch
checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

latest_checkpoint = os.path.join(model_dir, checkpoints[-1])
print(f"Loading Faster R-CNN model from: {latest_checkpoint}")

faster_rcnn_model = create_faster_rcnn_model(num_classes=7)
faster_rcnn_model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
faster_rcnn_model.to(device)
faster_rcnn_model.eval()

detection_transform = transforms.ToTensor()

# ASCII art
ASCII_BRAIN = """
                                 .-=+++=-::..::..
                          .-+===+*#*+=+*###**#*====-.
                       :=++==+#%#+==+*%%#+===+**+++++=-:.
                     -++=+*#***===*%%**++=-*%##*##*==*+-=-.
                  .:+*+:=%%#+-::*#%%*-+#%%#*%%%#+=+=:+#*=-:-:
                .=***##**%%*=:-*%+#*+%+###%******+:-++**#+-=+:.
               .=%++%%*##*+=-=%@**%+%#+@+#%=%+*##+:#@%*++*=-*+-::
             .=+*#=*%%*++====**+-*-=*:#==%-#*+%#+==***#+=---=+*=-:.
            .+##**#%%@#*#+=*==**#***+*++++=+=+*+*%%%#*+#*=-**=-+=-:.
           .-+%++%#**####=+**%@@@%%%%%%%%%%##==*=*#%#+=##+-+#+-:-::-.
          :==**+#%%###*#+=*#@#%%%%%%@%@%%@@@%#=*:=+#+*=+*=-=*+-:=-:::
         :+-+**%**#%@*##=+##-+-#-#++*=-+%++*%**=:+#*##+:--:=*+:-**=:::
         -+==-#%++#%**#+=+%-**=+.=-*:#**==*.#**:=*#*#*-:=**==*==#*=-:-:
        .+#%+=#%*+%%*#*=+#*:#-*-+.===%##:%+-#*=:+****=:-+*#*--+++=-:---:
       .=*##+@%#+*%###+=*%#++**+%=#==-++:+:***:-+#***=:+#*++=-==--::::::
       .=-**=#%%%@%*#*=++%@@@%%@%@%%%%%%#*##+=:=***#*=:=#*+=-:::::-=-::.
       -+=+%=-++**#%%#++-=%%%%%%%%%%%@%%@%#+*:-+****+=--+---:-++=--=--:-.
       =#*+%##%#**%%*++======+++*+*****##+-*=:***+++--::-=++++**+==-:-:-:
       -*#==+##%%%#*=-*#--+:+:-=:=:-------==:=**+=-::::=*#%#*+=---::::::.
       .=*#*==+##+=-:=%%+%**@=%#+%=#*-#-=*++++=::==++++*#*+===-::::::-::
        .-+##*+*##*=:+%#+%+%#+@+%#+@+*%=%#*+=-::+#%#**+===-::::------:::
         :-==+****++--+*#***++##%#+#++*+*=-::-+*##*=---::::======-:-::-:
          .:::-------::---==-=+****=====--:-*%%#*+=-.:-=--+++----:--::.
             .:------:::-=+++=---===---::-=#%%*=--:-+****++=-::-:-:::.          
                  :::::-::-------::::::-=*##*+=::=++*+=---------::::.
                   .-------:::::.::::::-=****=--====-:::-:::::::::::
                       ..::::..   .::-::-==+++==-::---::::::-----::
                                     ::::::::::::::::::::---==---:.
                                      :--:::::::::::::---===---:. 
                                       :-----:::.::--------::..   
                                        :==----.    .....         
                                         -==--:                 
                                         -=+=--.                
                                       ..-+++--.--::.           
                                    :=-==-+**+==++*##=:.        
                                            ...:::::.
"""


@app.get("/", response_class=HTMLResponse)
def read_root():
    return f"""
    <html>
        <head>
            <title>OncoBrain</title>
            <style>
                body {{
                    background-color: #000;
                    color: #0f0;
                    font-family: monospace;
                    text-align: center;
                    padding: 30px;
                }}
                a {{
                    color: #0f0;
                    text-decoration: none;
                    margin: 0 10px;
                    font-weight: bold;
                }}
                a:hover {{
                    color: #7fff00;
                    text-decoration: underline;
                }}
                .nav {{
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="nav">
                <a href="/docs" target="_blank">📘 Swagger Docs</a>
                <a href="/redoc" target="_blank">📗 ReDoc</a>
            </div>

            <h1>Welcome to OncoBrain!</h1>
            <p>API responsible for analyzing tumor images for Oncopixel.</p>
            <pre style="white-space: pre-wrap; text-align: left; display: inline-block; margin-top: 20px;">{ASCII_BRAIN}</pre>
        </body>
    </html>
    """


@app.post("/analyze/")
async def analyze_image(
        file: UploadFile = File(...),
        confidence_threshold: float = Form(50.0)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        result = predict_image(model, image, confidence_threshold=confidence_threshold)

        if result["class"] is None:
            return JSONResponse(content=result)

        return JSONResponse(content={
            "prediction": {
                "class": result["class"],
                "confidence": result["confidence"]
            }
        })

    except Exception as e:
        print("ERROR in /analyze/:", e)
        return JSONResponse(status_code=500, content={"error": f"Failed to analyze image: {str(e)}"})


OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/detect/")
async def detect_cells(
        file: UploadFile = File(...),
        confidence_threshold: float = Form(0.2)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = detection_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = faster_rcnn_model(img_tensor)

        output = outputs[0]

        detections = []

        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            if score >= confidence_threshold:
                detections.append({
                    "box": [round(x.item(), 2) for x in box],
                    "label": int(label.item()),
                    "score": round(score.item(), 4)
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        print("ERROR in /detect/:", e)
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {str(e)}"})


@app.get("/heartbeat/")
def heartbeat():
    return {"status": "ok"}


app.mount("/static_output", StaticFiles(directory="output_images"), name="static_output")
