import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from model.faster_rcnn_model import create_faster_rcnn_model
from torchvision import transforms
import torch
from model.transformer_model import load_model, predict_image
from PIL import Image
import io
from fastapi.staticfiles import StaticFiles
import subprocess

app = FastAPI(
    title="OncoBrain API",
    description="API responsible for analyzing tumor images for Oncopixel",
    version="1.0.0"
)

# Detecta se GPU está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🟢 Using device: {device}")

# Carregar modelo Transformer (ViT)
model = load_model()

# Carregar modelo Faster R-CNN (detecção)
model_dir = "./saved_models/faster_rcnn_model_checkpoints"
checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
if not checkpoints:
    raise FileNotFoundError(f"No .pth checkpoints found in {model_dir}")

# Ordena checkpoints por número de epoch
checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
latest_checkpoint = os.path.join(model_dir, checkpoints[-1])
print(f"📦 Loading Faster R-CNN model from: {latest_checkpoint}")

faster_rcnn_model = create_faster_rcnn_model(num_classes=7)
checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
faster_rcnn_model.load_state_dict(checkpoint, strict=False)
faster_rcnn_model.to(device)
faster_rcnn_model.eval()
print("✅ Faster R-CNN loaded and moved to", device)

# Transforms
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
    gpu_status = get_gpu_status()
    gpu_html = f"""
        <h2>CPU/GPU Status:</h2>
        <p><b>Active:</b> {'Yes' if gpu_status['gpu_active'] else 'No'}</p>
        <p><b>GPU Name:</b> {gpu_status['gpu_name']}</p>
        <p><b>Total Memory:</b> {gpu_status['total_memory_gb']} GB</p>
        <p><b>Allocated Memory:</b> {gpu_status['allocated_memory_gb']} GB</p>
        <p><b>Reserved Memory:</b> {gpu_status['reserved_memory_gb']} GB</p>
        """
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
                .gpu-status {{
                    position: absolute;
                    top: 300px;
                    right: 50px;
                    text-align: left;
                    background-color: rgba(0, 0, 0, 0.7);
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid #0f0;
                    box-shadow: 0 0 15px #0f0;
                    max-width: 300px;
                    cursor: move;
                }}
                body {{
                    overflow: hidden;
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
            <div class="gpu-status">
                {gpu_html}
            </div>
            <pre style="white-space: pre-wrap; text-align: left; display: inline-block; margin-top: 20px;">{ASCII_BRAIN}</pre>
             <script>
                const gpuStatus = document.querySelector('.gpu-status');
                let isDragging = false;
                let offsetX = 0;
                let offsetY = 0;
            
                gpuStatus.addEventListener('mousedown', (e) => {{
                    isDragging = true;
                    offsetX = e.clientX - gpuStatus.offsetLeft;
                    offsetY = e.clientY - gpuStatus.offsetTop;
                    gpuStatus.style.zIndex = 1000;
                }});
            
                    document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    let newLeft = e.clientX - offsetX;
                    let newTop = e.clientY - offsetY;
        
                    const maxLeft = window.innerWidth - gpuStatus.offsetWidth;
                    const maxTop = window.innerHeight - gpuStatus.offsetHeight;
        
                    newLeft = Math.max(0, Math.min(newLeft, maxLeft));
                    newTop = Math.max(0, Math.min(newTop, maxTop));
        
                    gpuStatus.style.left = newLeft + 'px';
                    gpuStatus.style.top = newTop + 'px';
                }}
            }});
            
                document.addEventListener('mouseup', () => {{
                    isDragging = false;
                }});
            </script>
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


# Função para obter status da GPU
def get_gpu_status():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)

        return {
            "gpu_active": True,
            "gpu_name": gpu_name,
            "total_memory_gb": round(total_mem, 2),
            "allocated_memory_gb": round(allocated_mem, 2),
            "reserved_memory_gb": round(reserved_mem, 2),
            "driver_version": get_driver_version()
        }
    else:
        return {
            "gpu_active": False,
            "gpu_name": None,
            "total_memory_gb": 0,
            "allocated_memory_gb": 0,
            "reserved_memory_gb": 0,
            "driver_version": None
        }


def get_driver_version():
    try:
        result = subprocess.run(['nvidia-smi', '--query-driver', '--format=csv,noheader'], stdout=subprocess.PIPE)
        return result.stdout.decode().strip()
    except Exception:
        return "Unknown"
