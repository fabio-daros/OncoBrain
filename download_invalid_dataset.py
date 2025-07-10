import requests
import os
from tqdm import tqdm

DEST_DIR = "dataset/bmt"
output_dir = "dataset/bmt/INVALID"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(200), desc="Downloading images"):
    try:
        response = requests.get("https://picsum.photos/512", timeout=5)
        if response.status_code == 200:
            with open(os.path.join(output_dir, f"invalid_{i+1}.jpg"), "wb") as f:
                f.write(response.content)
    except Exception as e:
        print(f"Error when downloading image {i+1}: {e}")
