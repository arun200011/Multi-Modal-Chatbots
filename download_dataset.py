# training/download_dataset.py
import os
import zipfile
import gdown

# Google Drive URL for PlantVillage dataset (38 classes, ~1.5GB)
URL = "https://drive.google.com/uc?id=1I3x7xv4o0t7wsV1YjD8wL9hYgk5Qx0gD"
OUTPUT = "data/PlantVillage.zip"
EXTRACT_DIR = "data/PlantVillage"

def download_dataset():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(OUTPUT):
        print("📥 Downloading PlantVillage dataset...")
        gdown.download(URL, OUTPUT, quiet=False)

    print("📂 Extracting dataset...")
    with zipfile.ZipFile(OUTPUT, 'r') as zip_ref:
        zip_ref.extractall("data")

    print("✅ Dataset ready at:", EXTRACT_DIR)

if __name__ == "__main__":
    download_dataset()
