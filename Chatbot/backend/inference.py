# backend/inference.py
import torch, json, os
from torchvision import transforms, models
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import librosa

from models.speech_model import SpeechClassifier   # ✅ use your LSTM-only model

class PlantGuard:
    def __init__(self, models_dir="models", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device set to use {self.device}")

        # --- vision ---
        v_ckpt = os.path.join(models_dir, "vision_resnet18.pt")
        classes_file = os.path.join(models_dir, "vision_classes.json")
        if os.path.exists(v_ckpt) and os.path.exists(classes_file):
            cls = json.load(open(classes_file))
            self.v_classes = cls
            resnet = models.resnet18(pretrained=False)
            resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(cls))
            resnet.load_state_dict(torch.load(v_ckpt, map_location=self.device)["model_state"])
            self.vision = resnet.to(self.device).eval()
        else:
            print("⚠️ Vision model not found. Please train it first.")
            self.vision = None
            self.v_classes = []

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # --- audio ---
        a_ckpt = os.path.join(models_dir, "audio_speech_lstm.pt")   # ✅ renamed
        if os.path.exists(a_ckpt):
            a_dict = torch.load(a_ckpt, map_location=self.device)
            self.audio_classes = a_dict["classes"]
            self.audio_model = SpeechClassifier(input_dim=40, num_classes=len(self.audio_classes))
            self.audio_model.load_state_dict(a_dict["model_state"])
            self.audio_model = self.audio_model.to(self.device).eval()
        else:
            print("⚠️ Audio model not found. Please train it first.")
            self.audio_model = None
            self.audio_classes = []

        # --- text QA pipeline ---
        text_dir = os.path.join(models_dir, "text_qa")
        if os.path.exists(text_dir):
            self.qa = pipeline("question-answering",
                               model=text_dir,
                               tokenizer=text_dir,
                               device=0 if self.device=="cuda" else -1)
        else:
            print("⚠️ Using default DistilBERT QA model")
            self.qa = pipeline("question-answering",
                               model="distilbert-base-uncased-distilled-squad",
                               tokenizer="distilbert-base-uncased-distilled-squad",
                               device=0 if self.device=="cuda" else -1)

        # --- fusion ---
        fusion_ckpt = os.path.join(models_dir, "fusion_mlp.pt")
        if os.path.exists(fusion_ckpt):
            f_dict = torch.load(fusion_ckpt, map_location=self.device)
            self.label_map = f_dict["label_map"]
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(512+768+len(self.audio_classes), 256),   # ✅ audio logits dim = num_classes
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, len(self.label_map))
            ).to(self.device)
            self.fusion.load_state_dict(f_dict["model_state"])
            self.fusion.eval()

            # load encoders for embeddings
            self.resnet_embed = models.resnet18(pretrained=False)
            self.resnet_embed.fc = torch.nn.Identity()
            self.resnet_embed.load_state_dict(torch.load(v_ckpt, map_location=self.device)["model_state"], strict=False)
            self.resnet_embed = self.resnet_embed.to(self.device).eval()

            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device).eval()
        else:
            print("⚠️ Fusion model not found. Please train it with train_fusion.py")
            self.fusion = None

    # --- predict_image() (unchanged) ---
    def predict_image(self, image_bytes):
        if self.vision is None:
            return {"error": "Vision model not available"}
        img = Image.open(image_bytes).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.vision(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {"label": self.v_classes[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}

    # --- predict_audio() (updated for SpeechClassifier) ---
    def predict_audio(self, audio_path):
        if self.audio_model is None:
            return {"error": "Audio model not available"}
        y, _ = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40).T
        if mfcc.shape[0] < 200:
            pad = np.zeros((200-mfcc.shape[0], 40)); mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:200,:]
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.audio_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {"label": self.audio_classes[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}

    # --- predict_fusion() (updated) ---
    def predict_fusion(self, image_file, audio_path, text):
        if self.fusion is None:
            return {"error": "Fusion model not available. Please train it first."}

        # vision embedding
        img = Image.open(image_file).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v_emb = self.resnet_embed(x)  # (1,512)

        # audio embedding (use logits as features)
        y, _ = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40).T
        if mfcc.shape[0] < 200:
            pad = np.zeros((200-mfcc.shape[0], 40)); mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:200,:]
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a_emb = self.audio_model(mfcc_tensor)  # (1, num_classes)

        # text embedding
        enc = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            t_emb = self.text_encoder(**enc).last_hidden_state[:,0,:]  # (1,768)

        # fuse all
        with torch.no_grad():
            feats = torch.cat([v_emb, t_emb, a_emb], dim=1)
            logits = self.fusion(feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {"label": self.label_map[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}



