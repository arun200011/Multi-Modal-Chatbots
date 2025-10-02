# training/train_audio.py
import os, csv, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from models.speech_model import SpeechClassifier
from sklearn.metrics import classification_report

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, classes, sr=22050, n_mfcc=40, max_len=200):
        self.rows = pd.read_csv(csv_path).to_dict(orient="records")
        self.audio_dir = audio_dir
        self.classes = classes
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = os.path.join(self.audio_dir, r['filename'])
        y, _ = librosa.load(path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc).T
        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], self.n_mfcc))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len, :]
        label = self.classes.index(r['label'])
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ------------------ main ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="../data/audio_labels.csv")
parser.add_argument("--audio_dir", default="../data/audio")
parser.add_argument("--out_dir", default="../models")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

# load classes
if os.path.exists("../models/vision_classes.json"):
    with open("../models/vision_classes.json") as f:
        classes = json.load(f)
else:
    df = pd.read_csv(args.csv)
    classes = sorted(df['label'].unique().tolist())

print("Classes:", classes)
ds = AudioDataset(args.csv, args.audio_dir, classes)
loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechClassifier(input_dim=40, num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# training loop
for epoch in range(args.epochs):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        _, preds = torch.max(out, 1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    print(f"Epoch {epoch+1}/{args.epochs} acc {correct/total:.4f}")

# save
os.makedirs(args.out_dir, exist_ok=True)
torch.save({"model_state": model.state_dict(), "classes": classes},
           os.path.join(args.out_dir, "audio_speech_lstm.pt"))
print("Saved audio model.")

# evaluation (optional, for metrics)
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
print(classification_report(y_true, y_pred, target_names=classes))

