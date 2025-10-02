# training/train_fusion.py
import os, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse, librosa, pandas as pd

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fusion_csv", default="../data/fusion.csv")  # columns: image,audio,text,label
parser.add_argument("--out_dir", default="../models")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Vision backbone
# ---------------------------
vision_ckpt = os.path.join(args.out_dir, "vision_resnet18.pt")
vision_classes = json.load(open(os.path.join(args.out_dir, "vision_classes.json")))
resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Identity()  # keep 512-d features
resnet.load_state_dict(torch.load(vision_ckpt, map_location=device)["model_state"], strict=False)
resnet = resnet.to(device).eval()

# ---------------------------
# Audio backbone
# ---------------------------
from training.train_audio import CNNLSTM
audio_ckpt = os.path.join(args.out_dir, "audio_cnnlstm.pt")
audio_dict = torch.load(audio_ckpt, map_location=device)
audio_model = CNNLSTM(n_mfcc=40, hidden_dim=128, num_classes=len(vision_classes))

# Modify forward to return hidden representation instead of logits
def audio_forward_features(self, x):
    x = x.unsqueeze(1)  # (b,1,t,f)
    x = self.cnn(x)
    b,c,t,f = x.size()
    x = x.permute(0,2,1,3).reshape(b,t,c*f)
    out, (h, _) = self.lstm(x)
    return h[-1]  # last hidden state (batch, hidden_dim)

audio_model.forward_features = audio_forward_features.__get__(audio_model, CNNLSTM)
audio_model.load_state_dict(audio_dict["model_state"], strict=False)
audio_model = audio_model.to(device).eval()

# ---------------------------
# Text backbone
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

# ---------------------------
# Dataset
# ---------------------------
df = pd.read_csv(args.fusion_csv)
label_map = sorted(df['label'].unique().tolist())
label2idx = {l: i for i, l in enumerate(label_map)}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class FusionDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # image
        img = Image.open(row['image']).convert("RGB")
        img = transform(img)
        # audio
        y, _ = librosa.load(row['audio'], sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40).T
        if mfcc.shape[0] < 200:
            pad = np.zeros((200-mfcc.shape[0], 40))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:200,:]
        # text
        text = str(row['text']) if not pd.isna(row['text']) else ""
        label = label2idx[row['label']]
        return img, mfcc.astype(np.float32), text, label

dataset = FusionDataset(df)
loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)

# ---------------------------
# Fusion model
# ---------------------------
class FusionMLP(nn.Module):
    def __init__(self, v_dim=512, t_dim=768, a_dim=128, hidden=256, num_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(v_dim + t_dim + a_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, v, t, a):
        x = torch.cat([v,t,a], dim=1)
        return self.mlp(x)

fusion = FusionMLP(v_dim=512, t_dim=768, a_dim=128, hidden=256, num_classes=len(label_map)).to(device)
opt = optim.Adam(fusion.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(args.epochs):
    fusion.train()
    total, correct = 0, 0
    for img, mfcc, text, label in loader:
        img, mfcc, label = img.to(device), torch.tensor(mfcc).to(device), label.to(device)

        # vision embedding
        with torch.no_grad():
            v_emb = resnet(img)

        # audio embedding
        with torch.no_grad():
            a_emb = audio_model.forward_features(mfcc)

        # text embedding
        enc = tokenizer(list(text), padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            t_emb = text_encoder(**enc).last_hidden_state[:,0,:]

        # flatten
        v_emb = v_emb.view(v_emb.size(0), -1)
        a_emb = a_emb.view(a_emb.size(0), -1)
        t_emb = t_emb.view(t_emb.size(0), -1)

        # fusion forward
        logits = fusion(v_emb, t_emb, a_emb)
        loss = crit(logits, label)

        opt.zero_grad(); loss.backward(); opt.step()
        _, preds = torch.max(logits, 1)
        total += label.size(0); correct += (preds==label).sum().item()
    print(f"Epoch {epoch+1}: acc={correct/total:.4f}")

# ---------------------------
# Save model
# ---------------------------
os.makedirs(args.out_dir, exist_ok=True)
torch.save({"model_state": fusion.state_dict(), "label_map": label_map}, os.path.join(args.out_dir, "fusion_mlp.pt"))
print("✅ Saved fusion model.")

