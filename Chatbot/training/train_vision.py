# training/train_vision.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import json
from sklearn.metrics import classification_report
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data/images", type=str)
parser.add_argument("--out_dir", default="../models", type=str)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch", default=32, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Transforms
# ---------------------------
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform_train)
val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# ---------------------------
# Model setup
# ---------------------------
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

best_acc = 0.0
for epoch in range(args.epochs):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += imgs.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    print(f"Epoch {epoch+1}/{args.epochs} Train loss {epoch_loss:.4f} acc {epoch_acc:.4f}")

    # validation
    model.eval()
    val_total, val_corrects = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_corrects += torch.sum(preds == labels).item()
    val_acc = val_corrects / val_total
    print(f"Validation acc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "classes": train_ds.classes
        }, os.path.join(args.out_dir, "vision_resnet18.pt"))
        print("✅ Saved best model.")

# ---------------------------
# Save class mapping
# ---------------------------
with open(os.path.join(args.out_dir, "vision_classes.json"), "w") as f:
    json.dump(train_ds.classes, f)
print("Training complete.")

# ---------------------------
# Final evaluation with metrics
# ---------------------------
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n📊 Classification Report (Validation Set):")
print(classification_report(y_true, y_pred, target_names=train_ds.classes))


