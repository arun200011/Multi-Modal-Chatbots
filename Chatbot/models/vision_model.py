# models/vision_model.py
import torch
import torch.nn as nn
from torchvision import models

def get_vision_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Example: load and test
# model = get_vision_model(38)
# output = model(torch.randn(1, 3, 224, 224))
