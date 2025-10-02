# utils/vision_utils.py
import cv2
import numpy as np

def preprocess_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Batch dimension
    return img
