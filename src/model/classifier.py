from typing import Dict, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch, torch.nn as nn, torch.nn.functional as F

class CatDogClassifier:
    def __init__(self, use_weights: bool = True, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        weights = MobileNet_V3_Small_Weights.DEFAULT if use_weights else None
        self.model = mobilenet_v3_small(weights=weights)
        # если весов нет (или вы offline), модель всё равно создастся
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, 2)

        self.model.eval().to(self.device)

        # нормализация берётся из weights, если они есть
        if weights is not None:
            self.transform = weights.transforms()
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        # Имена классов
        self.labels = ["cat", "dog"]

    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type=="cuda"):
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        label_idx = int(np.argmax(probs))
        return {"label": ["cat", "dog"][label_idx], "prob": round(float(probs[label_idx]), 3)}

