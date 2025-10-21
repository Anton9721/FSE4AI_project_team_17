from typing import Dict, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models

class CatDogClassifier:
    def __init__(self):
        # Загружаем предобученную лёгкую модель
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v3_small(pretrained=True)
        # Меняем последний слой под 2 класса
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, 2)
        self.model.eval()
        self.model.to(self.device)

        # Препроцессинг изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Имена классов
        self.labels = ["cat", "dog"]

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, float | str]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs))
        return {"label": self.labels[label_idx], "prob": round(float(probs[label_idx]), 3)}
