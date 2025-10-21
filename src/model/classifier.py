from typing import Dict, Union
from PIL import Image
import numpy as np

class CatDogClassifier:
    def __init__(self):
        pass

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        return np.array(img)

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, float | str]:
        arr = np.array(image) if not isinstance(image, np.ndarray) else image
        prob_dog = float(arr.mean() / 255.0) if arr.size > 0 else 0.5
        label = "dog" if prob_dog >= 0.5 else "cat"
        prob = prob_dog if label == "dog" else 1.0 - prob_dog
        return {"label": label, "prob": round(prob, 3)}
