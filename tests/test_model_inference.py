from PIL import Image
import numpy as np
from src.model.classifier import CatDogClassifier

def test_model_inference_smoke():
    clf = CatDogClassifier(use_weights=False)
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    out = clf.predict(img)
    assert set(out.keys()) == {"label", "prob"}
    assert out["label"] in {"cat", "dog"}
    assert 0.0 <= out["prob"] <= 1.0

def test_numpy_input():
    clf = CatDogClassifier(use_weights=False)
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 150
    out = clf.predict(arr)
    assert isinstance(out["label"], str)
