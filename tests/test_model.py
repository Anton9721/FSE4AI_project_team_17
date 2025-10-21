from PIL import Image
from src.model.classifier import CatDogClassifier

def test_predict_basic():
    clf = CatDogClassifier()
    img = Image.new("RGB", (8, 8), color=(128, 128, 128))
    out = clf.predict(img)
    assert set(out.keys()) == {"label", "prob"}
    assert 0.0 <= out["prob"] <= 1.0
