from fastapi.testclient import TestClient
from src.api import app
from PIL import Image
import io

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_ok():
    img = Image.new("RGB", (8, 8), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    files = {"image": ("x.png", buf, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    j = r.json()
    assert set(j.keys()) == {"label", "prob"}
