from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image
from src.model.classifier import CatDogClassifier

app = FastAPI(title="Cats vs Dogs API")
_model = CatDogClassifier()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    content = await image.read()
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return _model.predict(img)
