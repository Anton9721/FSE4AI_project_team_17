FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install -U pip && pip install .
# Предскачиваем веса MobileNetV3 (чтобы не качались при каждой сборке)
RUN python - <<'PY'
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
PY
COPY src /app/src
RUN useradd -m app && chown -R app:app /app
USER app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
