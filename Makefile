.PHONY: setup test run docker-build docker-run

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]

test:
	. .venv/bin/activate && pytest

run:
	. .venv/bin/activate && uvicorn src.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t catsdogs:latest .

docker-run:
	docker run --rm -p 8000:8000 catsdogs:latest
