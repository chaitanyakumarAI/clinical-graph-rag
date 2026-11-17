# Backend: FastAPI + ingestion/query pipeline.
# Built and run by docker-compose.yml; see README "Run with Docker".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Embedding / reranker models are downloaded here on first use;
    # docker-compose mounts a volume so they survive rebuilds.
    HF_HOME=/models

WORKDIR /app

# CPU-only torch first: the default PyPI wheel drags in ~2 GB of CUDA
# libraries that a CPU container never uses. requirements.txt then sees
# torch as already satisfied.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/.cache \
    && chown -R app /models /app/.cache
USER app

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
