FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY requirements.txt requirements-classificador.txt requirements-inference.txt ./
# Instala a distribuição CPU antes do restante, evitando bibliotecas CUDA.
RUN python -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-inference.txt gunicorn==23.0.0 \
    && python -m pip check

COPY backend/ backend/
COPY ml/ ml/
COPY frontend/ frontend/
COPY --chown=app:app models/mdeberta-ensemble/ models/mdeberta-ensemble/
COPY scripts/check_model.py scripts/smoke_test.py scripts/
COPY deploy/healthcheck.py deploy/healthcheck.py
COPY run.py ./

USER app
# Confere também que o usuário do servidor consegue ler os pesos (podem ter modo 600).
RUN python scripts/check_model.py --verify-hashes

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "run:app"]
