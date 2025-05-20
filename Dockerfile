FROM python:3.9

WORKDIR /app

COPY . /app

RUN mkdir -p /tmp/.cache/gdown && chmod -R 777 /tmp/.cache

ENV GDOWN_CACHE_DIR=/tmp/.cache/gdown

RUN pip install --no-cache-dir -r requirements.txt

# Exposer le nouveau port
EXPOSE 8000

# Lancer uvicorn sur le port 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
