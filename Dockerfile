FROM python:3.11-slim

LABEL maintainer="Gabriel Lafis"
LABEL description="COVID-19 ML Prediction Pipeline"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--state", "SP", "--model-type", "all"]
