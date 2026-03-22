FROM python:3.10-slim

LABEL maintainer="Mohammad Thabet Hassan"
LABEL description="XAI-IDS: Explainable AI Intrusion Detection System"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs/figures outputs/models outputs/reports outputs/logs data/raw

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "run_pipeline.py", "--sample-size", "10000", "--skip-explain"]
