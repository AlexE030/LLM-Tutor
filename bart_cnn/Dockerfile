FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bart_cnn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_bart_cnn.py .
RUN python download_bart_cnn.py

COPY agent_bart_cnn.py agent.py
