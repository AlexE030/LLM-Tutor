FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bert_multilinugal

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_bert_multilingual.py .
RUN python download_bert_multilingual.py

COPY agent_bert_multilingual.py agent.py
