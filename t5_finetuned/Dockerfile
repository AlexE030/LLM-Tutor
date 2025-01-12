FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /t5_finetuned

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_t5_finetuned.py .
RUN python download_t5_finetuned.py

COPY agent_t5_finetuned.py agent.py
