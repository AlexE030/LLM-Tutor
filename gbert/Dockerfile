FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /gbert

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_gbert.py .
RUN python download_gbert.py

COPY agent_gbert.py agent.py
