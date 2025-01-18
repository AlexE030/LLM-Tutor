# DistilBERT Dockerfile
FROM python:3.10-slim

# Install basic packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl

RUN pip install --upgrade pip

# Install Python packages
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent script
COPY agent_distillbert.py /app/agent_distillbert.py

# Set up execution entrypoint
CMD ["python", "/app/agent_distillbert.py"]
