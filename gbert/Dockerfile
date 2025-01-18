# GBERT Dockerfile
FROM python:3.9-slim

# Install basic packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl


RUN pip install "numpy<2"

# Install Python packages
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent script
COPY agent_gbert.py /app/agent_gbert.py

# Set up execution entrypoint
CMD ["python", "/app/agent_gbert.py"]
