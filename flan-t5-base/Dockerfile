# Use Python slim image
FROM python:3.9-slim

# Install basic tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl

# Upgrade pip and set up NumPy
RUN pip install --upgrade pip
RUN pip install "numpy<2"

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY agent_flan_t5.py /app/agent_flan_t5.py

# Set entry point for the container
ENTRYPOINT ["python", "/app/agent_flan_t5.py"]

