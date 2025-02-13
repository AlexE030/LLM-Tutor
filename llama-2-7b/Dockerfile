FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

WORKDIR /app

# Installiere Python, pip und benötigte Tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Stelle sicher, dass GLIBC 2.39 installiert ist
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && apt-get install -y libc6

# Setze den richtigen Library-Pfad
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Erstelle eine virtuelle Umgebung für Python
RUN python3 -m venv /app/venv

COPY . .

# Installiere Python-Abhängigkeiten
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stelle sicher, dass der Python-Pfad gesetzt ist
ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "agent_llama:app", "--host", "0.0.0.0", "--port", "8000"]
