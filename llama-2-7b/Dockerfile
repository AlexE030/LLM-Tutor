FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/venv

COPY . .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "agent_llama:app", "--host", "0.0.0.0", "--port", "8000"]
