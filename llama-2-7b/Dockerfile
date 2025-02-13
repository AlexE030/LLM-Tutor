FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "agent_llama:app", "--host", "0.0.0.0", "--port", "8000"]

