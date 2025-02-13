FROM nvidia/cuda:11.8.0-base

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "agent_llama:app", "--host", "0.0.0.0", "--port", "8000"]

