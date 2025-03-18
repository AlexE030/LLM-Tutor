FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_mistral.py .
COPY llm_client.py .

CMD ["uvicorn", "agent_mistral:app", "--host", "0.0.0.0", "--port", "8000"]