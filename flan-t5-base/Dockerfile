FROM python:3.9-slim

WORKDIR /app

COPY . .

# Installiere curl für Debugging
RUN apt-get update && apt-get install -y curl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "agent_gbert:app", "--host", "0.0.0.0", "--port", "8000"]

