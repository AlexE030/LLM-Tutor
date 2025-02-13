from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()

@app.post("/process/")
async def process_text(input: TextInput):
    tokens = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokens)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return {"sprachliche_qualitaet": predicted_class}
