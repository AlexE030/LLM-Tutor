from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

MODEL_NAME = "deepset/gbert-base"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()

@app.post("/predict/")
async def predict(input: TextInput):

    tokens = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokens)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return {"text": input.text, "predicted_class": predicted_class}

