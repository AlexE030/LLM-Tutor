from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

MODEL_NAME = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

DISTILLBERT_API_URL = "http://distillbert_api:8000/process/"
GBERT_API_URL = "http://gbert_api:8000/process/"

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()

@app.post("/process/")
async def process_text(input: TextInput):

    prompt = input.text

    try:
        _, text = prompt.split(": ", 1)
    except ValueError:
        text = prompt

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, length_penalty=2.0)
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"generated_text": summarized_text}


