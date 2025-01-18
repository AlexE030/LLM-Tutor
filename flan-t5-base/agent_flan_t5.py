from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

MODEL_NAME = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()

@app.post("/summarize/")
async def summarize(input: TextInput):

    inputs = tokenizer(input.text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, length_penalty=2.0)
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"original_text": input.text, "summary": summarized_text}

