import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token


class TextInput(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()


@app.post("/process/")
async def generate_outline(input: TextInput):
    prompt = (
        f"Du bist ein Experte für das erstellen von Zitaten in Wissenschaftlichen Arbeiten."
        f"Deine Aufgabe ist es ein Zitat nach einer definierten Zitationsart zu erstellen."
        f"Bitte gebe nur das Zitat aus, ohne weitere Erläuterungen"
        f"Hier hast du die Eingabe des Benutzers\n\n"
        f"Benutzereingabe: {input.text}\n"
    ) # TODO: This is not a proper prompt, but just for Testing for basic integration of Zephyr

    print(prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=500, num_beams=5, early_stopping=True)

    outline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"citation": outline}
