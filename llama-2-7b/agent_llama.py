from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


class UserInput(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()


@app.post("/generate_outline/")
async def generate_outline(input: UserInput):
    prompt = (
        f"Du bist ein Experte für die Gliederung von Wissenschaftlichen Arbeiten."
        f"Deine Aufgabe ist es eine Gliederung zu einer Wissenschaftlichen Arbeit zu schreiben."
        f"Bitte nenne nur die Gliederung ohne weitere Erläuterungen"
        f"Ih Folgenden erhälst du weitere Informationen zum Thema\n\n"
        f"Benutzereingabe: {input.text}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=500, num_beams=5, early_stopping=True)

    # Dekodierung des generierten Texts
    outline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"gliederung": outline}
