import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)


class TextInput(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()


@app.post("/process/")
async def generate_outline(input: TextInput):
    prompt = (
        f"Du bist ein Experte für die Gliederung von Wissenschaftlichen Arbeiten."
        f"Deine Aufgabe ist es eine Gliederung zu einer Wissenschaftlichen Arbeit zu schreiben."
        f"Bitte nenne nur die Gliederung ohne weitere Erläuterungen"
        f"Im Folgenden erhälst du weitere Informationen zum Thema\n\n"
        f"Benutzereingabe: {input.text}\n"
    )

    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=500, num_beams=5, early_stopping=True)
    outline = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"gliederung": outline}
