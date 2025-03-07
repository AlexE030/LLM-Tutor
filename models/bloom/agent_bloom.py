from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Verwende das deutsch optimierte Modell
MODEL_NAME = "malteos/bloom-6b4-clp-german"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token


class TextInput(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()


@app.post("/process/")
async def check_grammar(input: TextInput):
    # Der Prompt wird so formuliert, dass das Modell als Experte für deutsche Grammatik agiert.
    prompt = (
        "Du bist ein Experte für deutsche Grammatik. "
        "Deine Aufgabe ist es, den folgenden Text auf Grammatikfehler zu überprüfen und diesen gegebenenfalls zu korrigieren. "
        "Bitte gib nur den korrigierten Text aus, ohne weitere Erklärungen.\n\n"
        f"Eingabetext: {input.text}\n"
    )
    print(prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=1, early_stopping=True)
    checked_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"checked_text": checked_text}
