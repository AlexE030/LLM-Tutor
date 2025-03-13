import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
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
        f"Du bist ein Experte für die Gliederung von Wissenschaftlichen Arbeiten."
        f"Deine Aufgabe ist es eine Gliederung zu einer Wissenschaftlichen Arbeit zu schreiben."
        f"Bitte nenne nur die Gliederung ohne weitere Erläuterungen"
        f"Im Folgenden erhälst du weitere Informationen zum Thema\n\n"
        f"Benutzereingabe: {input.text}\n"
    )

    print(prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.empty_cache()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=512, num_beams=1, early_stopping=True)
    generated_tokens = outputs[0][input_length:]
    output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    torch.cuda.empty_cache()

    return {"response": output}