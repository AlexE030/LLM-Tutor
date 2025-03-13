from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

import sys
import torch
import logging


# Verwende das deutsch optimierte Modell
MODEL_NAME = "malteos/bloom-6b4-clp-german"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TextInput(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.eval()
    logging.debug("Modell set to evaluation mode.")
    yield
    torch.cuda.empty_cache()
    logging.debug("Shutdown performed successfully.")

app = FastAPI(lifespan=lifespan)

# TODO: Bring down responding time
# TODO: Provide good Answers
@app.post("/process/")
async def check_grammar(input: TextInput):
    # Der Prompt wird so formuliert, dass das Modell als Experte für deutsche Grammatik agiert.
    prompt = (
        "Du bist ein hochqualifizierter Lektor für deutsche Sprache mit umfassender Expertise in Grammatik, Rechtschreibung und Stilistik. "
        "Deine Aufgabe ist es, den folgenden Text präzise zu analysieren und alle Fehler zu korrigieren. "
        "Gib ausschließlich den korrigierten Text zurück.\n\n"
        "Führe folgende Aufgaben aus:\n\n"
        "1. Überprüfe die Rechtschreibung und korrigiere alle Tippfehler.\n"
        "2. Analysiere die Grammatik, einschließlich:\n"
        "   - Deklination von Nomen und Adjektiven\n"
        "   - Konjugation von Verben (Zeitformen und Modi)\n"
        "   - Korrekte Verwendung von Präpositionen und Fällen.\n"
        "3. Prüfe die Zeichensetzung, insbesondere die Kommasetzung.\n"
        "4. Korrigiere die Groß- und Kleinschreibung.\n"
        "5. Überprüfe die Satzstruktur und den Satzbau auf grammatikalische Korrektheit und Verständlichkeit.\n\n"
        f"Eingabetext: {input.text}\n"
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
