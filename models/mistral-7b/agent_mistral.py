from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

import os
import torch
import logging

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TextInput(BaseModel):
    text: str
    metadata: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.eval()
    logging.debug("Modell set to evaluation mode.")
    yield
    torch.cuda.empty_cache()
    logging.debug("Shutdown performed successfully.")

app = FastAPI(lifespan=lifespan)


@app.post("/process/")
async def generate_outline(input: TextInput):
    prompt = (
        "Du bist ein hochspezialisierter Experte für die Erstellung von Gliederungen für wissenschaftliche Arbeiten."
        "Deine einzige Aufgabe ist es, detaillierte und logisch strukturierte Gliederungen für wissenschaftliche Themen zu erstellen."
        "Unter keinen Umständen solltest du über diese Aufgabe hinausgehen und zusätzliche Erklärungen oder Texte liefern."
        "Die Gliederung muss ein logischer Leitfaden für die Arbeit sein."
        "Beachte die folgenden Kriterien bei der Erstellung der Gliederung:"
        "1.  Zielgruppenorientierung: Berücksichtige, dass die Gliederung für Studierende und Forschende relevant sein muss."
        "2.  Logischer Aufbau: Stelle sicher, dass die Gliederung einen klaren und logischen Aufbau hat, der den Leser durch das Thema führt."
        "3.  Vollständigkeit: Die Gliederung sollte alle relevanten Aspekte des Themas abdecken."
        "4.  Hierarchie: Verwende eine klare Hierarchie (z. B. römische Zahlen, arabische Zahlen, Buchstaben), um Haupt- und Unterpunkte zu strukturieren."
        "5.  Präzision: Formuliere die Gliederungspunkte präzise und aussagekräftig, sodass der Inhalt der jeweiligen Abschnitte klar erkennbar ist."
        "6.  Thematische Relevanz: Stelle sicher, dass jeder Gliederungspunkt direkt zum Thema der wissenschaftlichen Arbeit beiträgt."
        "7.  Aktualität: Beziehe, wo möglich, aktuelle Forschungsergebnisse ein."
        "8.  Methodik: Berücksichtige, wenn in der Benutzereingabe erwähnt, die in der Gliederung verwendete Forschungsmethodik."
        "9.  Forschungsfrage: Stelle sicher, dass die Gliederung die Forschungsfrage logisch adressiert und beantwortet."
        "10. Quellen: Beziehe, wenn Quellen angegeben sind, diese ein."
        
        f"Hier hast du einige zusätzliche Informationen aus den Richtlinien der Hochschule, die die helfen können: {input.metadata}" 
        
        "Erstelle eine detaillierte Gliederung zu folgendem Thema:"
        f"{input.text}\n"
        "Gib nur die Gliederung ohne weitere Erklärungen an."
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