from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import asyncio
import functools
import logging

app = FastAPI()

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda:1")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextInput(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()

def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(None, partial_func)


@app.post("/process/")
async def combine_llm_outputs(input: TextInput):
    try:
        if not input.text:
            raise HTTPException(status_code=400, detail="No text provided")

        prompts = [
            "Du bist ein Experte für das Zusammenfassen und Verbessern von Texten, speziell für wissenschaftliche Arbeiten. Deine Aufgabe ist es, Informationen aus verschiedenen Quellen zu einem kohärenten, informativen und wissenschaftlich fundierten Text zu kombinieren.",
            "Verwende einen klaren, präzisen und akademischen Schreibstil. Vermeide Umgangssprache und unnötige Wiederholungen.",
            "Achte auf die logische Struktur und den roten Faden des Textes. Stelle sicher, dass die Informationen in einer sinnvollen Reihenfolge präsentiert werden.",
            "Identifiziere und extrahiere die wichtigsten Informationen, Argumente und Ergebnisse aus den bereitgestellten Texten.",
            "Fasse die Kernpunkte jedes Textabschnitts präzise zusammen.",
            "Vergleiche und kontrastiere die Informationen aus den verschiedenen Quellen. Identifiziere Gemeinsamkeiten, Unterschiede und mögliche Widersprüche.",
            "Synthetisiere die Informationen zu einem neuen, umfassenden Text, der die wichtigsten Erkenntnisse der Originalquellen widerspiegelt.",
            "Achte darauf, dass alle Behauptungen und Argumente durch stichhaltige Beweise gestützt werden.",
            "Gib korrekte Zitate und Referenzen für alle verwendeten Quellen an. Verwende einen einheitlichen Zitierstil (z. B. APA, MLA, Chicago).",
            "Überprüfe den resultierenden Text auf Rechtschreib-, Grammatik- und Zeichensetzungsfehler. Korrigiere alle Fehler sorgfältig.",
            "Stelle sicher, dass der Text den Standards für wissenschaftliches Schreiben entspricht. Verwende Fachsprache und Terminologie angemessen.",
            "Formuliere den Text so um, dass er schlüssig und leicht zu folgen ist. Achte auf einen guten Lesefluss.",
            "Sei kritisch gegenüber den Informationen in den Quelltexten Bewerte ihre Glaubwürdigkeit und Relevanz.",
            "Ergänze den Text gegebenfalls durch erklärende beispiele oder belege, um die verständlichkeit des textes zu erhöhen.",
            f"Hier ist der Text, den du verarbeiten sollst:\n\n{input.text}"
        ]
        full_prompt = "\n".join(prompts)

        logging.info(f"Verarbeite Anfrage mit Text: {input.text[:50]}...")

        print(full_prompt)

      #  device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        outputs = await run_in_threadpool(model.generate, **inputs, max_length=2048, num_beams=5, early_stopping=True,
                                          temperature=0.7, top_k=50)
        outline = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info("Anfrage erfolgreich verarbeitet.")
        return {"textausgabe": outline}

    except HTTPException as http_exception:
        logging.error(f"HTTP-Fehler: {http_exception.detail}")
        raise http_exception
    except Exception as e:
        logging.error(f"Unerwarteter Fehler: {e}")
        raise HTTPException(status_code=500, detail=f"Unerwarteter Fehler: {e}")

""""
app = FastAPI()

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()
    logging.info("Mistral-Modell geladen")

def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(None, partial_func)

@app.post("/process/")
async def combine_llm_outputs(input: TextInput):
    try:
        if not input.text:
            raise HTTPException(status_code=400, detail="No text provided")

        prompts = [
            "Du bist ein Experte für das Zusammenfassen und Verbessern von Texten, speziell für wissenschaftliche Arbeiten. Deine Aufgabe ist es, Informationen aus verschiedenen Quellen zu einem kohärenten, informativen und wissenschaftlich fundierten Text zu kombinieren.",
            "Verwende einen klaren, präzisen und akademischen Schreibstil. Vermeide Umgangssprache und unnötige Wiederholungen.",
            "Achte auf die logische Struktur und den roten Faden des Textes. Stelle sicher, dass die Informationen in einer sinnvollen Reihenfolge präsentiert werden.",
            "Identifiziere und extrahiere die wichtigsten Informationen, Argumente und Ergebnisse aus den bereitgestellten Texten.",
            "Fasse die Kernpunkte jedes Textabschnitts präzise zusammen.",
            "Vergleiche und kontrastiere die Informationen aus den verschiedenen Quellen. Identifiziere Gemeinsamkeiten, Unterschiede und mögliche Widersprüche.",
            "Synthetisiere die Informationen zu einem neuen, umfassenden Text, der die wichtigsten Erkenntnisse der Originalquellen widerspiegelt.",
            "Achte darauf, dass alle Behauptungen und Argumente durch stichhaltige Beweise gestützt werden.",
            "Gib korrekte Zitate und Referenzen für alle verwendeten Quellen an. Verwende einen einheitlichen Zitierstil (z. B. APA, MLA, Chicago).",
            "Überprüfe den resultierenden Text auf Rechtschreib-, Grammatik- und Zeichensetzungsfehler. Korrigiere alle Fehler sorgfältig.",
            "Stelle sicher, dass der Text den Standards für wissenschaftliches Schreiben entspricht. Verwende Fachsprache und Terminologie angemessen.",
            "Formuliere den Text so um, dass er schlüssig und leicht zu folgen ist. Achte auf einen guten Lesefluss.",
            "Sei kritisch gegenüber den Informationen in den Quelltexten. Bewerte ihre Glaubwürdigkeit und Relevanz.",
            "Ergänze den Text gegebenfalls durch erklärende beispiele oder belege, um die verständlichkeit des textes zu erhöhen.",
            f"Hier ist der Text, den du verarbeiten sollst:\n\n{input.text}"
        ]
        full_prompt = "\n".join(prompts)

        logging.info(f"Verarbeite Anfrage mit Text: {input.text[:50]}...")

        print(full_prompt)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        outputs = await run_in_threadpool(model.generate, **inputs, max_length=2048, num_beams=5, early_stopping=True,
                                          temperature=0.7, top_k=50)
        outline = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info("Anfrage erfolgreich verarbeitet.")
        return {"textausgabe": outline}

    except HTTPException as http_exception:
        logging.error(f"HTTP-Fehler: {http_exception.detail}")
        raise http_exception
    except Exception as e:
        logging.error(f"Unerwarteter Fehler: {e}")
        raise HTTPException(status_code=500, detail=f"Unerwarteter Fehler: {e}")

"""