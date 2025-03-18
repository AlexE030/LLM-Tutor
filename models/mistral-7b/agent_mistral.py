from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

import os
import logging

from llm_client import LLMClient

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
llm_client = LLMClient(HF_TOKEN)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TextInput(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.debug("Mistral Agent startup.")
    yield
    logging.debug("Mistral Agent shutdown.")

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
        "Erstelle eine detaillierte Gliederung zu folgendem Thema:"
        f"{input.text}\n"
        "Gib nur die Gliederung ohne weitere Erklärungen an."
    )

    response = llm_client.query_instruct(
        model=MODEL_NAME,
        message=prompt,
        max_tokens=1000
    )

    return {"response": response.content}