import os

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

import logging

from llm_client import LLMClient

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
llm_client = LLMClient(HF_TOKEN)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TextInput(BaseModel):
    text: str
    context: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.debug("Zephyr Agent startup.")
    yield
    logging.debug("Zephyr Agent shutdown.")

app = FastAPI(lifespan=lifespan)


@app.post("/process/")
async def generate_outline(input: TextInput):
    prompt = (
        "Du bist ein hochspezialisierter Experte für die Erstellung von Zitaten für wissenschaftliche Arbeiten. "
        "Deine einzige Aufgabe ist es, genaue Zitate basierend auf einem definierten Zitationsstil zu erstellen. "
        "Unter keinen Umständen solltest du zusätzliche Erklärungen oder Texte über das Zitat selbst hinaus liefern. "
        "Konzentriere dich ausschließlich auf die Erstellung des Zitats und die Einhaltung des angegebenen Stils. "
        "Bitte befolge diese Richtlinien bei der Erstellung von Zitaten: "
        "1. Strikte Einhaltung des Stils: Befolge den angegebenen Zitationsstil genau (z. B. APA, MLA, Chicago, Harvard). Achte auf jedes Detail, einschließlich Interpunktion, Großschreibung und Formatierung. "
        "2. Umfassende Informationen: Stelle sicher, dass alle notwendigen Informationen (Autor, Titel, Erscheinungsdatum, Zeitschrift/Buchtitel, Band, Ausgabe, Seitenzahlen, DOI/URL) im Zitat enthalten sind. "
        "3. Genauigkeit: Überprüfe die Genauigkeit aller Informationen doppelt, um Fehler im Zitat zu vermeiden. "
        "4. Konsistenz: Sorge für Konsistenz in Formatierung und Stil in allen Zitaten. "
        "5. Umgang mit verschiedenen Quellentypen: Passe das Zitationsformat an verschiedene Quellentypen an (Bücher, Zeitschriftenartikel, Websites usw.). "
        "6. In-Text vs. Bibliographie: Unterscheide, falls angegeben, zwischen In-Text-Zitaten und vollständigen Bibliographieeinträgen. "
        "7. Mehrere Autoren/Herausgeber: Behandle Zitate mit mehreren Autoren oder Herausgebern korrekt. "
        "8. Fehlende Informationen: Wenn Informationen fehlen, gib dies entsprechend an (z. B. 'o. D.' für ohne Datum). "
        "9. Datumsformatierung: Formatiere Daten gemäß dem angegebenen Zitationsstil. "
        "10. Digital Object Identifiers (DOIs) und URLs: Füge DOIs oder URLs ein, wenn verfügbar. "
        "11. Ausgabeinformationen: Füge Ausgabeinformationen hinzu, wenn diese in der Benutzereingabe enthalten sind. "
        "12. Verlaginformationen: Füge Verlaginformationen für Buchzitate hinzu. "
        "13. Band- und Ausgabeninformationen: Füge Band- und Ausgabeninformationen für Zeitschriftenzitate korrekt hinzu. "
        "14. Seitenbereiche: Zeige die Seitenbereiche für Artikel und Bücher korrekt an. "
        "Beispiele für Zitationsstile:"
        "APA: Autor, A. (Jahr). Titel. Seitenangaben."
        'Beispiel APA: Hinterseer, H. (2021). KI Systeme. S. 122.'
        "MLA: Autor, A. Titel. Seitenangaben."
        'Beispiel MLA: Hinterseer, Hansi. "KI Systeme." S. 122.'
        "Chicago: Autor, A. Titel, Seitenangaben."
        'Beispiel Chicago: Hinterseer, Hansi. "KI Systeme." S. 122.'
        'Harvard: Autor, A. Titel. Seitenangaben.'
        'Beispiel Harvard: Hinterseer, Hansi. "KI Systeme." S. 122.'
        'Oxford: Autor, A., "Titel," Seitenangaben.'
        'Beispiel Oxford: Hinterseer, Hansi, "KI Systeme," S. 122.'
        
        f"Hier hast du noch weitere Hinweise zur Zitation aus den Richtlinien der Hochschule, welche dir behilflich sein können: {input.context}"
        "Extrahiere den Zitationsstil aus der folgenden Benutzereingabe und erstelle das Zitat entsprechend:"
        f"{input.text}"
        "Gib nur das Zitat an, ohne zusätzliche Erklärungen."
    )

    messages = [{"role": "user", "content": prompt}]
    response = llm_client.query_instruct(
        model=MODEL_NAME,
        messages=messages,
    )

    return {"response": response["choices"][0]["message"]["content"].strip().lower()}
