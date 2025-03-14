from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

import torch
import logging


MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
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
        "Beispiele für Zitationsstile:\n"
        "APA: Autor, A. (Jahr). Titel. Seitenangaben.\n"
        'Beispiel APA: Hinterseer, H. (2021). KI Systeme. S. 122.\n'
        "MLA: Autor, A. Titel. Seitenangaben.\n"
        'Beispiel MLA: Hinterseer, Hansi. "KI Systeme." S. 122.\n'
        "Chicago: Autor, A. Titel, Seitenangaben.\n"
        'Beispiel Chicago: Hinterseer, Hansi. "KI Systeme." S. 122.\n'
        'Harvard: Autor, A. Titel. Seitenangaben.\n'
        'Beispiel Harvard: Hinterseer, Hansi. "KI Systeme." S. 122.\n'
        'Oxford: Autor, A., "Titel," Seitenangaben.\n'
        'Beispiel Oxford: Hinterseer, Hansi, "KI Systeme," S. 122.\n'
        "Extrahiere den Zitationsstil aus der folgenden Benutzereingabe und erstelle das Zitat entsprechend:\n"
        f"{input.text}\n"
        "Gib nur das Zitat an, ohne zusätzliche Erklärungen."
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
