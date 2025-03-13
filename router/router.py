from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

from enum import Enum

app = FastAPI()
forbidden_chars = ['\"', '\'']

class TextRequest(BaseModel):
    text: str


class Model(Enum):
    LLAMA = "http://llama_api:8000/process/"
    ZEPHYR = "http://zephyr_api:8000/process/"
    BLOOM = "http://bloom_api:8000/process/"
    MISTRAL = "http://mistral_api:8000/process/"


async def get_model_response(model: Model, text: str):
    try:
        response = requests.post(model.value, json={"text": text})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Aufruf von {model.name}: {e}")


# TODO: Only use first part of prompt for classification (split by :?)
async def classify_prompt(text: str):
    prompt = f"""
        You are an expert Manager Agent. Your task is to classify user questions into one of the following categories.
        The questions might be in german language. 
        Each question must belong to exactly one category. Each question must receive precisely and only one category:

        - **"citation"**: For Questions referring to the creation of quotes. Especially watch out for the word "Zitat". Look for phrases like:
        - "Erstelle mir ein Zitat"
        - "Mache ein Zitat zu"
        Examples:
            - "Erstelle mir hieraus ein Zitat im Chicago Stil: 'Es gibt 2 Dinge die unendlich sind...'" -> citation
            - "Ich möchte daraus ein Zitat haben" -> citation

        - **"structure"**: For Questions that refer to the creation of a structure. Especially watch out for the words "Gliederung" or "Struktur" Look for phrases like:
        - "Erstelle mir eine Gliederung hierzu"
        - "Gib mir eine Struktur zu diesem Thema"
        Examples:
            - "Erstelle eine Gliederung zum Thema Delfine" -> structure
            - "Gib mir eine Struktur für meine Bachelorarbeit" -> structure

        - **"grammar"**: For requests about the grammatical improvement of text. Look for phrases like:
        - "Verbessere mir diesen Text"
        - "Schreib das schöner"
        Examples:
            - "Verbessere mir das Folgende:" -> grammar
            - "Bitte schreib das so um, dass es besser klingt" -> grammar

        - **"none"**: For questions that are not related to the above categories or the writing of scientific Papers. 
        - In general, for questions that have nothing to do with writing an scientific paper. 
        - Also for questions where you are unsure which category applies. Examples:
            - "Wie heißt mein Hund?" -> none
            - "Wie ist das Wetter morgen?" -> none
            - "Wie groß ist die Erde?" -> none
            - "Was ist die Hauptstadt von Frankreich?" -> none

        Respond **only** and truly **ONLY** with the category name!

        User Question: "{text}"
        """

    response_list = await asyncio.gather(get_model_response(Model.LLAMA, prompt))
    response = response_list[0]

    classification = response.get("classification")

    match classification:
        case "citation":
            return Model.ZEPHYR
        case "structure":
            return Model.MISTRAL
        case "grammar":
            return Model.BLOOM
        case "none":
            return None

    return None

async def classify_prompt_backfall(text: str):
    zephyr_strings = ["zitat", "zitiere"]
    mistral_strings = ["struktur", "glieder", "strucktur", "glider"]
    bloom_strings = ["verbesser", "überprüf", "schön"]

    if any(word in text.lower() for word in zephyr_strings):
        return Model.ZEPHYR
    if any(word in text.lower() for word in mistral_strings):
        return Model.MISTRAL
    if any(word in text.lower() for word in bloom_strings):
        return Model.BLOOM
    return None


async def handle_backfall(text: str):
    model_list = await asyncio.gather(classify_prompt_backfall(text))
    model = model_list[0]
    subject = ""

    match model:
        case Model.ZEPHYR:
            subject = "Zitat"
        case Model.MISTRAL:
            subject = "Gliederung"
        case Model.BLOOM:
            subject = "Formulierung und Grammatik"

    result = "Kein passendes Modell gefunden"

    if not subject == "":
        result += f"\n Geht es in deiner Anfrage um folgendes: {subject} (Bestätige mit ja oder nein)"
        # TODO: If ja, dann entsprechendes modell aufrufen.
        #  Bei nein selbe frage wie in else stellen
        #  Ansonsten erneut fragen
    else:
        result += "\n Bitte gib die Art deiner Anfrage manuell ein (1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon)"
        # TODO: If eins von denen entsprechendes Modell aufrufen.
        #  Wenn nichts davon folgende Nachricht: "Es sieht so aus als wäre unser KI-Assistent nicht auf deine Anfrage ausgelegt."
        #  Ansonsten erneut fragen

    return {"result": result}


@app.post("/process/")
async def process_text(request: TextRequest):
    try:
        text = request.text
        for char in forbidden_chars:
            text = text.replace(char, "")
        model_list = await asyncio.gather(classify_prompt(text))
        model = model_list[0]

        if model:
            result_list = await asyncio.gather(get_model_response(model, text))
            result = result_list[0]
            return result
        else:
            result_list = await asyncio.gather(handle_backfall(text))
            return result_list[0]


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")