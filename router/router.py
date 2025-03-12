from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

from enums.model import Model

app = FastAPI()

class TextRequest(BaseModel):
    text: str


async def get_model_response(model: Model, text: str):
    try:
        response = requests.post(model.value, json={"text": text})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Aufruf von {model.name}: {e}")


async def classify_prompt(text: str):
    prompt = f"""
        You are an expert Manager Agent. Your task is to classify user questions into one of the following categories.
        The questions might be in german language. 
        Each question must belong to exactly one category. Each question must receive precisely and only one category:

        - **"citation"**: For Questions refering to the creation of quotes. Especially watch out for the word "Zitat". Look for phrases like:
        - "Erstelle mir ein Zitat"
        - "Mache ein Zitat zu"
        Examples:
            - "Erstelle mir hierraus ein Zitat im Chicago Stil: 'Es gibt 2 Dinge die unendlich sind...'" -> citation
            - "Ich möchte daraus ein Zitat haben" -> citation

        - **"structure"**: For Questions that refer to the creation of a structure. Especially watch out for the words "Gliederung" or "Struktur" Look for phrases like:
        - "Erstelle mir eine Giederung hierzu"
        - "Gib mir eine Struktur zu diesem Thema"
        Examples:
            - "Erstelle eine Gliederung zum Thema Delfine" -> structure
            - "Gib mir eine Struktur für meine Bachelorarbeit" -> structure

        - **"grammar"**: For requests about the gramatical improvement of text. Look for phrases like:
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

    response = await asyncio.gather(get_model_response(Model.MISTRAL, prompt))

    classification = response.get("classification")

    match classification:
        case "citation":
            return Model.ZEPHYR
        case "structure":
            return Model.LLAMA
        case "grammar":
            return Model.BLOOM
        case "none":
            return None

    return None


@app.post("/process/")
async def process_text(request: TextRequest):
    try:
        text = request.text
        model = await asyncio.gather(classify_prompt(text))
        result = await asyncio.gather(get_model_response(model, text))

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")