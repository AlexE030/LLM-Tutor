from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio
import logging
import sys

from enum import Enum


logger = logging.getLogger("router")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

class TextRequest(BaseModel):
    text: str


class Model(Enum):
    LLAMA = "http://llama_api:8000/process/"
    ZEPHYR = "http://zephyr_api:8000/process/"
    BLOOM = "http://bloom_api:8000/process/"
    MISTRAL = "http://mistral_api:8000/process/"
    NONE = "None"


class InputState(Enum):
    REQUEST = 1
    CONFIRM = 2
    CHOOSE_MODEL = 3


class NoResposeError(Exception):
    """Raised when no response is received from LLM."""
    pass

app = FastAPI()
forbidden_chars = ['\"', '\'']
input_state = InputState.REQUEST
overpass = {
    "userQuery": '',
    "model": Model.NONE
    }


async def get_model_response(model: Model, text: str):
    try:
        response = requests.post(model.value, json={"text": text})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Aufruf von {model.name}: {e}")


# TODO: Only use first part of prompt for classification (split by :?)
async def classify_prompt(text: str):
    relevant_text = text.split(':', 1)[0]
    prompt = f"""
        You are an expert Manager Agent. Your task is to classify user questions into one of the following categories.
        The questions might be in german language. 
        Each question must belong to exactly one category. Each question must receive only ONE category
        The possible categories are citation, structure, grammar, none 
        In the following you will find a description for each category.

        - **"citation"**: For Questions referring to the creation of quotes. Especially watch out for the word "Zitat". Look for phrases like:
        - "Erstelle mir ein Zitat"
        - "Mache ein Zitat zu"
        Examples:
            - "Erstelle mir hieraus ein Zitat im Chicago Stil" -> citation
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

        User Question: "{relevant_text}"
        """


    response_list = await asyncio.gather(get_model_response(Model.LLAMA, prompt))
    response = response_list[0]

    classification = response.get("response")

    match classification:
        case "citation":
            return Model.ZEPHYR
        case "structure":
            return Model.MISTRAL
        case "grammar":
            return Model.BLOOM
        case "none":
            return Model.NONE

    logger.debug(f"llama-Response for classification: {classification}")
    return Model.NONE


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
    return Model.NONE


async def handle_backfall(text: str):
    global input_state
    global overpass
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

    overpass["userQuery"] = text
    overpass["model"] = model

    if not subject == "":
        result += f"\nGeht es in deiner Anfrage um folgendes: {subject} (Bestätige mit ja oder nein)"
        input_state = InputState.CONFIRM
    else:
        result += "\nBitte gib die Art deiner Anfrage manuell ein (1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon)"
        input_state = InputState.CHOOSE_MODEL

    return {"response": result}


async def handle_request_state(text: str):
    model_list = await asyncio.gather(classify_prompt(text))
    model = model_list[0]

    if model in [Model.ZEPHYR, Model.MISTRAL, Model.BLOOM]:
        result_list = await asyncio.gather(get_model_response(model, text))
    elif model in [Model.NONE]:
        result_list = await asyncio.gather(handle_backfall(text))
    else:
        raise NoResposeError("No response from llama")

    return result_list[0]


async def handle_confirm_state(text: str):
    global input_state
    if text.lower() in ["ja", "yes", "j", "y"]:
        response_list = await asyncio.gather(get_model_response(overpass["model"], overpass["userQuery"]))
        input_state = InputState.REQUEST
        return response_list[0]
    elif text.lower() in ["nein", "no", "n"]:
        input_state = InputState.CHOOSE_MODEL
        result = "Bitte gib die Art deiner Anfrage manuell ein (1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon)"
        return {"response": result}
    else:
        result = "Bitte gib eine sinnvolle Antwort ein. Möglich sind ja oder nein"
        return {"response": result}

async def handle_choose_model_state(text: str):
    global input_state
    if text.lower() in ["zitat", "z", "1"]:
        response_list = await asyncio.gather(get_model_response(Model.ZEPHYR, overpass["userQuery"]))
        input_state = InputState.REQUEST
        return response_list[0]
    if text.lower() in ["gliederung", "g", "2"]:
        response_list = await asyncio.gather(get_model_response(Model.MISTRAL, overpass["userQuery"]))
        input_state = InputState.REQUEST
        return response_list[0]
    if text.lower() in ["formulierung", "f", "3"]:
        response_list = await asyncio.gather(get_model_response(Model.BLOOM, overpass["userQuery"]))
        input_state = InputState.REQUEST
        return response_list[0]
    if text.lower() in ["nichts davon", "n", "4"]:
        result = "Es sieht so aus als wäre unser KI-Assistent nicht auf deine Anfrage ausgelegt."
        input_state = InputState.REQUEST
        return {"response": result}
    else:
        result = "Bitte gib eine sinnvolle Antwort ein. Möglich sind 1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon"
        return {"response": result}


@app.post("/reset/")
async def reset_state():
    global input_state
    input_state = InputState.REQUEST
    return {"response": "Input state wurde auf REQUEST zurückgesetzt."}


@app.post("/process/")
async def process_text(request: TextRequest):
    try:
        text = request.text
        logger.info(f"Processing user request: {text}")
        for char in forbidden_chars:
            text = text.replace(char, "")

        match input_state:
            case InputState.REQUEST:
                result_list = await asyncio.gather(handle_request_state(text))
                return result_list[0]
            case InputState.CONFIRM:
                result_list = await asyncio.gather(handle_confirm_state(text))
                return result_list[0]
            case InputState.CHOOSE_MODEL:
                result_list = await asyncio.gather(handle_choose_model_state(text))
                return result_list[0]

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")