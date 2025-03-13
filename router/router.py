from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import requests
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("router")
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.input_state = InputState.REQUEST
    app.state.overpass = {"userQuery": "", "model": Model.NONE}
    logger.debug("State initialized via lifespan.")
    yield
    logger.debug("Shutdown completed.")

app = FastAPI(lifespan=lifespan)

origins = ["http://192.168.23.112:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

forbidden_chars = ['\"', '\'']

async def get_model_response(model: Model, text: str):
    try:
        response = requests.post(model.value, json={"text": text})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Aufruf von {model.name}: {e}")

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
        Therefore your response looks like one of the following:
        
        citation
        
        structure
        
        grammar
        
        none
        
        You have only those 4 possible responses!

        User Question: "{relevant_text}"
        """
    response_list = await asyncio.gather(get_model_response(Model.LLAMA, prompt))
    response = response_list[0]
    classification = response.get("response")
    if classification == "citation":
        return Model.ZEPHYR
    elif classification == "structure":
        return Model.MISTRAL
    elif classification == "grammar":
        return Model.BLOOM
    elif classification == "none":
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

async def handle_backfall(text: str, state):
    model_list = await asyncio.gather(classify_prompt_backfall(text))
    model = model_list[0]
    subject = ""
    if model == Model.ZEPHYR:
        subject = "Zitat"
    elif model == Model.MISTRAL:
        subject = "Gliederung"
    elif model == Model.BLOOM:
        subject = "Formulierung und Grammatik"
    result = "Kein passendes Modell gefunden"
    state.overpass["userQuery"] = text
    state.overpass["model"] = model
    if subject:
        result += f"\nGeht es in deiner Anfrage um folgendes: {subject} (Bestätige mit ja oder nein)"
        state.input_state = InputState.CONFIRM
    else:
        result += "\nBitte gib die Art deiner Anfrage manuell ein (1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon)"
        state.input_state = InputState.CHOOSE_MODEL
    return {"response": result}

async def handle_request_state(text: str, state):
    model_list = await asyncio.gather(classify_prompt(text))
    model = model_list[0]
    if model in [Model.ZEPHYR, Model.MISTRAL, Model.BLOOM]:
        result_list = await asyncio.gather(get_model_response(model, text))
    elif model in [Model.NONE]:
        result_list = await asyncio.gather(handle_backfall(text, state))
    else:
        raise NoResposeError("No response from llama")
    return result_list[0]

async def handle_confirm_state(text: str, state):
    if text.lower() in ["ja", "yes", "j", "y"]:
        result_list = await asyncio.gather(get_model_response(state.overpass["model"], state.overpass["userQuery"]))
        state.input_state = InputState.REQUEST
        return result_list[0]
    elif text.lower() in ["nein", "no", "n"]:
        state.input_state = InputState.CHOOSE_MODEL
        result = "Bitte gib die Art deiner Anfrage manuell ein (1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon)"
        return {"response": result}
    else:
        result = "Bitte gib eine sinnvolle Antwort ein. Möglich sind ja oder nein"
        return {"response": result}

async def handle_choose_model_state(text: str, state):
    if text.lower() in ["zitat", "z", "1"]:
        result_list = await asyncio.gather(get_model_response(Model.ZEPHYR, state.overpass["userQuery"]))
        state.input_state = InputState.REQUEST
        return result_list[0]
    if text.lower() in ["gliederung", "g", "2"]:
        result_list = await asyncio.gather(get_model_response(Model.MISTRAL, state.overpass["userQuery"]))
        state.input_state = InputState.REQUEST
        return result_list[0]
    if text.lower() in ["formulierung", "f", "3"]:
        result_list = await asyncio.gather(get_model_response(Model.BLOOM, state.overpass["userQuery"]))
        state.input_state = InputState.REQUEST
        return result_list[0]
    if text.lower() in ["nichts davon", "n", "4"]:
        result = "Es sieht so aus als wäre unser KI-Assistent nicht auf deine Anfrage ausgelegt."
        state.input_state = InputState.REQUEST
        return {"response": result}
    else:
        result = "Bitte gib eine sinnvolle Antwort ein. Möglich sind 1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon"
        return {"response": result}

@app.post("/reset/")
async def reset_state(request: Request):
    request.app.state.input_state = InputState.REQUEST
    return {"response": "Input state has been reset to REQUEST."}

@app.post("/process/")
async def process_text(request: Request, req: TextRequest):
    try:
        text = req.text
        logger.info(f"Processing user request: {text}")
        for char in forbidden_chars:
            text = text.replace(char, "")
        state = request.app.state
        if state.input_state == InputState.REQUEST:
            result_list = await asyncio.gather(handle_request_state(text, state))
            return result_list[0]
        elif state.input_state == InputState.CONFIRM:
            result_list = await asyncio.gather(handle_confirm_state(text, state))
            return result_list[0]
        elif state.input_state == InputState.CHOOSE_MODEL:
            result_list = await asyncio.gather(handle_choose_model_state(text, state))
            return result_list[0]
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")