from encodings import undefined

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("router")


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


class Retriever:
    def __init__(self, collection_name="dhbw_rules"):
        self.client = chromadb.client = chromadb.HttpClient(host="host.docker.internal", port=8000)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug(f"Retriever initialized with collection: {self.collection.name}")

    def retrieve_relevant_documents(self, query, top_n=5):
        logger.debug(f"Retrieving documents for query: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        logger.debug(f"Retrieved documents: {results}")
        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.input_state = InputState.REQUEST
    app.state.userQuery = ""
    app.state.model = Model.NONE
    app.state.retriever = Retriever()
    logger.debug("State initialized via lifespan.")
    yield
    logger.debug("Shutdown completed.")


app = FastAPI(lifespan=lifespan)

origins = [
    "http://192.168.23.112:3000",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

forbidden_chars = ['\"', '\'']


async def get_model_response(model: Model, text: str, state = None):
    try:
        context = None
        if state:
            context = state.retriever.retrieve_relevant_documents(text)
            logger.debug(f"Retrieved context from chromaDB: {context}")
            context = " ".join(context)  # Combine documents into a single string
            logger.debug(f"Context after joining: {context}")

        payload = {"text": text}
        if context:
            payload["context"] = context
        logger.debug(f"Sending request to {model.name} with payload: {payload}")  # Add payload logging
        response = requests.post(model.value, json=payload)
        response.raise_for_status()
        response_json = response.json()
        logger.debug(f"Response from {model.name}: {response_json}")  # Log the response
        return response_json
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling {model.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Aufruf von {model.name}: {e}")


async def classify_prompt(text: str):
    relevant_text = text.split(':', 1)[0]
    prompt = f"""
        Du agierst als Experte im Management von Benutzeranfragen. 
        Deine Aufgabe ist es, Benutzeranfragen in exakt eine der folgenden vier Kategorien einzustufen: citation, structure, grammar, none. 
        Jede Anfrage muss ausschließlich einer Kategorie zugeordnet werden.

        Beachte: Die Anfragen können auf Deutsch formuliert sein.
        
        Mögliche Kategorien und deren Kriterien:
        
        "citation"
        Für Anfragen, die sich auf die Erstellung von Zitaten beziehen. Achte besonders auf Wörter wie „Zitat“ sowie Formulierungen wie:
        
        „Erstelle mir ein Zitat“
        
        „Mache ein Zitat zu“
        Beispiele:
        
        „Erstelle mir hieraus ein Zitat im Chicago-Stil“ → citation
        
        „Ich möchte daraus ein Zitat haben“ → citation
        
        "structure"
        Für Anfragen, die den Aufbau oder die Gliederung eines Textes oder Themas betreffen. Achte auf Wörter wie „Gliederung“ oder „Struktur“ und Formulierungen wie:
        
        „Erstelle mir eine Gliederung hierzu“
        
        „Gib mir eine Struktur zu diesem Thema“
        Beispiele:
        
        „Erstelle eine Gliederung zum Thema Delfine“ → structure
        
        „Gib mir eine Struktur für meine Bachelorarbeit“ → structure
        
        "grammar"
        Für Anfragen, bei denen es um die grammatikalische Verbesserung oder stilistische Überarbeitung eines Textes geht. Achte auf Phrasen wie:
        
        „Verbessere mir diesen Text“
        
        „Schreib das schöner“
        Beispiele:
        
        „Verbessere mir das Folgende:“ → grammar
        
        „Bitte schreib das so um, dass es besser klingt“ → grammar
        
        "none"
        Für Anfragen, die keinen Bezug zu den oben genannten Kategorien oder zur Erstellung wissenschaftlicher Texte haben. Auch wenn nicht klar ist, welche Kategorie zutrifft, solltest du „none“ wählen.
        Beispiele:
        
        „Wie heißt mein Hund?“ → none
        
        „Wie ist das Wetter morgen?“ → none
        
        „Wie groß ist die Erde?“ → none
        
        „Was ist die Hauptstadt von Frankreich?“ → none
        
        Wichtige Anweisung:
        Antworte ausschließlich mit dem Namen der Kategorie! Deine Antwort darf nur eine der folgenden Optionen enthalten:
        
        citation
        
        structure
        
        grammar
        
        none
        
        Benutzeranfrage:
        "{relevant_text}"
        """
    logger.debug(f"Classifying prompt: {prompt}")
    response_list = await asyncio.gather(get_model_response(Model.LLAMA, prompt))
    response = response_list[0]
    classification = response.get("response")
    logger.debug(f"llama-Response for classification: {classification}")
    if classification == "citation":
        return Model.ZEPHYR
    elif classification == "structure":
        return Model.MISTRAL
    elif classification == "grammar":
        return Model.BLOOM
    elif classification == "none":
        return Model.NONE
    return Model.NONE


async def classify_prompt_backfall(text: str):
    zephyr_strings = ["zitat", "zitiere"]
    mistral_strings = ["struktur", "glieder", "strucktur", "glider"]
    bloom_strings = ["verbesser", "überprüf", "schön"]
    logger.debug(f"Classifying prompt backfall for text: {text}")
    if any(word in text.lower() for word in zephyr_strings):
        logger.debug("Matched zephyr_strings")
        return Model.ZEPHYR
    if any(word in text.lower() for word in mistral_strings):
        logger.debug("Matched mistral_strings")
        return Model.MISTRAL
    if any(word in text.lower() for word in bloom_strings):
        logger.debug("Matched bloom_strings")
        return Model.BLOOM
    logger.debug("No model matched in backfall")
    return Model.NONE


async def handle_backfall(text: str, state):
    logger.debug(f"Handling backfall for text: {text}")
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
    state.userQuery = text
    state.model = model
    if subject:
        result += f"\nGeht es in deiner Anfrage um folgendes: {subject} (Bestätige mit ja oder nein)"
        state.input_state = InputState.CONFIRM
    else:
        result += "\nBitte gib die Art deiner Anfrage manuell ein (1 = zitat, 2 = gliederung, 3 = formulierung, 4 = nichts davon)"
        state.input_state = InputState.CHOOSE_MODEL
    logger.debug(f"Backfall result: {result}")
    return {"response": result}


async def handle_request_state(text: str, state):
    logger.debug(f"Handling request state for text: {text}")
    model_list = await asyncio.gather(classify_prompt(text))
    model = model_list[0]
    logger.debug(f"Model selected: {model}")
    if model in [Model.ZEPHYR, Model.MISTRAL, Model.BLOOM]:
        result_list = await asyncio.gather(get_model_response(model, text, state))  # Pass context to LLM
    elif model in [Model.NONE]:
        result_list = await asyncio.gather(handle_backfall(text, state))
    else:
        raise NoResposeError("No response from llama")
    return result_list[0]


async def handle_confirm_state(text: str, state):
    logger.debug(f"Handling confirm state for text: {text} with model: {state.model}")
    if text.lower() in ["ja", "yes", "j", "y"]:
        result_list = await asyncio.gather(get_model_response(state.model, state.userQuery, state))
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
    logger.debug(f"Handling choose model state for text: {text}")
    if text.lower() in ["zitat", "z", "1"]:
        result_list = await asyncio.gather(get_model_response(Model.ZEPHYR, state.userQuery, state))
        state.input_state = InputState.REQUEST
        return result_list[0]
    if text.lower() in ["gliederung", "g", "2"]:
        result_list = await asyncio.gather(get_model_response(Model.MISTRAL, state.userQuery, state))
        state.input_state = InputState.REQUEST
        return result_list[0]
    if text.lower() in ["formulierung", "f", "3"]:
        result_list = await asyncio.gather(get_model_response(Model.BLOOM, state.userQuery, state))
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
        if text is None or text is undefined:
            return {"error": "No text provided"}
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