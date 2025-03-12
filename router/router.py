import json
from unittest import case

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
            - "Verbessere mir das Folgende:" -> minio
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
    text = request.text

    model = await asyncio.gather(classify_prompt(text))

    result = await asyncio.gather(get_model_response(model, text))

    try:
        distillbert_result, gbert_result, llama_result, zephyr_result, bloom_result = await asyncio.gather(
            get_model_response(Model.DISTILBERT, text),
            get_model_response(Model.GBERT, text),
            get_model_response(Model.LLAMA, text),
            get_model_response(Model.ZEPHYR, text),
            get_model_response(Model.BLOOM, text),
        )

        prompt = f"Verbessere den folgenden Text auf Deutsch unter Berücksichtigung der sprachlichen Qualität ({distillbert_result.get('sprachliche_qualitaet')}) und der formalen Vorgaben ({gbert_result.get('formale_vorgaben')}): {text}"

        flan_result = await get_model_response(Model.FLAN, prompt)

        context = f"Sprachliche Qualität: {distillbert_result.get('sprachliche_qualitaet')}, Formale Vorgaben: {gbert_result.get('formale_vorgaben')}"
        full_text = f"{context} {text}"

        aggregated_result = {
            "generated_text": flan_result.get("generated_text"),
            "sprachliche_qualitaet": distillbert_result.get("sprachliche_qualitaet"),
            "formale_vorgaben": gbert_result.get("formale_vorgaben"),
            "gliederung": llama_result.get("gliederung"),
            "citation": zephyr_result.get("citation"),
            "corrected_grammar": bloom_result.get("checked_text"),
        }

        gewichtetes_ergebnis = 0

        sprachliche_qualitaet = aggregated_result["sprachliche_qualitaet"]
        gewichtetes_ergebnis += sprachliche_qualitaet * 0.4

        sachliche_richtigkeit = flan_result.get("sachliche_richtigkeit")
        if sachliche_richtigkeit == 1:
            gewichtetes_ergebnis += 0.5

        formale_vorgaben = aggregated_result["formale_vorgaben"]
        if formale_vorgaben == 1:
            gewichtetes_ergebnis += 0.1

        aggregated_result["gewichtetes_ergebnis"] = gewichtetes_ergebnis

        return aggregated_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")