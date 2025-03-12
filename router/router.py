from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

app = FastAPI()

MODELS = {
    "llama": "http://llama_api:8000/process/",
    "zephyr": "http://zephyr_api:8000/process/",
    "mistral": "http://mistral_api:8000/process/",
    "bloom": "http://bloom_api:8000/process/",
}

class TextRequest(BaseModel):
    text: str

async def get_model_response(model_name: str, text: str):
    try:
        response = requests.post(MODELS[model_name], json={"text": text})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Aufruf von {model_name}: {e}")

@app.post("/process/")
async def process_text(request: TextRequest):
    text = request.text

    try:
        llama_result, zephyr_result, bloom_result, mistral_result = await asyncio.gather(
            get_model_response("llama", text),
            get_model_response("zephyr", text),
            get_model_response("bloom", text),
            get_model_response("mistral", text),
        )

        aggregated_result = {
            "gliederung": llama_result.get("gliederung"),
            "citation": zephyr_result.get("citation"),
            "mistral": mistral_result.get("mistral"),
            "corrected_grammar": bloom_result.get("checked_text"),
        }

        gewichtetes_ergebnis = 0

        sprachliche_qualitaet = aggregated_result["sprachliche_qualitaet"]
        gewichtetes_ergebnis += sprachliche_qualitaet * 0.4

        formale_vorgaben = aggregated_result["formale_vorgaben"]
        if formale_vorgaben == 1:
            gewichtetes_ergebnis += 0.1

        aggregated_result["gewichtetes_ergebnis"] = gewichtetes_ergebnis

        return aggregated_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")