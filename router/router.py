from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

app = FastAPI()

MODELS = {
    "flan-t5-base": "http://flan_t5_api:8000/process/",
    "distillbert": "http://distillbert_api:8000/process/",
    "gbert": "http://gbert_api:8000/process/",
    "llama": "http://llama_api:8000/process/",
    "zephyr": "http://zephyr_api:8000/process/",
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
        distillbert_result, gbert_result, llama_result, zephyr_result, bloom_result = await asyncio.gather(
            get_model_response("distillbert", text),
            get_model_response("gbert", text),
            get_model_response("llama", text),
            get_model_response("zephyr", text),
            get_model_response("bloom", text),
        )

        prompt = f"Verbessere den folgenden Text auf Deutsch unter Berücksichtigung der sprachlichen Qualität ({distillbert_result.get('sprachliche_qualitaet')}) und der formalen Vorgaben ({gbert_result.get('formale_vorgaben')}): {text}"

        flan_result = await get_model_response("flan-t5-base", prompt)

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