from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

app = FastAPI()

MODELS = {
    "flan-t5-base": "http://flan_t5_api:8000/process/",
    "distillbert": "http://distillbert_api:8000/process/",
    "gbert": "http://gbert_api:8000/process/",
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
        flan_result, distillbert_result, gbert_result = await asyncio.gather(
            get_model_response("flan-t5-base", text),
            get_model_response("distillbert", text),
            get_model_response("gbert", text)
        )

        aggregated_result = {
            "generated_text": flan_result.get("generated_text"),
            "sprachliche_qualitaet": distillbert_result.get("sprachliche_qualitaet"),
            "formale_vorgaben": gbert_result.get("formale_vorgaben")
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