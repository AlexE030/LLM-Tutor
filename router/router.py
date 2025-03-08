from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import asyncio

app = FastAPI()

MODELS = {
    "distillbert": "http://distillbert_api:8000/process/",
    "gbert": "http://gbert_api:8000/process/",
    "llama": "http://llama_api:8000/process/",
    "zephyr": "http://zephyr_api:8000/process/",
    "mistral": "http://mistral_api:8000/process/",
}

class TextRequest(BaseModel):
    text: str
  # task: str

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
  # task = request.task.lower()

 #   try:
 #      if task == "zitationen":
 #          result = await get_model_response("llama", text)
 #      elif task == "grammatik":
 #          result = await get_model_response("zephyr", text)
 #      elif task == "formulierung":
 #          result = await get_model_response("bloom", text)
 #      elif task == "allgemein":
 #          result = await get_model_response("mistral", text)
 #      else:
 #          raise HTTPException(status_code=400, detail="Ungültige Aufgabenauswahl.")

 #      return result

    try:
        distillbert_result, gbert_result, llama_result, zephyr_result, mistral_result = await asyncio.gather(
            get_model_response("distillbert", text),
            get_model_response("gbert", text),
            get_model_response("llama", text),
            get_model_response("zephyr", text),
            get_model_response("mistral", text),
        )

        aggregated_result = {
            "sprachliche_qualitaet": distillbert_result.get("sprachliche_qualitaet"),
            "formale_vorgaben": gbert_result.get("formale_vorgaben"),
            "gliederung": llama_result.get("gliederung"),
            "citation": zephyr_result.get("citation"),
            "mistral": mistral_result.get("mistral"),
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