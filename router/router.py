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
    """Ruft die Antwort eines bestimmten Modells ab."""
    try:
        response = requests.post(MODELS[model_name], json={"text": text})
        response.raise_for_status()  # Fehler auslösen, wenn der Request fehlschlägt
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
            "sprachliche_qualitaet": distillbert_result.get("sprachliche_qualitaet"),  # Korrigierter Schlüssel
            "formale_vorgaben": gbert_result.get("formale_vorgaben")  # Korrigierter Schlüssel
        }

        # Gewichtung der Ergebnisse
        gewichtetes_ergebnis = 0

        # DistilBERT: Sprachliche Qualität (0-1, 1 = best quality)
        sprachliche_qualitaet = aggregated_result["sprachliche_qualitaet"]
        gewichtetes_ergebnis += sprachliche_qualitaet * 0.4

        # Flan-T5: Sachliche Richtigkeit (0 or 1, 1 is correct, 0 is wrog)
        sachliche_richtigkeit = flan_result.get("sachliche_richtigkeit")
        if sachliche_richtigkeit == 1:
            gewichtetes_ergebnis += 0.5

        # GBERT: Einhaltung formaler Vorgaben (0 or 1, wobei 1 konform ist)
        formale_vorgaben = aggregated_result["formale_vorgaben"]
        if formale_vorgaben == 1:
            gewichtetes_ergebnis += 0.1

        aggregated_result["gewichtetes_ergebnis"] = gewichtetes_ergebnis # gewichtetes Ergebnis zum aggregated_result hinzufügen

        return aggregated_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while connecting: {e}")