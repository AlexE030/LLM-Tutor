import requests

from chromaDB.chunk_processor import ChunkProcessor
from chromaDB.data_loader import DataLoader

ROUTER_API_RESET_URL = "http://localhost:8080/reset/"
pdf_path = "./data/191212_Leitlinien_Praxismodule_Studien_Bachelorarbeiten.pdf"
output_path = './data/dhbw_rules.json'

def reset_chat():
    try:
        response = requests.post(ROUTER_API_RESET_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error contacting router_api: {e}"}



chunker = ChunkProcessor(pdf_path, output_path)
chunker.chunk()
data_loader = DataLoader(output_path)
data_loader.load()
reset_chat()