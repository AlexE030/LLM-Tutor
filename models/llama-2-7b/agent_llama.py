from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

import torch
import os
import logging
import sys

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TextInput(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.eval()
    logging.debug("Modell set to evaluation mode.")
    yield
    torch.cuda.empty_cache()
    logging.debug("Shutdown performed successfully.")

app = FastAPI(lifespan=lifespan)


@app.post("/process/")
async def generate_outline(input: TextInput):
    print(input)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.empty_cache()

    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=10, num_beams=1, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    first_word = generated_text.split()[0] if generated_text else ""

    logging.debug(f"Full llama output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    logging.debug(f"Llama output without prompt: {generated_text}")
    logging.debug(f"relevant words: {first_word}")
    logging.debug(f"amount of tokens in prompt: {input_length}")

    torch.cuda.empty_cache()

    return {"response": first_word}