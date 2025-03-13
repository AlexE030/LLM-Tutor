from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
import sys

app = FastAPI()

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
)

logger = logging.getLogger("router")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TextInput(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tokenizer
    model.eval()


@app.post("/process/")
async def generate_outline(input: TextInput):
    print(input)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.empty_cache()

    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=1, num_beams=1, early_stopping=True)
    generated_tokens = outputs[0][input_length:]
    output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    logger.debug(f"Full llama output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    logger.debug(f"Llama output without prompt: {output}")
    logger.debug(f"amount of tokens in prompt: {input_length}")

    torch.cuda.empty_cache()

    return {"response": output}