from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

import os
import sys
import torch
import logging

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

logger = logging.getLogger("agent_mistral")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TextInput(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.eval()
    logger.debug("Modell set to evaluation mode.")
    yield
    torch.cuda.empty_cache()
    logger.debug("Shutdown performed successfully.")

app = FastAPI(lifespan=lifespan)


@app.post("/process/")
async def generate_outline(input: TextInput):
    prompt = (
        "You are a highly specialized expert in creating outlines for academic papers. "
        "Your sole task is to generate detailed and logically structured outlines for scientific topics. "
        "Under no circumstances should you go beyond this task and provide any additional explanations or text. "
        "The outline must be a logical thread for the paper.\n"
        "Consider the following criteria when creating the outline:\n\n"
        "1.  **Target Audience Orientation:** Consider that the outline must be relevant to students and researchers.\n"
        "2.  **Logical Structure:** Ensure that the outline has a clear and logical structure that guides the reader through the topic.\n"
        "3.  **Completeness:** The outline should cover all relevant aspects of the topic.\n"
        "4.  **Hierarchy:** Use a clear hierarchy (e.g., Roman numerals, Arabic numerals, letters) to structure main and sub-points.\n"
        "5.  **Precision:** Formulate the outline points precisely and meaningfully, so that the content of the respective sections is clearly recognizable.\n"
        "6.  **Thematic Relevance:** Ensure that each outline point contributes directly to the topic of the academic paper.\n"
        "7. **Up-to-dateness:** Include current research results where possible.\n"
        "8. **Methodology:** If mentioned in the user input, consider the research methodology used in the outline.\n"
        "9. **Research Question:** Make sure the outline logically addresses and answers the research question.\n"
        "10. **Sources:** If sources are provided, include them.\n\n"
        "Create a detailed outline on the following topic:\n\n"
        f"Benutzereingabe: {input.text}\n\n"
        "Provide only the outline without any further explanations."

    )
    print(prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.empty_cache()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, max_new_tokens=512, num_beams=1, early_stopping=True)
    generated_tokens = outputs[0][input_length:]
    output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    torch.cuda.empty_cache()

    return {"response": output}