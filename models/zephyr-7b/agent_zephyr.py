from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

import torch
import logging
import sys


MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

logger = logging.getLogger("agent_llama")
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
        "You are a highly specialized expert in generating citations for academic works. "
        "Your sole task is to create accurate citations based on a defined citation style. "
        "Under no circumstances should you provide any additional explanations or text beyond the citation itself. "
        "Focus exclusively on generating the citation and adhering to the given style.\n"
        "Please follow these guidelines when creating citations:\n\n"
        "1.  **Strict Adherence to Style:** Follow the specified citation style precisely (e.g., APA, MLA, Chicago, Harvard). Pay attention to every detail, including punctuation, capitalization, and formatting.\n"
        "2.  **Comprehensive Information:** Ensure all necessary information (author, title, publication date, journal/book title, volume, issue, page numbers, DOI/URL) is included in the citation.\n"
        "3.  **Accuracy:** Double-check the accuracy of all information to avoid errors in the citation.\n"
        "4.  **Consistency:** Maintain consistency in formatting and style throughout the citations.\n"
        "5.  **Handling Different Source Types:** Adapt the citation format to different source types (books, journal articles, websites, etc.).\n"
        "6.  **In-Text vs. Bibliography:** If specified, differentiate between in-text citations and full bibliography entries.\n"
        "7.  **Multiple Authors/Editors:** Handle citations with multiple authors or editors correctly.\n"
        "8.  **Missing Information:** If information is missing, indicate it appropriately (e.g., 'n.d.' for no date).\n"
        "9. **Date Formatting:** Format dates according to the specified citation style.\n"
        "10. **Digital Object Identifiers (DOIs) and URLs:** Include DOIs or URLs when available.\n"
        "11. **Edition Information:** add edition information if it is contained in the Userinput.\n"
        "12. **Publisher information:** Include Publisher information for book citations. \n"
        "13. **Volume and Issue Information:** correctly add Volume and issue information for journal citations.\n"
        "14. **Page Ranges:** correctly display the Page ranges for articles and books.\n\n"
        "Here is the user input and the specified citation style:\n\n"
        f"Benutzereingabe: {input.text}\n"
        "Provide only the citation, without any additional explanations."
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
