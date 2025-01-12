from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "facebook/mbart-large-50"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_fast=False)

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json.get("text", "")
    inputs = tokenizer(data, return_tensors="pt")
    outputs = model(**inputs)
    return jsonify({"result": outputs.logits.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)