from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print(f"Model {MODEL_NAME} successfully downloaded.")