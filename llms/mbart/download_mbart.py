from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "facebook/mbart-large-50"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print(f"Modell {MODEL_NAME} wurde erfolgreich heruntergeladen.")

