from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "mrm8488/t5-base-finetuned-question-generation-ap"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print(f"Modell {MODEL_NAME} wurde erfolgreich heruntergeladen.")

