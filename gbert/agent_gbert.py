from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
model_name = "deepset/gbert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Example functionality
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.logits.argmax().item()

if __name__ == "__main__":
    # Test the model
    text = "Das ist ein Beispiel für Textklassifikation."
    print("Input:", text)
    print("Predicted class:", classify(text))
