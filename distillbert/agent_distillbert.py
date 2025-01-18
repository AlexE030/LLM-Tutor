from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the model and tokenizer
model_name = "distilbert/distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Example functionality
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    return outputs.logits.argmax().item()

if __name__ == "__main__":
    # Test the model
    text = "This is an example for text classification."
    print("Input:", text)
    print("Predicted class:", classify(text))
