from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example functionality
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test the model
    text = "This is a simple example. We will test FLAN-T5 summarization."
    print("Original:", text)
    print("Summarized:", summarize(text))
