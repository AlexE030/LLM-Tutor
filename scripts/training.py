import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, pipeline


def train(model, tokenizer):
    df = pd.read_csv("data/english_german.csv")

    df['label'] = 1

    negatives = df.sample(frac=0.2).copy()  # 20% of data for negative Examples
    negatives['German'] = negatives['German'].sample(frac=1).reset_index(drop=True)
    negatives['label'] = 0

    df = pd.concat([df, negatives])

    train, val = train_test_split(df, test_size=0.2, stratify=df['label'])

    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)




    train_data = load_dataset("csv", data_files="data/train.csv")["train"]
    val_data = load_dataset("csv", data_files="data/val.csv")["train"]

    def tokenize_function(examples):
        return tokenizer(
            #examples["English"],
            examples["German"],
            return_tensors="np"
        )

    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)




    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )


    trainer.train()


    model.save_pretrained("./distilbert-finetuned")
    tokenizer.save_pretrained("./distilbert-finetuned")




    nlp = pipeline("text-classification", model="./distilbert-finetuned", tokenizer=tokenizer)

    result = nlp({"This is a test. [SEP] Das ist ein Test."})
    print(result)  # Ausgabe: [{'label': 'LABEL_1', 'score': 0.98}]
