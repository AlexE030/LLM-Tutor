import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

def load_data(file_path):
    """Lädt Daten aus einer TSV-Datei mit Pandas."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def create_embeddings(text_list, model):
    """Erzeugt Vektoreinbettungen für eine Liste von Texten."""
    embeddings = model.encode(text_list)
    return embeddings

if __name__ == '__main__':
    train_data = load_data("data/in_domain_train.tsv")
    dev_data = load_data("data/in_domain_dev.tsv")
    test_data = load_data("data/out_of_domain_dev.tsv")

    if train_data is not None:
        print("Train data loaded successfully.")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_texts = train_data['sentence1'].tolist() + train_data['sentence2'].tolist()
        train_embeddings = create_embeddings(train_texts, model)
        print(f"Generated {len(train_embeddings)} training embeddings.")

        client = chromadb.Client()
        collection = client.create_collection("my_collection")
        collection.add(
            embeddings=train_embeddings.tolist(),
            documents=train_texts,
            ids=[f"train-{i}" for i in range(len(train_texts))]
        )
        print("Training embeddings added to ChromaDB.")

    if dev_data is not None:
        print("Dev data loaded successfully.")
    if test_data is not None:
        print("Test data loaded successfully.")