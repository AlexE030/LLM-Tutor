import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import logging

# Configure logging (if not already configured)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_loader")  # Get a logger for this module

def load_data(file_path):
    """Lädt Daten aus einer TSV-Datei mit Pandas."""
    logger.debug(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t')
        logger.info(f"Successfully loaded data from: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return None

def create_embeddings(text_list, model):
    """Erzeugt Vektoreinbettungen für eine Liste von Texten."""
    logger.debug(f"Creating embeddings for {len(text_list)} texts")
    embeddings = model.encode(text_list)
    logger.debug(f"Embeddings created with shape: {embeddings.shape}")  # Log shape
    return embeddings

if __name__ == '__main__':
    train_data = load_data("../data/in_domain_train.tsv")
    dev_data = load_data("../data/in_domain_dev.tsv")
    test_data = load_data("../data/out_of_domain_dev.tsv")

    if train_data is not None:
        logger.info("Train data loaded successfully.")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_texts = train_data['sentence1'].tolist() + train_data['sentence2'].tolist()
        logger.debug(f"Number of training texts: {len(train_texts)}")
        train_embeddings = create_embeddings(train_texts, model)
        logger.info(f"Generated {len(train_embeddings)} training embeddings.")

        client = chromadb.Client()
        logger.debug("Chroma client initialized.")
        collection = client.create_collection("my_collection")
        logger.debug("Collection 'my_collection' created.")
        collection.add(
            embeddings=train_embeddings.tolist(),
            documents=train_texts,
            ids=[f"train-{i}" for i in range(len(train_texts))]
        )
        logger.info("Training embeddings added to ChromaDB.")

    if dev_data is not None:
        logger.info("Dev data loaded successfully.")
    if test_data is not None:
        logger.info("Test data loaded successfully.")