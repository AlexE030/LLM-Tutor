import json
import logging
from sentence_transformers import SentenceTransformer
import chromadb


class DataLoader:
    def __init__(self, chunked_data_path):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("data_loader")
        self.path = chunked_data_path

    def load_json_data(self):
        self.logger.debug(f"Loading JSON data from: {self.path}")
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Successfully loaded {len(data)} items from: {self.path}")
            return data
        except FileNotFoundError:
            self.logger.error(f"Error: File not found at {self.path}")
            return None

    def create_embeddings(self, text_list, model):
        self.logger.debug(f"Creating embeddings for {len(text_list)} texts")
        embeddings = model.encode(text_list)
        self.logger.debug(f"Embeddings created with shape: {embeddings.shape}")
        return embeddings

    def extract_meta_data(self, data):
        metadata_list = []
        for item in data:
            headings = ""
            tags = ""
            for heading in item["headings"]:
                headings += f"{heading}, "
            for tag in item["tags"]:
                tags += f"{tag}, "
            metadata_list.append({"headings": headings, "tags": tags})
        return metadata_list

    def insert_into_chorma_db(self, texts, embeddings, metadata_list):
        client = chromadb.HttpClient(host="chromadb", port=8000)
        self.logger.debug("Persistent Chroma client initialized.")

        # Collection erstellen oder abrufen
        collection = client.get_or_create_collection("dhbw_rules")
        self.logger.debug("Collection 'dhbw_rules' created or retrieved.")

        # IDs für jeden Chunk generieren
        ids = [f"chunk-{i}" for i in range(len(texts))]

        # Daten in ChromaDB hinzufügen
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata_list,
            ids=ids
        )
        self.logger.info("Embeddings and documents added to ChromaDB.")
        self.logger.info(f"Number of items in collection: {collection.count()}")

    def load(self):
        # JSON-Datei laden (z.B. "chunks.json")
        data = self.load_json_data()
        if data is None:
            exit(1)

        # Extrahiere Texte und Metadaten
        texts = [item["text"] for item in data]
        metadata_list = self.extract_meta_data(data)

        self.logger.debug(f"Number of texts: {len(texts)}")

        # Embeddings erstellen mit SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self.create_embeddings(texts, model)
        self.logger.info(f"Generated {len(embeddings)} embeddings.")

        self.insert_into_chorma_db(texts, embeddings, metadata_list)

