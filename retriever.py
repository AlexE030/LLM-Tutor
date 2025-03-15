import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Configure logging (if not already configured)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retriever")  # Get a logger for this module

class Retriever:
    def __init__(self, collection_name="my_collection"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug(f"Retriever initialized with collection: {self.collection.name}")

    def retrieve_relevant_documents(self, query, top_n=5):
        """
        Ruft die relevantesten Dokumente aus ChromaDB basierend auf der Anfrage ab.

        Args:
            query (str): Die Suchanfrage.
            top_n (int): Die Anzahl der zurückzugebenden Dokumente.

        Returns:
            list: Eine Liste von Dokumenten, die der Anfrage am ähnlichsten sind.
        """
        logger.debug(f"Retrieving documents for query: {query}")
        query_embedding = self.embedding_model.encode(query).tolist()
        logger.debug(f"Query embedding: {query_embedding}")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )
        logger.debug(f"ChromaDB results: {results}")
        retrieved_docs = results["documents"][0]
        logger.debug(f"Retrieved documents: {retrieved_docs}")
        return retrieved_docs

if __name__ == '__main__':
    retriever = Retriever()
    query = "Was sind die Vorteile von ChromaDB?"
    relevant_docs = retriever.retrieve_relevant_documents(query)
    print("Relevant documents:", relevant_docs)