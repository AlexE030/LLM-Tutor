from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retriever")

class Retriever:
    def __init__(self, collection_name="my_collection"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug(f"Retriever initialized with collection: {self.collection.name}")

    def retrieve_relevant_documents(self, query, top_n=5):
        logger.debug(f"Retrieving documents for query: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        logger.debug(f"Retrieved documents: {results}")
        return results

if __name__ == '__main__':
    retriever = Retriever("dhbw_rules")
    query = "Was sind die Vorteile von chromaDB?"
    relevant_docs = retriever.retrieve_relevant_documents(query)
    print("Relevant documents:", relevant_docs)