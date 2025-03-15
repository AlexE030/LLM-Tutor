import chromadb
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, collection_name="my_collection"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve_relevant_documents(self, query, top_n=5):
        """
        Ruft die relevantesten Dokumente aus ChromaDB basierend auf der Anfrage ab.

        Args:
            query (str): Die Suchanfrage.
            top_n (int): Die Anzahl der zurückzugebenden Dokumente.

        Returns:
            list: Eine Liste von Dokumenten, die der Anfrage am ähnlichsten sind.
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )
        return results["documents"][0]

if __name__ == '__main__':
    retriever = Retriever()
    query = "Was sind die Vorteile von ChromaDB?"
    relevant_docs = retriever.retrieve_relevant_documents(query)
    print("Relevant documents:", relevant_docs)