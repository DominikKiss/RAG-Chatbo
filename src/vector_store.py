import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        print("Beágyazó modell betöltése")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("ChromaDb vektortár inicializálása")
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("rag_documents")
        print("Vektortár kész!")

    def add_documents(self,documents):
        
        embeddings = self.embedding_model.encode(documents).tolist()

        ids = [f"document{i}" for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"{len(documents)} hozzáadva a vektortárhoz!")

    def search(self,query, k = 3):

        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        return results["documents"][0]
    

    def get_collection_info(self):

        count = self.collection.count()
        return {
            "document_count": count,
            "embedding_dimension": 384
        }
