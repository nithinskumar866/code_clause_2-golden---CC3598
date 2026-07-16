class DocumentParser:
    """
    Abstract interface for Document Parser.
    """
    def parse(self, file_path: str) -> str:
        raise NotImplementedError("parse() is not implemented in Sprint 1.")


class EmbeddingProvider:
    """
    Abstract interface for generating embeddings.
    """
    def get_embedding(self, text: str) -> list[float]:
        raise NotImplementedError("get_embedding() is not implemented in Sprint 1.")


class VectorStore:
    """
    Abstract interface for vector storage (e.g. FAISS).
    """
    def add_documents(self, documents: list[dict]) -> None:
        raise NotImplementedError("add_documents() is not implemented in Sprint 1.")

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        raise NotImplementedError("search() is not implemented in Sprint 1.")


class Retriever:
    """
    Abstract interface for RAG document chunk retrieval.
    """
    def retrieve(self, query: str, index_id: str) -> list[dict]:
        raise NotImplementedError("retrieve() is not implemented in Sprint 1.")
