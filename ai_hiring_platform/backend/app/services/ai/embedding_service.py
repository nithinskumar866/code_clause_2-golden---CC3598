from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from app.core.constants import EMBEDDING_MODEL_NAME
from app.core.logging import logger

# Lazy-loaded singleton instance
_embedding_model_instance = None

def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Returns the initialized LlamaIndex HuggingFaceEmbedding model instance (Singleton).
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            _embedding_model_instance = HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL_NAME,
                # Can specify device (cpu, cuda) - default will auto-detect
            )
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embedding model: {e}", exc_info=True)
            raise e
    return _embedding_model_instance

def generate_embeddings_for_nodes(nodes: List[TextNode]) -> List[TextNode]:
    """
    Computes embeddings for a list of TextNodes using BGE-small and attaches them.
    """
    model = get_embedding_model()
    logger.info(f"Generating embeddings for {len(nodes)} TextNodes...")
    try:
        for idx, node in enumerate(nodes):
            # get_text_embedding returns list of floats
            node.embedding = model.get_text_embedding(node.text)
        logger.info("Embeddings generated successfully.")
        return nodes
    except Exception as e:
        logger.error(f"Error during node embedding generation: {e}", exc_info=True)
        raise e
