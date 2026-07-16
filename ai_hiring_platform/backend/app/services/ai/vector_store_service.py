import os
import faiss
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from app.core.constants import VECTOR_STORE_DIR
from app.core.logging import logger
from app.services.ai.embedding_service import get_embedding_model

def get_persist_dir(resume_id: int) -> str:
    """
    Returns the vector index path for a specific resume ID.
    """
    persist_dir = os.path.join(VECTOR_STORE_DIR, f"resume_{resume_id}")
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir

def has_existing_index(resume_id: int) -> bool:
    """
    Checks if a persistent index exists on disk.
    LlamaIndex FAISS vector store serializes to 'default__vector_store.json' or 'index.faiss'.
    """
    persist_dir = get_persist_dir(resume_id)
    # Check for core LlamaIndex serialization file
    required_file = os.path.join(persist_dir, "index_store.json")
    return os.path.exists(required_file)

def save_nodes_to_index(nodes: List[TextNode], resume_id: int) -> VectorStoreIndex:
    """
    Builds a FAISS index from TextNodes and persists it to disk.
    Dimension for BGE-small embeddings is 384.
    """
    persist_dir = get_persist_dir(resume_id)
    logger.info(f"Creating new FAISS vector store index under {persist_dir}")
    
    try:
        # BGE-small embedding dimension is 384
        d = 384
        # Flat Inner Product index for cosine similarity ranking
        faiss_index = faiss.IndexFlatIP(d)
        
        # Configure LlamaIndex FaissVectorStore
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Bind the embedding model to LlamaIndex globally/index scope
        embed_model = get_embedding_model()
        
        index = VectorStoreIndex(
            nodes=nodes, 
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Save to disk
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Successfully persisted FAISS index for Resume ID: {resume_id}")
        return index
    except Exception as e:
        logger.error(f"Failed to create and save FAISS index for Resume ID {resume_id}: {e}", exc_info=True)
        raise e

def load_index(resume_id: int) -> Optional[VectorStoreIndex]:
    """
    Loads an existing FAISS index from disk. Returns None if it doesn't exist.
    """
    if not has_existing_index(resume_id):
        logger.info(f"No existing vector index found for Resume ID: {resume_id}")
        return None
        
    persist_dir = get_persist_dir(resume_id)
    logger.info(f"Loading existing FAISS index from {persist_dir}...")
    
    try:
        # Set up storage context with loaded FaissVectorStore
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            vector_store=vector_store
        )
        
        embed_model = get_embedding_model()
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model
        )
        logger.info(f"Successfully loaded index for Resume ID: {resume_id}")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index for Resume ID {resume_id}: {e}", exc_info=True)
        return None
