import os
import hashlib
import faiss
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from app.core.constants import VECTOR_STORE_DIR
from app.core.logging import logger
from app.services.ai.embedding_service import get_embedding_model

# Sidecar recording which source file an index was built from. Guards against a
# resume_id reused across databases (or a wiped DB over a persisted vector store)
# silently loading a DIFFERENT resume's embeddings — which would evaluate the wrong
# candidate. The index is content-addressed: rebuild when the fingerprint differs.
_FINGERPRINT_FILE = "source_fingerprint.txt"


def get_persist_dir(resume_id: int) -> str:
    """
    Returns the vector index path for a specific resume ID.
    """
    persist_dir = os.path.join(VECTOR_STORE_DIR, f"resume_{resume_id}")
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def compute_fingerprint(resume_path: str) -> str:
    """Content fingerprint (sha256 of the file bytes) identifying the exact source doc."""
    try:
        with open(resume_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:32]
    except OSError:
        return ""


def _fingerprint_path(resume_id: int) -> str:
    return os.path.join(get_persist_dir(resume_id), _FINGERPRINT_FILE)


def has_existing_index(resume_id: int) -> bool:
    """
    Checks if a persistent index exists on disk.
    LlamaIndex FAISS vector store serializes to 'default__vector_store.json' or 'index.faiss'.
    """
    persist_dir = get_persist_dir(resume_id)
    # Check for core LlamaIndex serialization file
    required_file = os.path.join(persist_dir, "index_store.json")
    return os.path.exists(required_file)


def has_valid_index(resume_id: int, fingerprint: Optional[str] = None) -> bool:
    """
    True only if an index exists AND (no fingerprint requested, or the persisted
    fingerprint matches the current source file). A legacy index with no recorded
    fingerprint is treated as invalid when a fingerprint is supplied, forcing a
    one-time rebuild that guarantees the index matches the actual resume.
    """
    if not has_existing_index(resume_id):
        return False
    if not fingerprint:
        return True
    fp_path = _fingerprint_path(resume_id)
    if not os.path.exists(fp_path):
        return False
    try:
        with open(fp_path, encoding="utf-8") as f:
            return f.read().strip() == fingerprint
    except OSError:
        return False

def save_nodes_to_index(nodes: List[TextNode], resume_id: int, fingerprint: Optional[str] = None) -> VectorStoreIndex:
    """
    Builds a FAISS index from TextNodes and persists it to disk.
    Dimension for BGE-small embeddings is 384. When ``fingerprint`` is given it is
    recorded alongside the index so future loads can verify the source matches.
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
        # Record the source fingerprint so a later load can prove it matches this file.
        if fingerprint:
            with open(_fingerprint_path(resume_id), "w", encoding="utf-8") as f:
                f.write(fingerprint)
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
