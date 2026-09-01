"""
Company intelligence — a separate, opt-in knowledge base about employers.

This package is deliberately isolated from the candidate pipeline. It has its own
store (Qdrant Cloud, not FAISS), its own retrieval, its own chat orchestration and
its own router. Nothing in `services/ai/` imports it, and the recruiter chatbot only
reaches it when the UI toggle is on. Deleting this package would leave the hiring
platform byte-for-byte unchanged.

Why separate rather than a new branch inside `chat_service.answer`: company records
answer a different question ("what does this employer do") against a different corpus
with a different notion of relevance. Folding them into candidate retrieval would put
two unrelated populations behind one ranking, which is exactly the kind of coupling
the platform's module boundaries exist to prevent.
"""
