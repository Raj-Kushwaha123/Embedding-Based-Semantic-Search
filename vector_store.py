# Vector Store - stores and searches document embeddings using FAISS

import os
from langchain_community.vectorstores import FAISS
from config import TOP_K, VECTORSTORE_DIR
from embedding_manager import EmbeddingManager


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Manages the FAISS vector store for storing and searching embeddings."""

    def __init__(self, embedding_manager=None):
        self.emb_manager = embedding_manager or EmbeddingManager()
        self.embeddings = self.emb_manager.get_embeddings()

    def create_from_documents(self, documents):
        """Create a new FAISS index from a list of document chunks."""

        if not documents:
            logger.error("Attempted to create vector store from empty document list")
            raise ValueError("Cannot create vector store from empty document list")

        logger.info(f"Creating FAISS index from {len(documents)} chunks...")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        logger.info("FAISS index created successfully!")

        return vectorstore

    def add_documents(self, vectorstore, documents):
        """Add more documents to an existing FAISS index."""

        if not documents:
            logger.warning("No documents provided to add_documents method")
            return vectorstore

        logger.info(f"Adding {len(documents)} chunks to index...")
        vectorstore.add_documents(documents)
        logger.info("Documents added successfully!")

        return vectorstore

    def save(self, vectorstore, path=None):
        """Save the FAISS index to disk so we can load it later."""

        save_dir = path or VECTORSTORE_DIR
        try:
            os.makedirs(save_dir, exist_ok=True)
            vectorstore.save_local(save_dir)
            logger.info(f"Index successfully saved to {save_dir}")
        except Exception as e:
            logger.error(f"Failed to save index to {save_dir}: {str(e)}")
            raise

    def load(self, path=None):
        """Load a previously saved FAISS index from disk."""

        load_dir = path or VECTORSTORE_DIR
        index_file = os.path.join(load_dir, "index.faiss")

        if not os.path.isfile(index_file):
            logger.info(f"No saved index found at {load_dir}")
            return None

        logger.info(f"Loading index from {load_dir}...")
        try:
            vectorstore = FAISS.load_local(
                load_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Index loaded successfully!")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading FAISS index from {load_dir}: {str(e)}")
            return None
        return vectorstore

    def search(self, vectorstore, query, k=None):
        """Search for similar documents. Returns list of (document, score) pairs."""

        num_results = k or TOP_K
        results = vectorstore.similarity_search_with_score(query, k=num_results)
        return results

    def get_index_stats(self, vectorstore):
        """Get basic stats about the FAISS index."""
        index = vectorstore.index
        return {
            "num_vectors": index.ntotal,
            "dimension": index.d,
        }
