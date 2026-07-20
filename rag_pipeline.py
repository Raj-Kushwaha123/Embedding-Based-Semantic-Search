import os
from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStoreManager
from config import VECTORSTORE_DIR
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class RAGPipeline:
    """Main pipeline that orchestrates the RAG components."""

    def __init__(self):
        self.doc_loader = DocumentLoader()
        self.emb_manager = EmbeddingManager()
        self.vs_manager = VectorStoreManager(self.emb_manager)
        
        # Load existing index if available
        self.vectorstore = self.vs_manager.load(VECTORSTORE_DIR)
        
        self.total_docs_indexed = 0
        if self.vectorstore is not None:
            # We don't have a direct way to count original docs from FAISS,
            # so we'll just track it for this session or estimate
            self.total_docs_indexed = 1 # Placeholder if loaded from disk

    def ingest_uploaded_files(self, uploaded_files):
        """Ingest files uploaded via Streamlit."""
        all_chunks = []
        for file in uploaded_files:
            chunks = self.doc_loader.load_uploaded_file(file, file.name)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"num_chunks": 0, "num_documents": len(uploaded_files)}

        if self.vectorstore is None:
            logger.info("Initializing new vector store with uploaded files")
            self.vectorstore = self.vs_manager.create_from_documents(all_chunks)
        else:
            logger.info("Adding uploaded files to existing vector store")
            self.vectorstore = self.vs_manager.add_documents(self.vectorstore, all_chunks)
            
        # Save after modifying
        self.vs_manager.save(self.vectorstore, VECTORSTORE_DIR)
        self.total_docs_indexed += len(uploaded_files)

        return {
            "num_chunks": len(all_chunks),
            "num_documents": len(uploaded_files)
        }
        
    def ingest_directory(self, dir_path, batch_size=10):
        """Ingest all supported documents from a directory.
        
        Uses batch processing for large collections to keep memory usage low.
        
        Args:
            dir_path: Path to directory containing documents.
            batch_size: Number of files per batch (default: 10).
            
        Returns:
            Total number of chunks ingested.
        """
        total_chunks = 0
        
        for batch in self.doc_loader.load_directory_batched(dir_path, batch_size=batch_size):
            chunks = batch["chunks"]
            if not chunks:
                continue
                
            if self.vectorstore is None:
                logger.info(f"Creating new vector store from batch {batch['batch_number']}")
                self.vectorstore = self.vs_manager.create_from_documents(chunks)
            else:
                logger.info(f"Adding batch {batch['batch_number']} to existing vector store")
                self.vectorstore = self.vs_manager.add_documents(self.vectorstore, chunks)
            
            total_chunks += len(chunks)
        
        if total_chunks > 0:
            self.vs_manager.save(self.vectorstore, VECTORSTORE_DIR)
            logger.info(f"Directory ingestion complete: {total_chunks} total chunks indexed")
        
        return total_chunks

    def search(self, query, k=5):
        """Search the vector store and return formatted results.
        
        Handles edge cases like empty queries, special characters,
        and missing index gracefully.
        """
        # Edge case: empty or whitespace-only query
        if not query or not query.strip():
            logger.warning("Empty search query received")
            return []
        
        # Edge case: no index loaded
        if self.vectorstore is None:
            logger.warning(f"Search attempted for query '{query}' but no index is loaded")
            return []

        # Sanitize query: remove special regex characters that could cause issues
        clean_query = re.sub(r'[^\w\s\-\'\"\?\.,]', ' ', query.strip())
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        if not clean_query:
            logger.warning(f"Query '{query}' reduced to empty after sanitization")
            return []

        logger.info(f"Executing search for query: '{clean_query}' (top_k={k})")
        
        try:
            raw_results = self.vs_manager.search(self.vectorstore, clean_query, k=k)
        except Exception as e:
            logger.error(f"FAISS search failed for query '{clean_query}': {e}")
            return []
        
        formatted_results = []
        for doc, score in raw_results:
            # For normalized embeddings, FAISS L2 distance squared is related to cosine similarity:
            # L2_sq = 2 - 2 * cosine_sim => cosine_sim = 1 - (L2_sq / 2)
            # This gives a proper score between -1 and 1. We'll clip it to 0-1 for UI display.
            cosine_sim = 1.0 - (float(score) / 2.0)
            normalized_score = max(0.0, min(1.0, cosine_sim))
            
            source = doc.metadata.get("source", "Unknown")
            # PyPDFLoader usually puts page in metadata
            page = doc.metadata.get("page", 1) 
            
            formatted_results.append({
                "content": doc.page_content,
                "source": os.path.basename(source),
                "page": page,
                "score": normalized_score
            })
            
        return formatted_results

    def get_stats(self):
        """Get statistics about the current index."""
        if self.vectorstore is None:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "index_loaded": False
            }
            
        stats = self.vs_manager.get_index_stats(self.vectorstore)
        return {
            "total_chunks": stats.get("num_vectors", 0),
            "total_documents": self.total_docs_indexed,
            "index_loaded": True
        }

    def clear_index(self):
        """Clear the vector store."""
        logger.info("Clearing the vector store index and deleting physical files")
        self.vectorstore = None
        self.total_docs_indexed = 0
        
        # Remove physical files if they exist
        if os.path.exists(VECTORSTORE_DIR):
            for file in os.listdir(VECTORSTORE_DIR):
                file_path = os.path.join(VECTORSTORE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
