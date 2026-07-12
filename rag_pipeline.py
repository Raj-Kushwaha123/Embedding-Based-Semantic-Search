import os
from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStoreManager
from config import VECTORSTORE_DIR


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
            self.vectorstore = self.vs_manager.create_from_documents(all_chunks)
        else:
            self.vectorstore = self.vs_manager.add_documents(self.vectorstore, all_chunks)
            
        # Save after modifying
        self.vs_manager.save(self.vectorstore, VECTORSTORE_DIR)
        self.total_docs_indexed += len(uploaded_files)

        return {
            "num_chunks": len(all_chunks),
            "num_documents": len(uploaded_files)
        }
        
    def ingest_directory(self, dir_path):
        """Ingest all supported documents from a directory."""
        chunks = self.doc_loader.load_directory(dir_path)
        if not chunks:
            return 0
            
        if self.vectorstore is None:
            self.vectorstore = self.vs_manager.create_from_documents(chunks)
        else:
            self.vectorstore = self.vs_manager.add_documents(self.vectorstore, chunks)
            
        self.vs_manager.save(self.vectorstore, VECTORSTORE_DIR)
        return len(chunks)

    def search(self, query, k=5):
        """Search the vector store and return formatted results."""
        if self.vectorstore is None:
            return []

        raw_results = self.vs_manager.search(self.vectorstore, query, k=k)
        
        formatted_results = []
        for doc, score in raw_results:
            # Convert FAISS L2 distance to a pseudo-similarity score (0 to 1)
            # Smaller distance is better. This is a simple inversion.
            sim_score = 1.0 / (1.0 + float(score))
            
            source = doc.metadata.get("source", "Unknown")
            # PyPDFLoader usually puts page in metadata
            page = doc.metadata.get("page", 1) 
            
            formatted_results.append({
                "content": doc.page_content,
                "source": os.path.basename(source),
                "page": page,
                "score": sim_score
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
        self.vectorstore = None
        self.total_docs_indexed = 0
        
        # Remove physical files if they exist
        if os.path.exists(VECTORSTORE_DIR):
            for file in os.listdir(VECTORSTORE_DIR):
                file_path = os.path.join(VECTORSTORE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
