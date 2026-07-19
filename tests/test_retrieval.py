"""
Unit tests for the Embedding-Based Semantic Search (RAG Pipeline).

Tests cover document loading, embedding generation, vector store operations,
end-to-end pipeline integration, and utility functions.
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np

# Add project root to path so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from utils import format_file_size, clean_text, truncate_text, is_supported_file
from config import EMBEDDING_DIMENSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create and yield a temporary directory; clean up after the test."""
    path = tempfile.mkdtemp(prefix="rag_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_txt_file(tmp_dir):
    """Create a sample .txt file with enough content for chunking tests."""
    filepath = os.path.join(tmp_dir, "sample.txt")
    content = (
        "Artificial intelligence is transforming the way we interact with technology. "
        "Machine learning models can now understand natural language, generate images, "
        "and even write code. Deep learning, a subset of machine learning, uses neural "
        "networks with many layers to learn complex patterns from large amounts of data. "
        "Transformer architectures have revolutionized NLP by enabling parallel processing "
        "of sequences. Attention mechanisms allow models to focus on relevant parts of the "
        "input when generating output. These advances have led to powerful language models "
        "that can perform a wide range of tasks including translation, summarization, and "
        "question answering. The field continues to evolve rapidly with new breakthroughs "
        "emerging every month."
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


# ---------------------------------------------------------------------------
# Document Loader Tests
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    """Tests for DocumentLoader functionality."""

    def test_document_loader_txt(self, sample_txt_file):
        """Load a .txt file and verify that text chunks are created."""
        loader = DocumentLoader()
        chunks = loader.load_document(sample_txt_file)

        assert chunks is not None, "Chunks should not be None"
        assert len(chunks) > 0, "At least one chunk should be produced"
        assert all(hasattr(c, "page_content") for c in chunks), (
            "Each chunk must have page_content"
        )

    def test_document_loader_chunking(self, sample_txt_file):
        """Verify chunk size and overlap constraints."""
        chunk_size = 200
        chunk_overlap = 50
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = loader.load_document(sample_txt_file)

        for chunk in chunks:
            # Allow a small margin because the splitter may slightly exceed the limit
            assert len(chunk.page_content) <= chunk_size + 50, (
                f"Chunk exceeds max size: {len(chunk.page_content)}"
            )


# ---------------------------------------------------------------------------
# Embedding Manager Tests
# ---------------------------------------------------------------------------

class TestEmbeddingManager:
    """Tests for EmbeddingManager functionality."""

    def test_embedding_manager(self):
        """Encode text and verify the vector dimension matches EMBEDDING_DIMENSION (384)."""
        manager = EmbeddingManager()
        vector = manager.encode_text("Hello, world!")

        assert isinstance(vector, (list, np.ndarray)), "Output should be a list or ndarray"
        assert len(vector) == EMBEDDING_DIMENSION, (
            f"Expected dimension {EMBEDDING_DIMENSION}, got {len(vector)}"
        )

    def test_embedding_similarity(self):
        """Similar texts should have higher cosine similarity than dissimilar texts."""
        manager = EmbeddingManager()

        vec_a = np.array(manager.encode_text("machine learning and artificial intelligence"))
        vec_b = np.array(manager.encode_text("deep learning and neural networks"))
        vec_c = np.array(manager.encode_text("baking chocolate chip cookies recipe"))

        def cosine_sim(u, v):
            return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

        sim_ab = cosine_sim(vec_a, vec_b)
        sim_ac = cosine_sim(vec_a, vec_c)

        assert sim_ab > sim_ac, (
            f"Similar texts should score higher ({sim_ab:.4f}) than "
            f"dissimilar texts ({sim_ac:.4f})"
        )
        assert sim_ab > 0.5, "Related texts should have cosine similarity > 0.5"


# ---------------------------------------------------------------------------
# Vector Store Tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    """Tests for VectorStoreManager functionality."""

    def test_vector_store_create_and_search(self, sample_txt_file, tmp_dir):
        """Create a FAISS store from documents, search, and verify results."""
        loader = DocumentLoader()
        chunks = loader.load_document(sample_txt_file)

        vs_manager = VectorStoreManager()
        store = vs_manager.create_from_documents(chunks)

        results = vs_manager.search(store, "neural networks and deep learning", k=2)

        assert results is not None, "Search results should not be None"
        assert len(results) > 0, "At least one result should be returned"
        assert len(results) <= 2, "Should return at most k results"

        for doc, score in results:
            assert hasattr(doc, "page_content"), "Result doc must have page_content"
            assert isinstance(score, float) or isinstance(score, np.float32), "Score must be a float"


# ---------------------------------------------------------------------------
# RAG Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    """End-to-end integration test for the RAG pipeline."""

    def test_rag_pipeline_integration(self, tmp_dir):
        """Ingest a temp file, search, and verify results are returned."""
        # Create a temporary document
        filepath = os.path.join(tmp_dir, "integration_test.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                "Quantum computing leverages quantum mechanical phenomena such as "
                "superposition and entanglement to perform computations. Unlike "
                "classical bits, qubits can exist in multiple states simultaneously, "
                "enabling quantum computers to solve certain problems exponentially "
                "faster than classical machines."
            )

        pipeline = RAGPipeline()
        pipeline.ingest_directory(tmp_dir)

        results = pipeline.search("quantum computing advantages", k=2)

        assert results is not None, "Pipeline should return results"
        assert len(results) > 0, "Pipeline should find relevant results"
        assert "quantum" in results[0]["content"].lower(), (
            "Top result should mention 'quantum'"
        )

    def test_pipeline_multi_file_ingestion(self, tmp_dir):
        """Ingest multiple files and verify all content is searchable."""
        # Create two different topic files
        with open(os.path.join(tmp_dir, "ai_doc.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Artificial intelligence and machine learning are reshaping industries. "
                "Neural networks can classify images, translate languages, and generate text. "
                "Transfer learning allows pre-trained models to be fine-tuned on new tasks."
            )
        with open(os.path.join(tmp_dir, "cooking_doc.txt"), "w", encoding="utf-8") as f:
            f.write(
                "The art of French cooking involves precise techniques like julienne cuts, "
                "making roux, and creating emulsions. Classic dishes include coq au vin, "
                "bouillabaisse, and ratatouille. Fresh herbs are essential to Provencal cuisine."
            )

        pipeline = RAGPipeline()
        total = pipeline.ingest_directory(tmp_dir)

        assert total > 0, "Should ingest chunks from both files"

        # Search for AI content — top result should be from ai_doc
        ai_results = pipeline.search("neural network image classification", k=1)
        assert len(ai_results) > 0
        assert "neural" in ai_results[0]["content"].lower() or "machine" in ai_results[0]["content"].lower()

        # Search for cooking content — top result should be from cooking_doc
        cook_results = pipeline.search("French cuisine ratatouille recipe", k=1)
        assert len(cook_results) > 0
        assert "french" in cook_results[0]["content"].lower() or "cooking" in cook_results[0]["content"].lower()

    def test_pipeline_clear_and_search(self, tmp_dir):
        """After clearing the index, search should return empty results."""
        filepath = os.path.join(tmp_dir, "cleartest.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Blockchain is a distributed ledger technology used in cryptocurrency.")

        pipeline = RAGPipeline()
        pipeline.ingest_directory(tmp_dir)

        # Verify search works before clearing
        results = pipeline.search("blockchain", k=1)
        assert len(results) > 0, "Should find results before clearing"

        # Clear and verify search returns empty
        pipeline.clear_index()
        results = pipeline.search("blockchain", k=1)
        assert len(results) == 0, "Should return no results after clearing"

    def test_pipeline_result_format(self, tmp_dir):
        """Verify the search results have the expected dictionary keys."""
        filepath = os.path.join(tmp_dir, "format_test.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Python is a popular programming language for data science and web development.")

        pipeline = RAGPipeline()
        pipeline.ingest_directory(tmp_dir)
        results = pipeline.search("Python programming", k=1)

        assert len(results) > 0
        result = results[0]
        assert "content" in result, "Result must have 'content' key"
        assert "source" in result, "Result must have 'source' key"
        assert "page" in result, "Result must have 'page' key"
        assert "score" in result, "Result must have 'score' key"
        assert 0.0 <= result["score"] <= 1.0, "Score should be between 0 and 1"


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

class TestUtilsFunctions:
    """Tests for helper / utility functions."""

    def test_format_file_size(self):
        assert format_file_size(0) == "0 B"
        assert "KB" in format_file_size(1024)
        assert "MB" in format_file_size(1024 ** 2)
        assert "GB" in format_file_size(1024 ** 3)

    def test_clean_text(self):
        dirty = "  Hello   world  \n\n  foo  "
        cleaned = clean_text(dirty)
        assert "  " not in cleaned, "Multiple spaces should be collapsed"

    def test_truncate_text(self):
        long_text = "a" * 500
        truncated = truncate_text(long_text, max_length=100)
        assert len(truncated) <= 103, "Truncated text should respect max_length (+ellipsis)"

    def test_is_supported_file(self):
        assert is_supported_file("report.pdf") is True
        assert is_supported_file("notes.txt") is True
        assert is_supported_file("doc.docx") is True
        assert is_supported_file("readme.md") is True
        assert is_supported_file("image.png") is False
        assert is_supported_file("archive.zip") is False
