# Embedding Manager - converts text into vector embeddings using sentence-transformers

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingManager:
    """Manages the embedding model for converting text to vectors."""

    def __init__(self, model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL
        self._hf_embeddings = None  # for langchain integration
        self._st_model = None       # for direct encoding

    def get_embeddings(self):
        """Get a LangChain-compatible embedding object.
        Used when creating FAISS vector store."""

        if self._hf_embeddings is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._hf_embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("Model loaded successfully!")

        return self._hf_embeddings

    def _load_st_model(self):
        """Load the sentence-transformer model for direct encoding."""
        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    def encode_text(self, text):
        """Convert a single text string into a vector (list of numbers)."""

        if not text or not text.strip():
            # return zero vector for empty input
            return [0.0] * EMBEDDING_DIMENSION

        model = self._load_st_model()
        vector = model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def encode_batch(self, texts):
        """Convert multiple text strings into vectors at once."""

        if not texts:
            return []

        model = self._load_st_model()
        vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
        return [vec.tolist() for vec in vectors]

    def get_model_info(self):
        """Return basic info about the embedding model."""
        model = self._load_st_model()
        return {
            "name": self.model_name,
            "dimension": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
        }
