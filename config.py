# Configuration settings for Semantic Search project

import os

# project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# embedding model - using a small free model that works on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# chunking settings - how to split documents into smaller pieces
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# how many results to return when searching
TOP_K = 5

# file types we can handle
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]

# app display settings
APP_TITLE = "Semantic Search Engine"
APP_DESCRIPTION = "AI-powered document search using vector embeddings"
