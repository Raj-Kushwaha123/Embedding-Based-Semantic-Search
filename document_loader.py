# Document Loader - loads PDF, TXT, DOCX files and splits them into chunks

import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


class DocumentLoader:
    """Loads documents and splits them into smaller chunks for embedding."""

    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        # this splitter tries to split by paragraphs first, then sentences
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def load_document(self, file_path):
        """Load a single file and return list of chunks."""

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is completely empty: {os.path.basename(file_path)}")

        ext = os.path.splitext(file_path)[1].lower()

        # pick the right loader based on file type
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

        # load the raw document pages
        raw_docs = loader.load()

        # split into chunks
        chunks = self.splitter.split_documents(raw_docs)
        print(f"Loaded '{os.path.basename(file_path)}': {len(raw_docs)} page(s) -> {len(chunks)} chunk(s)")

        return chunks

    def load_directory(self, dir_path):
        """Load all supported files from a directory."""

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        all_chunks = []
        for root, dirs, files in os.walk(dir_path):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                file_path = os.path.join(root, filename)
                try:
                    chunks = self.load_document(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Could not load '{filename}': {e}")

        print(f"Loaded {len(all_chunks)} total chunks from directory")
        return all_chunks

    def load_uploaded_file(self, uploaded_file, file_name):
        """Load a file uploaded through Streamlit."""

        ext = os.path.splitext(file_name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        # save uploaded file to a temp location so loaders can read it
        tmp_path = None
        try:
            file_bytes = uploaded_file.read()
            if not file_bytes:
                raise ValueError(f"Uploaded file '{file_name}' is completely empty.")
                
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, "wb") as f:
                f.write(file_bytes)

            chunks = self.load_document(tmp_path)

            # replace the temp path with original filename in metadata
            for chunk in chunks:
                chunk.metadata["source"] = file_name

            return chunks
        finally:
            # clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_supported_extensions(self):
        """Return list of supported file extensions."""
        return list(SUPPORTED_EXTENSIONS)
