# Utility functions for the project

import os
import re
from config import SUPPORTED_EXTENSIONS


def get_file_extension(file_path):
    """Get the file extension in lowercase."""
    return os.path.splitext(file_path)[1].lower()


def is_supported_file(file_path):
    """Check if file type is supported."""
    return get_file_extension(file_path) in SUPPORTED_EXTENSIONS


def format_file_size(size_bytes):
    """Convert bytes to human readable format like '2.5 MB'."""
    if size_bytes < 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            if unit == "B":
                return f"{size_bytes} {unit}"
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

    return f"{size_bytes:.2f} TB"


def clean_text(text):
    """Remove extra whitespace and clean up text."""
    if not text:
        return ""

    # replace multiple spaces with single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # replace 3+ newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def highlight_query_in_text(text, query):
    """Wrap matching query words with **bold** markers."""
    if not text or not query:
        return text

    words = query.split()
    if not words:
        return text

    # build pattern to match any query word
    escaped = [re.escape(w) for w in words]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    return pattern.sub(r"**\1**", text)


def truncate_text(text, max_length=300):
    """Shorten text to max_length and add ... if needed."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def get_file_icon(extension):
    """Return an emoji icon for the file type."""
    icons = {
        ".pdf": "📄",
        ".txt": "📝",
        ".md": "📑",
        ".docx": "📃",
    }
    return icons.get(extension.lower(), "📎")
