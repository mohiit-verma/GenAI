import re
import requests
from typing import List, Dict, Optional, Tuple, Union
from PyPDF2 import PdfReader
import io
import os
import hashlib

# -----------------------------
# URL processing & downloading
# -----------------------------
def process_book_url(url: str) -> str:
    """
    Normalize a book URL so it's directly downloadable as a PDF.
    - Converts GitHub 'blob' URLs to 'raw' URLs.
    - Returns the original URL if no changes are needed.
    """
    # GitHub "blob" URL -> "raw" URL
    gh_blob = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if gh_blob:
        user, repo, path = gh_blob.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"

    # GitHub "tree" URLs sometimes appear; not a direct file
    gh_tree = re.match(r"https?://github\.com/([^/]+)/([^/]+)/tree/(.+)", url)
    if gh_tree:
        raise ValueError(
            "GitHub 'tree' URL provided. Please point to a specific file (PDF) or use a 'blob' link."
        )

    # Google Drive 'view' links, Dropbox share links, etc. could be added here if needed.
    return url


def download_pdf(
    url: str,
    save_path: Optional[str] = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> Tuple[bytes, Optional[str]]:
    """
    Download a PDF from a (processed) URL.
    Returns (pdf_bytes, saved_path).
    - If save_path is provided, writes the PDF to disk and returns that path.
    - Raises requests.HTTPError for non-200 responses.
    """
    processed = process_book_url(url)
    resp = requests.get(processed, stream=True, timeout=timeout, verify=verify_ssl)
    resp.raise_for_status()

    # Read into memory
    pdf_bytes = resp.content

    # Optionally save to disk
    saved = None
    if save_path:
        # Ensure .pdf extension if missing
        if not save_path.lower().endswith(".pdf"):
            save_path = save_path + ".pdf"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(pdf_bytes)
        saved = save_path

    return pdf_bytes, saved


# --------------------------------
# PDF reading & page text extraction
# --------------------------------
def _open_pdf(pdf_source: Union[str, bytes, io.BytesIO]) -> PdfReader:
    """
    Internal helper to open a PDF from a path or bytes-like object.
    """
    if isinstance(pdf_source, str):
        # Path to file on disk
        with open(pdf_source, "rb") as f:
            return PdfReader(f)
    elif isinstance(pdf_source, bytes):
        return PdfReader(io.BytesIO(pdf_source))
    elif isinstance(pdf_source, io.BytesIO):
        return PdfReader(pdf_source)
    else:
        raise TypeError("pdf_source must be a file path (str), bytes, or BytesIO.")


def read_pdf_pages(
    pdf_source: Union[str, bytes, io.BytesIO],
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> List[Dict[str, Union[int, str]]]:
    """
    Read a PDF and return a list of dicts: [{"page_number": int, "text": str}, ...]
    - page_number is 1-based (useful for human-readable citations).
    - start_page / end_page are 1-based inclusive bounds. If None, read the whole file.
    - Extracted text may be empty ('') for image-only pages; handle gracefully.
    """
    reader = _open_pdf(pdf_source)
    n = len(reader.pages)

    sp = 1 if start_page is None else max(1, start_page)
    ep = n if end_page is None else min(end_page, n)
    if sp > ep:
        return []

    pages = []
    for i in range(sp - 1, ep):  # convert to 0-based
        page = reader.pages[i]
        text = page.extract_text() or ""  # fall back to empty string
        # Normalize whitespace a bit (optional)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append({"page_number": i + 1, "text": text.strip()})

    return pages


def get_pdf_page_numbers(pdf_source: Union[str, bytes, io.BytesIO]) -> List[int]:
    """
    Return a simple list of page numbers [1, 2, ..., N].
    Useful when you want to attach page metadata during vector indexing.
    """
    reader = _open_pdf(pdf_source)
    return list(range(1, len(reader.pages) + 1))



def stable_pdf_id(pdf_bytes: bytes) -> str:
    """
    Deterministic ID for the PDF content, handy for caching or reusing vector indexes.
    """
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]

