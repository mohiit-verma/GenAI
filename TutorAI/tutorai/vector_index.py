from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.docstore.document import Document
import hashlib

@dataclass
class ChunkingConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    separators: Optional[List[str]] = None  # override if you want custom split rules


class TutorAIVectorIndex:
    """
    Minimal vector index wrapper for RAG over a single book.
    - Chunking via RecursiveCharacterTextSplitter
    - Embeddings via sentence-transformers/all-MiniLM-L6-v2 (normalized)
    - Vector store: FAISS
    """
    def __init__(
        self,
        chunk_cfg: ChunkingConfig = ChunkingConfig(),
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,  # e.g., "cuda" or "cpu"
    ):
        # Handle common misspelling "allminilm-v6"
        if embedding_model.lower().replace("-", "").replace("_", "") in {
            "allminilmv6", "allminilml6v2"
        }:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_cfg.chunk_size,
            chunk_overlap=chunk_cfg.chunk_overlap,
            separators=chunk_cfg.separators,  # if None, uses sensible defaults
            length_function=len,
        )

        # Embeddings (normalized → cosine-friendly)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device} if device else {},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vs: Optional[FAISS] = None
        self._book_fingerprint: Optional[str] = None

    # 1) Generate chunks using RecursiveCharacterTextSplitter
    def make_chunks(
        self,
        pages: List[Dict[str, Any]],
        source_id: Optional[str] = None,   # e.g., stable_pdf_id from your utils
    ) -> List[Document]:
        """
        Input: pages = [{"page_number": int, "text": str}, ...] from your utility.
        Output: List[Document] with preserved page_number for citations.
        """
        docs: List[Document] = []
        for p in pages:
            text = (p.get("text") or "").strip()
            if not text:
                continue
            page_no = int(p.get("page_number", 0))
            # One doc per page before splitting
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": page_no,
                        "source_id": source_id,
                    },
                )
            )

        # Split into chunks (preserving metadata on each split)
        chunk_docs = self.splitter.split_documents(docs)

        # Add a stable chunk id (handy later for tracing)
        for i, d in enumerate(chunk_docs):
            page_no = d.metadata.get("page_number", 0)
            base = f"{source_id or 'book'}|p{page_no}|{i}|{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}"
            d.metadata["chunk_id"] = base

        return chunk_docs

    # 2) Index chunks with FAISS (all-MiniLM-L6-v2, normalized)
    def build_index(self, chunk_docs: List[Document]) -> None:
        """
        Build/replace the index from chunk documents.
        """
        if not chunk_docs:
            raise ValueError("No chunk documents supplied to build_index().")
        self.vs = FAISS.from_documents(chunk_docs, self.embeddings)

        # Keep a tiny fingerprint to detect mismatches later (optional)
        fp_src = {d.metadata.get("source_id") for d in chunk_docs if d.metadata.get("source_id")}
        self._book_fingerprint = next(iter(fp_src)) if fp_src else None

    # 3) Retrieve top-k chunks for a query
    def retrieve(
        self,
        query: str,
        k: int = 5,
        with_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with: text, page_number, score (lower is better if using distances),
        and metadata (including chunk_id and source_id).
        """
        if self.vs is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        # similarity_search_with_score returns tuples: (Document, score)
        # With normalized embeddings, score approximates cosine distance.
        pairs: List[Tuple[Document, float]] = self.vs.similarity_search_with_score(query, k=k)

        results: List[Dict[str, Any]] = []
        for doc, score in pairs:
            results.append({
                "text": doc.page_content,
                "page_number": doc.metadata.get("page_number"),
                "score": float(score) if with_scores else None,
                "metadata": doc.metadata,
                "citation": f"(p. {doc.metadata.get('page_number')})",
            })
        return results

    # Optional: persist & load (handy in Colab runs)
    def save(self, folder_path: str) -> None:
        """
        Save FAISS index + docstore to disk.
        """
        if self.vs is None:
            raise RuntimeError("Index not built yet. Nothing to save.")
        self.vs.save_local(folder_path)

    def load(self, folder_path: str) -> None:
        """
        Load FAISS index + docstore from disk.
        """
        self.vs = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)

