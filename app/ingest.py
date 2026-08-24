import json
import os
import re
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings import build_embeddings
from app.logging_setup import get_logger

logger = get_logger("ingest")

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
FILTER_EXERCISE_CHUNKS = os.getenv("RAG_FILTER_EXERCISE_CHUNKS", "1").strip() != "0"
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1").strip() != "0"


@dataclass
class IngestSummary:
    files_indexed: int
    pages_loaded: int
    chunks_total_after_filter: int
    chunks_added: int
    chunks_skipped_existing: int
    chunks_dropped: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    vectorstore_dir: str


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and not re.fullmatch(r"\d{1,4}", line)]
    text = " ".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _load_pdf_pages(pdf_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    cleaned_pages: List[Document] = []
    for page_doc in pages:
        source_name = pdf_path.name
        page_index = int(page_doc.metadata.get("page", 0))
        cleaned_pages.append(
            Document(
                page_content=clean_text(page_doc.page_content),
                metadata={
                    "source": source_name,
                    "page": page_index + 1,
                    "source_path": str(pdf_path),
                },
            )
        )
    return cleaned_pages


def load_documents(data_dir: str = DATA_DIR) -> List[Document]:
    data_path = Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    all_pages: List[Document] = []
    for pdf_path in sorted(data_path.glob("*.pdf")):
        pages = _load_pdf_pages(pdf_path)
        logger.info("Loaded PDF %s (%d pages)", pdf_path.name, len(pages))
        all_pages.extend(pages)
    return all_pages


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[List[Document], int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    raw_chunks = splitter.split_documents(documents)
    chunks: List[Document] = []
    dropped = 0

    per_page_counts: Dict[str, int] = defaultdict(int)
    for chunk in raw_chunks:
        if FILTER_EXERCISE_CHUNKS and is_noise_chunk(chunk.page_content):
            dropped += 1
            continue
        source = chunk.metadata.get("source", "unknown.pdf")
        page = chunk.metadata.get("page", 0)
        key = f"{source}|{page}"
        per_page_counts[key] += 1
        chunk.metadata["chunk_id"] = f"{source}:p{page}:c{per_page_counts[key]}"
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunks.append(chunk)
    return chunks, dropped


def is_noise_chunk(text: str) -> bool:
    lower = text.lower()
    option_markers = len(re.findall(r"\b[a-d]\.\s", lower))
    question_marks = text.count("?")
    words = text.split()
    if not words:
        return True
    alpha_ratio = sum(1 for char in text if char.isalpha()) / max(1, len(text))
    digit_ratio = sum(1 for char in text if char.isdigit()) / max(1, len(text))

    if "questions" in lower and option_markers >= 3:
        return True
    if question_marks >= 2 and option_markers >= 2:
        return True
    if option_markers >= 5:
        return True
    if alpha_ratio < 0.45 and digit_ratio > 0.18:
        return True
    return False


def _normalize_for_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"[^a-z0-9 .,:;!?()-]", "", normalized)
    return normalized


def chunk_fingerprint(chunk: Document) -> str:
    source = str(chunk.metadata.get("source", ""))
    page = str(chunk.metadata.get("page", ""))
    payload = f"{source}|{page}|{_normalize_for_hash(chunk.page_content)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_registry(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Could not read chunk registry at %s: %s: %s", path, type(exc).__name__, exc
        )
        return {}


def _save_registry(path: Path, registry: Dict[str, Dict[str, str]]) -> None:
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _bootstrap_registry_from_index(output_path: Path, embeddings) -> Dict[str, Dict[str, str]]:
    """Rebuild the registry from an existing index without unpickling by hand."""
    if not (output_path / "index.faiss").exists():
        return {}
    try:
        store = FAISS.load_local(str(output_path), embeddings, allow_dangerous_deserialization=True)
    except Exception as exc:
        logger.warning("Could not bootstrap registry from index: %s: %s", type(exc).__name__, exc)
        return {}
    registry = {}
    for doc in getattr(store.docstore, "_dict", {}).values():
        if isinstance(doc, Document):
            registry[chunk_fingerprint(doc)] = {
                "chunk_id": str(doc.metadata.get("chunk_id", "")),
                "source": str(doc.metadata.get("source", "")),
                "page": str(doc.metadata.get("page", "")),
            }
    return registry


def _split_new_chunks(
    chunks: List[Document], registry: Dict[str, Dict[str, str]]
) -> Tuple[List[Document], int]:
    new_chunks: List[Document] = []
    skipped_existing = 0
    for chunk in chunks:
        fingerprint = chunk_fingerprint(chunk)
        if fingerprint in registry:
            skipped_existing += 1
            continue
        chunk.metadata["chunk_hash"] = fingerprint
        new_chunks.append(chunk)
    return new_chunks, skipped_existing


def create_or_update_vectorstore(
    all_chunks: List[Document],
    output_dir: str = VECTORSTORE_DIR,
    rebuild: bool = False,
) -> Tuple[int, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    registry_path = output_path / "chunk_registry.json"

    embeddings = build_embeddings(get_settings())

    registry = {} if rebuild else _load_registry(registry_path)
    if not rebuild and not registry:
        registry = _bootstrap_registry_from_index(output_path, embeddings)
        if registry:
            _save_registry(registry_path, registry)
    new_chunks, skipped_existing = _split_new_chunks(all_chunks, registry)
    if not new_chunks and not rebuild:
        return 0, skipped_existing

    index_file = output_path / "index.faiss"
    if rebuild or not index_file.exists():
        store = FAISS.from_documents(new_chunks, embeddings)
    else:
        store = FAISS.load_local(
            str(output_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        store.add_documents(new_chunks)

    store.save_local(str(output_path))

    for chunk in new_chunks:
        chunk_hash = str(chunk.metadata["chunk_hash"])
        registry[chunk_hash] = {
            "chunk_id": str(chunk.metadata.get("chunk_id", "")),
            "source": str(chunk.metadata.get("source", "")),
            "page": str(chunk.metadata.get("page", "")),
        }
    _save_registry(registry_path, registry)
    return len(new_chunks), skipped_existing


def _write_manifest(summary: IngestSummary) -> None:
    manifest_path = Path(summary.vectorstore_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def ingest_pipeline(
    data_dir: str = DATA_DIR,
    vectorstore_dir: str = VECTORSTORE_DIR,
    rebuild: bool = False,
) -> IngestSummary:
    pages = load_documents(data_dir=data_dir)
    if not pages:
        raise ValueError(f"No PDF pages loaded from '{data_dir}'.")

    chunks, dropped_chunks = split_documents(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("All chunks were filtered out during ingestion. Disable RAG_FILTER_EXERCISE_CHUNKS.")
    chunks_added, chunks_skipped = create_or_update_vectorstore(
        chunks,
        output_dir=vectorstore_dir,
        rebuild=rebuild,
    )

    files_indexed = len({doc.metadata.get("source", "") for doc in pages})
    summary = IngestSummary(
        files_indexed=files_indexed,
        pages_loaded=len(pages),
        chunks_total_after_filter=len(chunks),
        chunks_added=chunks_added,
        chunks_skipped_existing=chunks_skipped,
        chunks_dropped=dropped_chunks,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
        vectorstore_dir=vectorstore_dir,
    )
    _write_manifest(summary)
    logger.info(
        "Ingest complete: files=%d pages=%d chunks_added=%d chunks_skipped_existing=%d chunks_dropped=%d",
        files_indexed,
        len(pages),
        chunks_added,
        chunks_skipped,
        dropped_chunks,
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest AIRMAN PDFs into FAISS.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch.")
    args = parser.parse_args()

    result = ingest_pipeline(rebuild=args.rebuild)
    print(json.dumps(asdict(result), indent=2))
