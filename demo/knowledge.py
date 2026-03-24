"""
Talk2EP Demo — Vector-retrieval knowledge layer for EnergyPlus domain knowledge.

Uses SLIDERS' markdown chunker for text splitting + ChromaDB for semantic
retrieval.  At query time, retrieves the top-k most relevant chunks from the
EnergyPlus documentation and returns them as context for the orchestrator LLM.

One-time indexing:
    python knowledge.py --index

This replaces the full SLIDERS extraction/SQL pipeline with a lightweight
chunk → embed → retrieve approach.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Ensure demo/ and sliders/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EP_DOCS_PATH, SLIDERS_DIR  # noqa: E402

_sliders_path = str(SLIDERS_DIR)
if _sliders_path not in sys.path:
    sys.path.insert(0, _sliders_path)

# ── Constants ────────────────────────────────────────────────────────────
CHROMA_DIR = str(Path(__file__).resolve().parent / ".chroma")
COLLECTION_NAME = "energyplus_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 8


# ── Indexing ─────────────────────────────────────────────────────────────

def _get_collection(read_only: bool = False):
    """Return the ChromaDB collection (creates DB dir if needed)."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


async def index_documents(docs_path: str | None = None) -> int:
    """Chunk all markdown files in *docs_path* and upsert into ChromaDB.

    Returns the total number of chunks indexed.
    """
    docs_path = docs_path or EP_DOCS_PATH
    if not docs_path:
        raise ValueError("EP_DOCS_PATH is not set.")

    p = Path(docs_path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.md")) + sorted(p.glob("*.txt"))
    else:
        raise FileNotFoundError(f"EP_DOCS_PATH not found: {docs_path}")

    if not files:
        raise FileNotFoundError(f"No .md / .txt files in {docs_path}")

    # Use SLIDERS' chunker for markdown-aware splitting
    from sliders.chunkers.chunker import Chunker
    chunker = Chunker(chunk_size=CHUNK_SIZE, overlap_size=CHUNK_OVERLAP)

    collection = _get_collection()

    total = 0
    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        chunks = chunker.chunk_text(text)

        doc_name = file.stem
        ids = [f"{doc_name}__chunk_{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "source": file.name,
                "doc_name": doc_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Upsert in batches of 500 (ChromaDB limit)
        for start in range(0, len(ids), 500):
            end = start + 500
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        print(f"  indexed {file.name}: {len(chunks)} chunks")
        total += len(chunks)

    print(f"Total: {total} chunks in ChromaDB ({CHROMA_DIR})")
    return total


# ── Retrieval ────────────────────────────────────────────────────────────

class EnergyPlusKnowledge:
    """Lightweight vector-retrieval wrapper over the EnergyPlus docs."""

    def __init__(self, docs_path: Optional[str] = None, top_k: int = TOP_K):
        self._docs_path = docs_path or EP_DOCS_PATH
        self._top_k = top_k
        self._collection = None
        self._initialized = False

    @property
    def available(self) -> bool:
        """True if the ChromaDB index exists and has documents."""
        if self._collection is None:
            try:
                self._collection = _get_collection(read_only=True)
            except Exception:
                return False
        return self._collection.count() > 0

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._collection = _get_collection(read_only=True)
        if self._collection.count() == 0:
            print(
                "[knowledge] ChromaDB collection is empty. "
                "Run `python knowledge.py --index` first."
            )
        self._initialized = True

    async def query(self, question: str) -> str:
        """Retrieve the most relevant chunks for *question*."""
        if not self._initialized:
            await self.initialize()

        if self._collection is None or self._collection.count() == 0:
            return (
                "[Knowledge base unavailable] "
                "Run `python knowledge.py --index` with EP_DOCS_PATH set "
                "to index EnergyPlus documentation into ChromaDB."
            )

        results = self._collection.query(
            query_texts=[question],
            n_results=self._top_k,
        )

        # Format retrieved chunks into a single context string
        parts: list[str] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            source = meta.get("source", "unknown")
            idx = meta.get("chunk_index", "?")
            similarity = 1 - dist  # cosine distance → similarity
            parts.append(
                f"--- [{source} chunk {idx}] (similarity: {similarity:.3f}) ---\n{doc}"
            )

        return "\n\n".join(parts)


# ── CLI entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EnergyPlus Knowledge Base")
    parser.add_argument("--index", action="store_true", help="Index documents into ChromaDB")
    parser.add_argument("--query", "-q", type=str, help="Test a retrieval query")
    parser.add_argument("--docs", type=str, default=None, help="Override EP_DOCS_PATH")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.index:
        asyncio.run(index_documents(args.docs))
    elif args.query:
        kb = EnergyPlusKnowledge(docs_path=args.docs, top_k=args.top_k)
        result = asyncio.run(kb.query(args.query))
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
