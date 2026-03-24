from copy import deepcopy
from typing import List

import chromadb
import networkx as nx
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from sliders.chunkers.chunker import Chunker
from sliders.document import Document
from sliders.llm import get_llm_client, load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler


class ContextualChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 4000,
        overlap_size: int = 200,
        context_ratio: float = 0.2,
        summarize_context: bool = True,
        splitter_class=None,
        splitter_kwargs=None,
        format_chunks: bool = True,
    ):
        """Initialize the ContextualChunker.

        Args:
            chunk_size: Number of characters per chunk
            overlap_size: Number of characters to overlap between chunks
            context_ratio: How much of surrounding chunks to include (0.0-1.0)
            summarize_context: Whether to summarize context chunks if too large
            splitter_class: Class to use for splitting text
            splitter_kwargs: Keyword arguments to pass to the splitter
            format_chunks: Whether to format chunks with headers
        """
        super().__init__(chunk_size, overlap_size, splitter_class, splitter_kwargs, format_chunks)
        self.context_ratio = min(max(context_ratio, 0.0), 1.0)
        self.summarize_context = summarize_context

        summarize_text_llm_client = get_llm_client(model="gpt-4.1-mini", temperature=0.0)
        summarize_text_template = load_fewshot_prompt_template(
            template_file="summarize_text.prompt",
            template_blocks=[
                (
                    "instruction",
                    "Summarize the given text chunks. The summary should be concise and no more than 1 sentence.",
                ),
                ("input", "{{text}}"),
            ],
        )
        self.summarize_text_chain = summarize_text_template | summarize_text_llm_client

    def _get_context(self, chunks: List[str], chunk_idx: int) -> tuple[str, str]:
        """Get previous and next context for a chunk."""
        prev_context = ""
        next_context = ""

        if chunk_idx > 0:
            prev_chunk = chunks[chunk_idx - 1]
            if self.summarize_context and len(prev_chunk) > self.chunk_size * self.context_ratio:
                prev_context = prev_chunk[: int(len(prev_chunk) * self.context_ratio)]
            else:
                prev_context = prev_chunk

        if chunk_idx < len(chunks) - 1:
            next_chunk = chunks[chunk_idx + 1]
            if self.summarize_context and len(next_chunk) > self.chunk_size * self.context_ratio:
                next_context = next_chunk[: int(len(next_chunk) * self.context_ratio)]
            else:
                next_context = next_chunk

        return prev_context, next_context

    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks with context.

        Args:
            document: Document to split

        Returns:
            List of Document objects containing chunks with context
        """
        # Get base chunks
        chunks = self.splitter.split_text(document.content)

        # Extract content from chunks (handle both string and LangchainDocument types)
        chunk_contents = []
        for chunk in chunks:
            if hasattr(chunk, "page_content"):
                chunk_contents.append(chunk.page_content)
            else:
                chunk_contents.append(str(chunk))

        # Create new documents with context
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            prev_context, next_context = self._get_context(chunk_contents, i)

            doc = deepcopy(document)
            doc.content = chunk.page_content if hasattr(chunk, "page_content") else chunk
            doc.metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "previous_context": prev_context,
                    "next_context": next_context,
                }
            )
            chunked_docs.append(doc)

        return chunked_docs


class GraphChunker(ContextualChunker):
    def __init__(
        self,
        chunk_size: int = 2000,
        overlap_size: int = 200,
        splitter_class=None,
        splitter_kwargs=None,
        context_ratio: float = 0.2,
        summarize_context: bool = True,
        similarity_threshold: float = 0.8,
        max_semantic_edges: int = 3,
        embedding_model: str = "dunzhang/stella_en_1.5B_v5",
        chroma_path: str = "./chroma",
        collection_name: str = "hypothesis",
    ):
        """Initialize the GraphChunker.

        Args:
            chunk_size: Number of characters per chunk
            overlap_size: Number of characters to overlap between chunks
            splitter_class: Class to use for splitting text
            splitter_kwargs: Keyword arguments to pass to the splitter
            context_ratio: How much of surrounding chunks to include (0.0-1.0)
            summarize_context: Whether to summarize context chunks if too large
            similarity_threshold: Minimum similarity score for semantic edges
            max_semantic_edges: Maximum number of semantic edges per node
            embedding_model: Name of the sentence transformer model to use
        """
        super().__init__(
            chunk_size,
            overlap_size,
            context_ratio,
            summarize_context,
            splitter_class,
            splitter_kwargs,
        )
        self.similarity_threshold = similarity_threshold
        self.max_semantic_edges = max_semantic_edges
        self.graph = nx.Graph()

        # Initialize ChromaDB with sentence transformers
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _add_semantic_edges(self, chunks: List[str], document: Document) -> None:
        """Add edges between semantically similar chunks using vector similarity."""
        # Query all chunks at once instead of one by one
        results = self.collection.query(
            query_texts=chunks,
            n_results=self.max_semantic_edges + 1,  # +1 because it includes self
            where={"document_id": document.id},
        )

        # Process results in bulk
        for i, (chunk_ids, distances) in enumerate(zip(results["ids"], results["distances"])):
            for doc_id, distance in zip(chunk_ids, distances):
                j = int(doc_id.split("_")[0])  # works because we has chunk_index + "_" + document_id
                if i != j and distance >= self.similarity_threshold:
                    if self.graph.has_edge(i, j):
                        continue
                    self.graph.add_edge(
                        i,
                        j,
                        weight=distance,
                        edge_type="semantic",
                    )

    async def add_chunks_to_collection(self, chunks: List[str], document: Document) -> None:
        """Add chunks to collection and summarize them."""

        existing_ids = set(
            self.collection.get(
                where={"document_id": document.id},
            )["ids"]
        )
        chunks_to_add = [
            {
                "chunk": chunk,
                "document_id": document.id,
                "chunk_index": i,
                "chunk_id": str(i) + "_" + document.id,
            }
            for i, chunk in enumerate(chunks)
            if str(i) + "_" + document.id not in existing_ids
        ]

        if len(chunks_to_add):
            self.collection.add(
                documents=[chunk["chunk"] for chunk in chunks_to_add],
                ids=[chunk["chunk_id"] for chunk in chunks_to_add],
                metadatas=[
                    {
                        "document_id": document.id,
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": len(chunks),
                    }
                    for chunk in chunks_to_add
                ],
            )

    async def chunk_document(self, document: Document) -> List[dict]:
        """Split document into chunks and build a graph representation.

        Args:
            document: Document to split

        Returns:
            List of dictionaries containing chunks with context
        """
        # Reset graph for new document
        self.graph.clear()

        # Get base chunks using parent class method
        chunks = self.splitter.split_text(document.content)

        # Summarize chunks
        summarize_text_handler = LoggingHandler(
            prompt_file="summarize_text.prompt",
            metadata={"document": document.document_name},
        )
        summaries = await self.summarize_text_chain.abatch(
            [{"text": chunk} for chunk in chunks], config={"callbacks": [summarize_text_handler]}
        )

        # Build graph
        # Add nodes
        for i, chunk in enumerate(chunks):
            self.graph.add_node(i, content=chunk, summary=summaries[i])

        # Add sequential edges
        for i in range(len(chunks) - 1):
            self.graph.add_edge(
                i,
                i + 1,
                weight=1.0,
                edge_type="sequential",
            )

        # Add chunks to collection if not already present
        await self.add_chunks_to_collection(chunks, document)

        # Add semantic edges
        self._add_semantic_edges(chunks, document)

        # Create new documents with context from graph
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            # Get context from graph neighbors
            neighbors = sorted(
                [(n, self.graph[i][n]["weight"]) for n in self.graph.neighbors(i)],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            neighbors = [n[0] for n in neighbors]
            prev_context = ""
            next_context = ""

            for n in neighbors:
                if n < i:  # Previous context
                    n_content = self.graph.nodes[n]["content"]
                    if self.summarize_context:
                        prev_context += self.graph.nodes[n]["summary"] + "\n"
                    else:
                        prev_context += n_content + "\n"
                elif n > i:  # Next context
                    n_content = self.graph.nodes[n]["content"]
                    if self.summarize_context:
                        next_context += self.graph.nodes[n]["summary"] + "\n"
                    else:
                        next_context += n_content + "\n"

            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "previous_context": prev_context.strip(),
                "next_context": next_context.strip(),
                "graph_neighbors": neighbors,
            }
            chunked_docs.append({"content": chunk, "metadata": metadata})

        return chunked_docs


class SlidingWindowChunker(ContextualChunker):
    """Summarizes all the previous chunks and provides a context for the next chunk."""

    def __init__(self, chunk_size: int = 2000, overlap_size: int = 200, *args, **kwargs):
        super().__init__(chunk_size, overlap_size, *args, **kwargs)

    async def chunk_document(self, document: Document) -> List[dict]:
        chunks = self.splitter.split_text(document.content)

        # summarize chunk 1 for chunk 2, summarize chunk 1 and 2 for chunk 3, etc.
        forward_summaries = []

        for i in range(len(chunks)):
            summarize_text_handler = LoggingHandler(
                prompt_file="summarize_text.prompt",
                metadata={"document": document.document_name, "chunk_index": i, "type": "forward"},
            )
            if i == 0:
                forward_summaries.append(
                    await self.summarize_text_chain.ainvoke(
                        {"text": chunks[i]}, config={"callbacks": [summarize_text_handler]}
                    )
                )
            else:
                forward_summaries.append(
                    await self.summarize_text_chain.ainvoke(
                        "Summary till now: " + forward_summaries[i - 1] + "\n\n" + "Current chunk: " + chunks[i]
                    ),
                    config={"callbacks": [summarize_text_handler]},
                )

        # all the summary after the current chunk
        backward_summaries = []
        for i in range(len(chunks) - 1, -1, -1):
            summarize_text_handler = LoggingHandler(
                prompt_file="summarize_text.prompt",
                metadata={"document": document.document_name, "chunk_index": i, "type": "backward"},
            )
            if i == len(chunks) - 1:
                backward_summaries.insert(
                    0,
                    await self.summarize_text_chain.ainvoke(
                        {"text": chunks[i]}, config={"callbacks": [summarize_text_handler]}
                    ),
                )
            else:
                backward_summaries.insert(
                    0,
                    await self.summarize_text_chain.ainvoke(
                        {
                            "text": "Summary of following text: "
                            + backward_summaries[0]
                            + "\n\n"
                            + "Current chunk: "
                            + chunks[i]
                        },
                        config={"callbacks": [summarize_text_handler]},
                    ),
                )

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "previous_context": forward_summaries[i],
                "next_context": backward_summaries[i],
            }
            chunked_docs.append({"content": chunk, "metadata": metadata})
        return chunked_docs
