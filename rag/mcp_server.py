"""MCP server exposing glowfic semantic search as a tool for Claude."""

import sys
from pathlib import Path
from typing import Optional

import chromadb
import torch
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "glowfic"
MODEL_NAME = "thenlper/gte-large"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class GTEEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


print("Loading embedding model...", file=sys.stderr)
device = get_device()
print(f"Using device: {device}", file=sys.stderr)
_model = SentenceTransformer(MODEL_NAME, device=device)
_embed_fn = GTEEmbeddingFunction(_model)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=_embed_fn)
print(f"Ready. Collection has {_collection.count()} documents.", file=sys.stderr)

mcp = FastMCP("glowfic-search")


@mcp.tool()
def search_glowfic(
    query: str,
    n_results: int = 10,
    continuity: Optional[str] = None,
    author: Optional[str] = None,
) -> str:
    """Search the glowfic corpus for passages semantically similar to the query.

    Returns the most relevant posts/replies from indexed glowfic continuities
    (e.g. planecrash, Sandboxes). Each result includes the text, author,
    character, thread title, and a permalink to the original.

    Args:
        query: Natural language search query describing what you're looking for.
        n_results: Number of results to return (default 10, max 30).
        continuity: Optional filter to search only a specific continuity (e.g. "planecrash").
        author: Optional filter to search only posts by a specific author username.
    """
    n_results = min(n_results, 30)

    where_clauses = []
    if continuity:
        where_clauses.append({"continuity": continuity})
    if author:
        where_clauses.append({"author": author})

    query_kwargs: dict = {
        "query_texts": [query],
        "n_results": n_results,
    }
    if len(where_clauses) == 1:
        query_kwargs["where"] = where_clauses[0]
    elif len(where_clauses) > 1:
        query_kwargs["where"] = {"$and": where_clauses}

    results = _collection.query(**query_kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    output_parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        header_parts = []
        if meta.get("character"):
            header_parts.append(meta["character"])
        if meta.get("author"):
            header_parts.append(meta["author"])
        header = " / ".join(header_parts) if header_parts else "Unknown"

        entry = (
            f"[Result {i+1}] (relevance: {1 - dist:.3f})\n"
            f"Author: {header}\n"
            f"Thread: {meta.get('thread', '?')}\n"
        )
        if meta.get("section"):
            entry += f"Section: {meta['section']}\n"
        if meta.get("permalink"):
            entry += f"Permalink: {meta['permalink']}\n"
        entry += f"\n{doc}\n"
        output_parts.append(entry)

    return "\n---\n".join(output_parts)


@mcp.tool()
def list_indexed_continuities() -> str:
    """List all glowfic continuities that have been indexed and are available for search.

    Returns continuity names, thread counts, and total post counts.
    """
    from collections import Counter

    total = _collection.count()
    continuities: Counter = Counter()
    threads_per_continuity: dict[str, set] = {}

    batch_size = 10000
    for offset in range(0, total, batch_size):
        batch = _collection.get(
            include=["metadatas"], limit=batch_size, offset=offset
        )["metadatas"]
        for meta in batch:
            cont = meta.get("continuity", "unknown")
            continuities[cont] += 1
            if cont not in threads_per_continuity:
                threads_per_continuity[cont] = set()
            threads_per_continuity[cont].add(meta.get("thread", ""))

    lines = [f"Indexed glowfic corpus: {total} total posts\n"]
    for cont, count in continuities.most_common():
        n_threads = len(threads_per_continuity[cont])
        lines.append(f"- {cont}: {count} posts across {n_threads} threads")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
