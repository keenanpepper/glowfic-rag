"""Minimal search CLI for testing retrieval quality against the glowfic index."""

import argparse
import textwrap
from pathlib import Path

import chromadb
import torch
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


# Reuse from index.py
class GTEEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


def format_result(doc: str, meta: dict, distance: float, index: int) -> str:
    snippet = textwrap.shorten(doc, width=300, placeholder="...")
    parts = [f"[{index+1}] (distance: {distance:.4f})"]
    label_parts = []
    if meta.get("character"):
        label_parts.append(meta["character"])
    if meta.get("author"):
        label_parts.append(meta["author"])
    if label_parts:
        parts.append(" / ".join(label_parts))
    parts.append(f"Thread: {meta.get('thread', '?')}")
    if meta.get("section"):
        parts.append(f"Section: {meta['section']}")
    if meta.get("permalink"):
        parts.append(meta["permalink"])
    parts.append(f"  {snippet}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Search the glowfic index.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--continuity", help="Filter by continuity name")
    parser.add_argument("--author", help="Filter by author")
    parser.add_argument("--thread", help="Filter by thread title (substring match via metadata)")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading model on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = GTEEmbeddingFunction(model)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    where_filter = {}
    if args.continuity:
        where_filter["continuity"] = args.continuity
    if args.author:
        where_filter["author"] = args.author

    query_kwargs = {
        "query_texts": [args.query],
        "n_results": args.top_k,
    }
    if where_filter:
        if len(where_filter) == 1:
            key, val = next(iter(where_filter.items()))
            query_kwargs["where"] = {key: val}
        else:
            query_kwargs["where"] = {"$and": [{k: v} for k, v in where_filter.items()]}

    results = collection.query(**query_kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Results: {len(docs)}")
    print(f"Collection size: {collection.count()}")
    print(f"{'='*60}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        print(format_result(doc, meta, dist, i))
        print()


if __name__ == "__main__":
    main()
