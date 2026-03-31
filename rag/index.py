"""Index extracted JSONL chunks into ChromaDB with GTE-Large embeddings on MPS."""

import argparse
import json
import sys
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "glowfic"
MODEL_NAME = "thenlper/gte-large"
BATCH_SIZE = 64
MIN_TEXT_LENGTH = 50


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


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def index_posts(posts: list[dict], collection: chromadb.Collection, batch_size: int = BATCH_SIZE):
    """Upsert posts into ChromaDB in batches."""
    skipped = sum(1 for p in posts if len(p["text"].strip()) < MIN_TEXT_LENGTH)
    posts = [p for p in posts if len(p["text"].strip()) >= MIN_TEXT_LENGTH]
    if skipped:
        print(f"  Skipped {skipped} posts shorter than {MIN_TEXT_LENGTH} chars")


    for i in tqdm(range(0, len(posts), batch_size), desc="Indexing"):
        batch = posts[i : i + batch_size]
        ids = [f"{p.get('continuity','')}/{p.get('thread','')}/{p['id']}" for p in batch]
        documents = [p["text"] for p in batch]
        metadatas = [
            {
                "thread": p.get("thread") or "",
                "section": p.get("section") or "",
                "continuity": p.get("continuity") or "",
                "author": p.get("author") or "",
                "character": p.get("character") or "",
                "screen_name": p.get("screen_name") or "",
                "permalink": p.get("permalink") or "",
            }
            for p in batch
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def main():
    parser = argparse.ArgumentParser(description="Index extracted JSONL into ChromaDB.")
    parser.add_argument(
        "jsonl_files",
        nargs="*",
        help="JSONL files to index (default: all *.jsonl in data/)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before indexing")
    args = parser.parse_args()

    if args.jsonl_files:
        jsonl_paths = [Path(f) for f in args.jsonl_files]
    else:
        jsonl_paths = sorted(DATA_DIR.glob("*.jsonl"))

    if not jsonl_paths:
        print("No JSONL files found. Run rag.extract first.")
        sys.exit(1)

    device = get_device()
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = GTEEmbeddingFunction(model)

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    for path in jsonl_paths:
        print(f"Loading {path.name}...")
        posts = load_jsonl(path)
        print(f"  {len(posts)} posts loaded, indexing...")
        index_posts(posts, collection, batch_size=args.batch_size)

    print(f"\nDone. Collection has {collection.count()} documents.")
    print(f"Stored at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
