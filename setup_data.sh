#!/usr/bin/env bash
set -euo pipefail

REPO="keenanpepper/glowfic-rag"
TAG="v1.0-data"
DATA_DIR="data"
CHROMA_DIR="$DATA_DIR/chroma_db"

if [ -d "$CHROMA_DIR" ]; then
    echo "data/chroma_db/ already exists. Delete it first to re-download."
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "Downloading pre-built vector database from GitHub Release..."
echo "  repo: $REPO  tag: $TAG"

if command -v gh &>/dev/null; then
    gh release download "$TAG" --repo "$REPO" --dir "$DATA_DIR" --pattern "*.tar.gz.part-*"
else
    echo "gh CLI not found, falling back to curl..."
    URLS=$(curl -sL "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
        | python3 -c "import sys,json; [print(a['browser_download_url']) for a in json.load(sys.stdin)['assets'] if 'tar.gz' in a['name']]")
    for url in $URLS; do
        echo "  Downloading $(basename "$url")..."
        curl -L -o "$DATA_DIR/$(basename "$url")" "$url"
    done
fi

echo "Reassembling and extracting..."
cat "$DATA_DIR"/glowfic-chroma.tar.gz.part-* > "$DATA_DIR/glowfic-chroma.tar.gz"
rm "$DATA_DIR"/glowfic-chroma.tar.gz.part-*
tar xzf "$DATA_DIR/glowfic-chroma.tar.gz" -C "$DATA_DIR"
rm "$DATA_DIR/glowfic-chroma.tar.gz"

echo ""
echo "Verifying..."
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='$CHROMA_DIR')
collection = client.get_collection(name='glowfic')
count = collection.count()
print(f'  ChromaDB collection has {count} documents.')
if count > 0:
    print('  Setup complete!')
else:
    print('  WARNING: Collection is empty.')
"
