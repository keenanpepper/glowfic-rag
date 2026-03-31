# glowfic-rag

Semantic search over [glowfic.com](https://glowfic.com) content, exposed as an MCP server for Claude Code and Cursor.

Uses [GTE-Large](https://huggingface.co/thenlper/gte-large) embeddings stored in ChromaDB. The pre-built index covers ~658K posts from:

- **planecrash** (board 215) — 19K posts, 28 threads
- **Sandboxes** (board 3) — 639K posts, 2,214 threads
- **trainwreck** (board 277) — 595 posts, 7 threads

Board structure fetching (`src/render.py`, `src/auth.py`) is derived from [rocurley/glowfic-dl](https://github.com/rocurley/glowfic-dl) by [Roger Curley](https://github.com/rocurley) and [Alyssa Riceman](https://github.com/LunarTulip).

## Quick start

Requires **Python 3.10+** (the `mcp` package needs it).

```bash
git clone https://github.com/keenanpepper/glowfic-rag.git
cd glowfic-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the pre-built vector database (~2GB compressed, ~5GB on disk)
./setup_data.sh
```

## Using with Claude Code

The repo includes `.mcp.json`, which Claude Code auto-detects. From the repo directory:

```bash
claude
```

Claude will have access to `search_glowfic` and `list_indexed_continuities` tools.

## Using with Cursor

Create `.cursor/mcp.json` with the absolute path to your clone:

```json
{
  "mcpServers": {
    "glowfic-search": {
      "command": "/path/to/glowfic-rag/.venv/bin/python3",
      "args": ["-m", "rag.mcp_server"],
      "cwd": "/path/to/glowfic-rag"
    }
  }
}
```

Then reload the Cursor window.

## CLI search

You can also search from the command line:

```bash
python -m rag.search "Keltham explains expected utility" -k 5
python -m rag.search "vampire meets werewolf" --continuity Sandboxes
python -m rag.search "dath ilan" --author Iarwain
```

## Adding more continuities

Scrape a board to JSONL, then add it to the index:

```bash
# Scrape (supports --resume for interrupted runs)
python -m rag.scrape https://glowfic.com/boards/277

# Index (adds to existing collection, doesn't rebuild)
python -m rag.index data/trainwreck.jsonl

# Or reindex everything from scratch
python -m rag.index --reset
```

The scraper handles pagination for large boards and retries throttled requests automatically.

## Performance notes

- **First query**: ~10 seconds (model load + index warm-up)
- **Subsequent queries**: < 1 second
- **Memory**: ~3-5GB when model + index are loaded
- **Disk**: ~5GB for the ChromaDB index
