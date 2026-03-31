"""Scrape glowfic threads directly to JSONL, bypassing epub generation.

Useful for large boards where the epub pipeline runs out of memory.
Processes one thread at a time and supports resumption.
"""

import asyncio
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import aiolimiter
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.render import get_book_structure, Continuity, Section, Thread
from src.auth import auth_get

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_posts_from_soup(soup: BeautifulSoup) -> list[dict]:
    """Extract post data directly from a thread's HTML page."""
    posts = []
    for container in soup.find_all("div", "post-container"):
        try:
            character = container.find("div", "post-character").text.strip()
        except AttributeError:
            character = None
        try:
            screen_name = container.find("div", "post-screenname").text.strip()
        except AttributeError:
            screen_name = None
        try:
            author = container.find("div", "post-author").text.strip()
        except AttributeError:
            author = None

        content_div = container.find("div", "post-content")
        text = content_div.get_text(separator="\n", strip=True) if content_div else ""

        permalink_img = container.find("img", title="Permalink", alt="Permalink")
        permalink = None
        reply_id = None
        if permalink_img and permalink_img.parent.get("href"):
            permalink = permalink_img.parent["href"]
            if not permalink.startswith("http"):
                permalink = f"https://glowfic.com{permalink}"
            fragment = urlparse(permalink).fragment
            if fragment:
                reply_id = fragment

        posts.append({
            "id": reply_id,
            "text": text,
            "permalink": permalink,
            "author": author,
            "character": character,
            "screen_name": screen_name,
        })
    return posts


async def scrape_thread(
    session: aiohttp.ClientSession,
    limiter: aiolimiter.AsyncLimiter,
    thread: Thread,
    continuity: str,
    section: str | None,
    max_retries: int = 4,
) -> list[dict]:
    """Download and extract all posts from a single thread.

    Retries with exponential backoff when the response looks throttled
    (no post-container divs found).
    """
    for attempt in range(max_retries):
        await limiter.acquire()
        resp = await auth_get(session, thread.url, params={"view": "flat"})
        html = await resp.text()
        resp.close()

        if "throttled" in html.lower() and len(html) < 1000:
            wait = 2 ** (attempt + 2)
            tqdm.write(f"  '{thread.title}': throttled (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
            await asyncio.sleep(wait)
            continue

        soup = BeautifulSoup(html, "html.parser")
        posts = extract_posts_from_soup(soup)

        if len(posts) == 0 and attempt < max_retries - 1:
            wait = 2 ** (attempt + 2)
            tqdm.write(f"  '{thread.title}': 0 posts (attempt {attempt+1}/{max_retries}), retrying in {wait}s...")
            await asyncio.sleep(wait)
            continue

        if len(posts) == 0:
            tqdm.write(f"  WARNING: '{thread.title}' returned 0 posts after {max_retries} attempts")

        for i, post in enumerate(posts):
            post["thread"] = thread.title
            post["section"] = section
            post["continuity"] = continuity
            if post["id"] is None:
                post["id"] = f"post-{i}"
        return posts

    return []


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape glowfic board/section/thread directly to JSONL."
    )
    parser.add_argument("url", help="glowfic thread, section, or board URL")
    parser.add_argument("-o", "--output", help="Output JSONL path (default: data/<title>.jsonl)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip threads already present in the output file")
    args = parser.parse_args()

    limiter = aiolimiter.AsyncLimiter(1, 1)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit_per_host=1)
    ) as session:
        print("Fetching board structure...")
        structure = await get_book_structure(session, limiter, args.url)

        out_path = Path(args.output) if args.output else DATA_DIR / f"{structure.title}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build (thread, section_name) pairs
        thread_sections: list[tuple[Thread, str | None]] = []
        if isinstance(structure, Continuity):
            for sec in structure.sections:
                for t in sec.threads:
                    thread_sections.append((t, sec.title))
            if structure.sectionless_threads:
                for t in structure.sectionless_threads.threads:
                    thread_sections.append((t, None))
        elif isinstance(structure, Section):
            for t in structure.threads:
                thread_sections.append((t, structure.title))
        else:
            thread_sections.append((structure, None))

        # Resume support: find threads already scraped
        done_threads: set[str] = set()
        if args.resume and out_path.exists():
            with open(out_path) as f:
                for line in f:
                    obj = json.loads(line)
                    done_threads.add(obj.get("thread", ""))
            print(f"Resuming: {len(done_threads)} threads already in {out_path}")

        remaining = [(t, s) for t, s in thread_sections if t.title not in done_threads]
        print(f"Threads to scrape: {len(remaining)} (of {len(thread_sections)} total)")

        mode = "a" if args.resume and out_path.exists() else "w"
        total_posts = 0
        with open(out_path, mode) as f:
            for thread, section in tqdm(remaining, desc="Scraping"):
                try:
                    posts = await scrape_thread(
                        session, limiter, thread, structure.title, section
                    )
                    for post in posts:
                        f.write(json.dumps(post, ensure_ascii=False) + "\n")
                    f.flush()
                    total_posts += len(posts)
                except Exception as e:
                    print(f"\n  Error scraping '{thread.title}': {e}")
                    continue

        print(f"Done: {total_posts} posts from {len(remaining)} threads -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
