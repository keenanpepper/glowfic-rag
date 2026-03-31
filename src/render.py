"""Board structure fetching for glowfic.com.

Stripped-down version of rocurley/glowfic-dl's render.py, containing only
the board/section/thread listing logic needed by the scraper. Epub rendering,
image handling, and post compilation have been removed.
"""

import asyncio
from itertools import chain
import re
from typing import Iterable, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
import aiolimiter
from bs4 import BeautifulSoup
from bs4.element import Tag, ResultSet

from .auth import auth_get
from .constants import GLOWFIC_ROOT


class Thread:
    def __init__(self, title: str, url: str, description: Optional[str] = None):
        self.title = title
        self.url = url
        self.description = description
        self.threads = [self]


class Section:
    def __init__(
        self,
        title: Optional[str],
        threads: list[Thread],
        description: Optional[str] = None,
    ):
        self.title = title
        self.threads = threads
        self.description = description


class Continuity:
    def __init__(
        self,
        title: str,
        sections: list[Section],
        sectionless_threads: Optional[Section] = None,
    ):
        self.title = title
        self.sections = sections
        self.sectionless_threads = sectionless_threads

        self.threads = list(chain(*[section.threads for section in self.sections]))
        if sectionless_threads is not None:
            self.threads += sectionless_threads.threads


def validate_tag(tag: Tag, soup: BeautifulSoup) -> Tag:
    if tag is not None:
        return tag
    err = soup.find("div", "flash error")
    if err is not None:
        raise RuntimeError(err.text.strip())
    else:
        raise RuntimeError("Unknown error: tag missing")


def thread_from_board_row(row: Tag) -> Thread:
    thread_link = row.find("a")
    title = thread_link.text.strip()
    description = thread_link.get("title")
    url = urljoin(GLOWFIC_ROOT, thread_link["href"])
    return Thread(title, url, description)


def sections_from_board_rows(rows: ResultSet) -> Iterable[Section]:
    current_title = None
    current_threads = []
    current_description = None

    for row in rows:
        if (title := row.find("th", "continuity-header")) is not None:
            current_title = next(title.children).text.strip()
        elif (description := row.find("td", "written-content")) is not None:
            current_description = description.text.strip()
        elif (thread := row.find("td", "post-subject")) is not None:
            current_threads.append(thread_from_board_row(thread))
        elif row.find("td", "continuity-spacer") is not None:
            if len(current_threads) == 0:
                current_title = None
                current_description = None
                continue
            elif current_title is not None:
                yield Section(current_title, current_threads, current_description)
                current_title = None
                current_threads = []
                current_description = None
            else:
                raise Exception(
                    "Encountered nonfinal titleless section. (This should be impossible.)"
                )

    if len(current_threads) > 0:
        yield Section(current_title, current_threads, current_description)


async def get_book_structure(
    session: aiohttp.ClientSession, limiter: aiolimiter.AsyncLimiter, url: str
) -> Thread | Section | Continuity:
    target_url = (
        "https://glowfic.com/api/v1%s" % urlparse(url).path if "posts" in url else url
    )
    await limiter.acquire()
    resp = await auth_get(session, target_url)

    if "posts" in url:
        post_json = await resp.json()
        return Thread(post_json["subject"], url, post_json.get("description"))
    elif "board_sections" in url:
        soup = BeautifulSoup(await resp.text(), "html.parser")
        title = soup.find("th", "table-title").text.strip()
        description = soup.find("td", "written-content")
        if description is not None:
            description = description.text.strip()
        rows = validate_tag(soup.find("div", id="content"), soup).find_all(
            "td", "post-subject"
        )
        threads = [thread_from_board_row(row) for row in rows]
        return Section(title, threads, description)
    elif "boards" in url:
        soup = BeautifulSoup(await resp.text(), "html.parser")
        title = next(soup.find("th", "table-title").children).strip()
        all_rows = validate_tag(soup.find("div", id="content"), soup).find_all("tr")

        last_link = soup.find("a", class_="last_page")
        if last_link:
            last_page = int(parse_qs(urlparse(last_link["href"]).query)["page"][0])
            print(f"  Board has {last_page} pages, fetching all...")
            for page_num in range(2, last_page + 1):
                page_url = f"{url}?page={page_num}" if "?" not in url else re.sub(r'page=\d+', f'page={page_num}', url)
                page_rows = None
                for attempt in range(3):
                    await limiter.acquire()
                    page_resp = await auth_get(session, page_url)
                    page_soup = BeautifulSoup(await page_resp.text(), "html.parser")
                    page_resp.close()
                    content_div = page_soup.find("div", id="content")
                    if content_div is not None:
                        page_rows = content_div.find_all("tr")
                        break
                    wait = 2 ** (attempt + 1)
                    print(f"  Page {page_num}: no content (attempt {attempt+1}/3), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                if page_rows is None:
                    print(f"  Warning: page {page_num} failed after 3 attempts, skipping")
                else:
                    all_rows.extend(page_rows)
                if page_num % 10 == 0:
                    print(f"  Page {page_num}/{last_page}...")

        sections = list(sections_from_board_rows(all_rows))
        if sections[-1].title is None:
            return Continuity(title, sections[:-1], sections[-1])
        else:
            return Continuity(title, sections)
    else:
        raise ValueError(
            "URL contains neither 'posts' nor 'board_sections' nor 'boards'."
        )
