#!/usr/bin/env python3
"""Crawl eecs.berkeley.edu pages and save cleaned text for RAG corpus.
Uses concurrent requests for speed."""

import re
import json
import time
import os
import hashlib
from urllib.parse import urljoin, urlparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

SEED_URLS = [
    "https://eecs.berkeley.edu/",
    "https://www2.eecs.berkeley.edu/",
    "https://eecs.berkeley.edu/academics/",
    "https://eecs.berkeley.edu/people/",
    "https://eecs.berkeley.edu/research/",
    "https://eecs.berkeley.edu/news/",
    "https://eecs.berkeley.edu/resources/",
    "https://eecs.berkeley.edu/about/",
    "https://www2.eecs.berkeley.edu/Courses/",
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://www2.eecs.berkeley.edu/Scheduling/",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule-draft.html",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/",
    "https://eecs.berkeley.edu/people/faculty/",
    "https://eecs.berkeley.edu/people/students-2/",
    "https://eecs.berkeley.edu/people/students-2/awards/",
    "https://eecs.berkeley.edu/people/faculty/in-memoriam/",
    "https://eecs.berkeley.edu/book/",
    "https://eecs.berkeley.edu/book/phd/",
    "https://eecs.berkeley.edu/book/phd/coursework/",
    "https://eecs.berkeley.edu/book/faculty/",
    "https://eecs.berkeley.edu/resources/undergrads/",
    "https://eecs.berkeley.edu/resources/undergrads/cs/",
    "https://eecs.berkeley.edu/resources/undergrads/cs/advising/",
    "https://eecs.berkeley.edu/resources/undergrads/eecs/",
    "https://eecs.berkeley.edu/resources/grads/",
]

URL_PATTERN = re.compile(
    r"https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?"
)

SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".css", ".js", ".zip", ".tar", ".gz", ".mp4", ".mp3",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".bib", ".ps", ".eps", ".dvi", ".xml", ".rss",
}

OUTPUT_DIR = "corpus"
MAX_PAGES = 5000
WORKERS = 10


def is_valid_url(url):
    if not URL_PATTERN.match(url):
        return False
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False
    if "login" in path_lower or "cas" in path_lower or "wp-json" in path_lower:
        return False
    # Skip very deep paths (likely not useful)
    if path_lower.count("/") > 8:
        return False
    return True


def normalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    # Also normalize query for dedup
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def clean_html(html, url):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    main = (soup.find("main") or soup.find("article") or
            soup.find("div", {"id": "content"}) or
            soup.find("div", {"class": "content"}) or soup.body)
    if main is None:
        main = soup

    text = main.get_text(separator="\n", strip=True)

    # Extract tables structurally
    tables_text = []
    for table in (main.find_all("table") if main else []):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            tables_text.append("\n".join(rows))

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    if tables_text:
        text += "\n\nTABLES:\n" + "\n\n".join(tables_text)

    return title, text


def fetch_url(session, url):
    try:
        resp = session.get(url, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None
        return resp
    except Exception:
        return None


def crawl():
    visited = set()
    queue = deque()
    pages = []

    for url in SEED_URLS:
        norm = normalize_url(url)
        if norm not in visited:
            visited.add(norm)
            queue.append(url)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    count = 0
    while queue and count < MAX_PAGES:
        # Grab a batch of URLs
        batch = []
        while queue and len(batch) < WORKERS:
            batch.append(queue.popleft())

        # Fetch in parallel
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_to_url = {
                executor.submit(fetch_url, session, url): url
                for url in batch
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                resp = future.result()
                if resp is None:
                    continue

                title, text = clean_html(resp.text, resp.url)
                if len(text) < 50:
                    continue

                pages.append({
                    "url": resp.url,
                    "title": title,
                    "text": text,
                })
                count += 1

                if count % 100 == 0:
                    print(f"Crawled {count} pages, queue: {len(queue)}...")

                # Extract links
                soup = BeautifulSoup(resp.text, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    full_url = urljoin(resp.url, href)
                    full_url = full_url.split("#")[0]
                    # Remove query params for cleaner dedup
                    if "?" in full_url:
                        # Keep some useful query params, skip tracking ones
                        pass
                    if is_valid_url(full_url):
                        norm = normalize_url(full_url)
                        if norm not in visited:
                            visited.add(norm)
                            queue.append(full_url)

        if count >= MAX_PAGES:
            break

    return pages


def chunk_text(text, max_chunk_size=500, overlap=75):
    """Split text into overlapping chunks."""
    words = text.split()
    if len(words) <= max_chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting crawl of eecs.berkeley.edu...")
    pages = crawl()
    print(f"Crawled {len(pages)} pages total.")

    with open(os.path.join(OUTPUT_DIR, "pages.jsonl"), "w") as f:
        for page in pages:
            f.write(json.dumps(page) + "\n")

    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{hashlib.md5(page['url'].encode()).hexdigest()}_{i}",
                "url": page["url"],
                "title": page["title"],
                "text": chunk,
            })

    with open(os.path.join(OUTPUT_DIR, "chunks.jsonl"), "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages.")


if __name__ == "__main__":
    main()
