#!/usr/bin/env python3
"""Crawl www2.eecs.berkeley.edu pages via web.archive.org and add to corpus."""

import re
import json
import os
import hashlib
import time
from urllib.parse import urljoin, urlparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

WAYBACK_PREFIX = "https://web.archive.org/web/2025/"

SEED_URLS = [
    "https://www2.eecs.berkeley.edu/",
    "https://www2.eecs.berkeley.edu/Courses/",
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Courses/EECS/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://www2.eecs.berkeley.edu/Scheduling/",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule-draft.html",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/",
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
MAX_PAGES = 3000
WORKERS = 5  # Be gentler with archive.org


def is_valid_url(url):
    if not URL_PATTERN.match(url):
        return False
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False
    if "login" in path_lower or "cas" in path_lower:
        return False
    if path_lower.count("/") > 8:
        return False
    # Only follow www2 links
    if "www2.eecs.berkeley.edu" not in parsed.netloc:
        return False
    return True


def normalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def clean_html(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Remove wayback toolbar
    for tag in soup.find_all(id="wm-ipp-base"):
        tag.decompose()
    for tag in soup.find_all(id="wm-ipp"):
        tag.decompose()

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


def extract_original_url(wayback_url):
    """Extract the original URL from a wayback URL."""
    # Pattern: https://web.archive.org/web/TIMESTAMP/ORIGINAL_URL
    match = re.match(r'https?://web\.archive\.org/web/\d+/(https?://.*)', wayback_url)
    if match:
        return match.group(1)
    return wayback_url


def fetch_url(session, url):
    wayback_url = WAYBACK_PREFIX + url
    try:
        resp = session.get(wayback_url, timeout=20, allow_redirects=True)
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
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    })

    count = 0
    while queue and count < MAX_PAGES:
        batch = []
        while queue and len(batch) < WORKERS:
            batch.append(queue.popleft())

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_to_url = {
                executor.submit(fetch_url, session, url): url
                for url in batch
            }
            for future in as_completed(future_to_url):
                original_url = future_to_url[future]
                resp = future.result()
                if resp is None:
                    continue

                title, text = clean_html(resp.text, original_url)
                if len(text) < 50:
                    continue

                pages.append({
                    "url": original_url,  # Store original URL, not wayback URL
                    "title": title,
                    "text": text,
                })
                count += 1

                if count % 50 == 0:
                    print(f"Crawled {count} www2 pages, queue: {len(queue)}...")

                # Extract links from the page
                soup = BeautifulSoup(resp.text, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    # Handle wayback-modified links
                    if "/web/" in href and "web.archive.org" in href:
                        href = extract_original_url(href)
                    full_url = urljoin(original_url, href)
                    full_url = full_url.split("#")[0]
                    if is_valid_url(full_url):
                        norm = normalize_url(full_url)
                        if norm not in visited:
                            visited.add(norm)
                            queue.append(full_url)

        time.sleep(0.5)  # Be nice to archive.org

    return pages


def chunk_text(text, max_chunk_size=500, overlap=75):
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

    print("Starting crawl of www2.eecs.berkeley.edu via web.archive.org...")
    pages = crawl()
    print(f"Crawled {len(pages)} www2 pages total.")

    # Append to existing pages
    pages_file = os.path.join(OUTPUT_DIR, "pages.jsonl")
    with open(pages_file, "a") as f:
        for page in pages:
            f.write(json.dumps(page) + "\n")

    # Create chunks and append to existing chunks
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

    chunks_file = os.path.join(OUTPUT_DIR, "chunks.jsonl")
    with open(chunks_file, "a") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Added {len(all_chunks)} chunks from {len(pages)} www2 pages.")

    # Count total
    total = sum(1 for _ in open(chunks_file))
    print(f"Total chunks in corpus: {total}")


if __name__ == "__main__":
    main()
