#!/usr/bin/env python3
"""Targeted crawl of www2.eecs.berkeley.edu key pages via web.archive.org."""

import re
import json
import os
import hashlib
import time
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup

WAYBACK_PREFIX = "https://web.archive.org/web/2025/"

# Key pages to fetch - these are the most important sections
SEED_URLS = [
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Courses/EECS/",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule-draft.html",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/",
]

SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".css", ".js", ".zip", ".tar", ".gz", ".mp4", ".mp3",
    ".bib", ".ps", ".eps", ".dvi", ".xml", ".rss",
}

OUTPUT_DIR = "corpus"


def is_valid_www2_url(url):
    parsed = urlparse(url)
    if "www2.eecs.berkeley.edu" not in parsed.netloc:
        return False
    path_lower = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path_lower.endswith(ext):
            return False
    return True


def normalize_url(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def extract_original_url(wayback_url):
    match = re.match(r'https?://web\.archive\.org/web/\d+/(https?://.*)', wayback_url)
    if match:
        return match.group(1)
    return wayback_url


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(id=re.compile("wm-")):
        tag.decompose()
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    main = (soup.find("main") or soup.find("article") or
            soup.find("div", {"id": "content"}) or soup.body)
    if main is None:
        main = soup

    text = main.get_text(separator="\n", strip=True)
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
    return title, text, soup


def fetch_and_extract_links(session, url):
    """Fetch a page and return its content + discovered links."""
    wayback_url = WAYBACK_PREFIX + url
    try:
        resp = session.get(wayback_url, timeout=20, allow_redirects=True)
        if resp.status_code != 200:
            return None, []
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return None, []

        title, text, soup = clean_html(resp.text)
        if len(text) < 50:
            return None, []

        # Extract links
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/web/" in href and "web.archive.org" in href:
                href = extract_original_url(href)
            full_url = urljoin(url, href).split("#")[0]
            if is_valid_www2_url(full_url):
                links.append(full_url)

        return {"url": url, "title": title, "text": text}, links
    except Exception as e:
        return None, []


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    visited = set()
    queue = deque()
    pages = []

    for url in SEED_URLS:
        norm = normalize_url(url)
        if norm not in visited:
            visited.add(norm)
            queue.append(url)

    max_pages = 2000
    count = 0

    while queue and count < max_pages:
        url = queue.popleft()
        print(f"[{count+1}] Fetching: {url}")

        page, links = fetch_and_extract_links(session, url)
        if page:
            pages.append(page)
            count += 1

            # Only follow links from index/listing pages (first 2 levels deep)
            depth = url.count("/") - 3  # base depth
            if depth <= 5:
                for link in links:
                    norm = normalize_url(link)
                    if norm not in visited:
                        visited.add(norm)
                        queue.append(link)

        time.sleep(0.3)

        if count % 50 == 0:
            print(f"  Crawled {count} pages, queue: {len(queue)}")

    print(f"Crawled {len(pages)} www2 pages total.")

    # Also crawl individual faculty homepages
    print("Fetching faculty homepage links...")
    faculty_page, faculty_links = fetch_and_extract_links(session, "https://www2.eecs.berkeley.edu/Faculty/Homepages/")
    if faculty_links:
        faculty_urls = [l for l in faculty_links if "/Faculty/Homepages/" in l and l not in visited]
        print(f"  Found {len(faculty_urls)} faculty pages to fetch")
        for i, furl in enumerate(faculty_urls[:200]):  # cap at 200
            if normalize_url(furl) in visited:
                continue
            visited.add(normalize_url(furl))
            page, _ = fetch_and_extract_links(session, furl)
            if page:
                pages.append(page)
                count += 1
            if (i + 1) % 20 == 0:
                print(f"  Fetched {i+1} faculty pages...")
            time.sleep(0.3)

    print(f"Total www2 pages: {len(pages)}")

    # Append to existing files
    with open(os.path.join(OUTPUT_DIR, "pages.jsonl"), "a") as f:
        for page in pages:
            f.write(json.dumps(page) + "\n")

    # Create chunks
    all_chunks = []
    for page in pages:
        words = page["text"].split()
        if len(words) <= 500:
            chunks = [page["text"]]
        else:
            chunks = []
            start = 0
            while start < len(words):
                end = min(start + 500, len(words))
                chunks.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start = end - 75
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{hashlib.md5(page['url'].encode()).hexdigest()}_{i}",
                "url": page["url"],
                "title": page["title"],
                "text": chunk,
            })

    with open(os.path.join(OUTPUT_DIR, "chunks.jsonl"), "a") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Added {len(all_chunks)} chunks from {len(pages)} www2 pages.")
    total = sum(1 for _ in open(os.path.join(OUTPUT_DIR, "chunks.jsonl")))
    print(f"Total chunks in corpus: {total}")


if __name__ == "__main__":
    main()
