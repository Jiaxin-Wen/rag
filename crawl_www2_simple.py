#!/usr/bin/env python3
"""Simple targeted crawl of www2.eecs.berkeley.edu key pages via web.archive.org.
Fetches listing pages, extracts real links, then fetches linked pages."""

import re
import json
import os
import hashlib
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

WAYBACK_PREFIX = "https://web.archive.org/web/2025/"
OUTPUT_DIR = "corpus"

SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".css", ".js", ".zip", ".tar", ".gz", ".mp4", ".mp3",
    ".bib", ".ps", ".eps", ".dvi", ".xml", ".rss",
}


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove wayback toolbar
    for tag in soup.find_all(id=re.compile("wm-")):
        tag.decompose()
    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    main = (soup.find("main") or soup.find("article") or
            soup.find("div", {"id": "content"}) or soup.body or soup)

    text = main.get_text(separator="\n", strip=True)

    # Extract tables structurally
    tables_text = []
    for table in main.find_all("table"):
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


def extract_www2_links(soup, base_url):
    """Extract links from a page, resolving wayback URLs to originals."""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Resolve wayback URLs to originals
        wb_match = re.match(r'/web/\d+/(https?://.*)', href)
        if wb_match:
            href = wb_match.group(1)
        elif href.startswith("http"):
            # Check for embedded wayback
            wb_match2 = re.match(r'https?://web\.archive\.org/web/\d+/(https?://.*)', href)
            if wb_match2:
                href = wb_match2.group(1)

        full_url = urljoin(base_url, href).split("#")[0].split("?")[0]
        parsed = urlparse(full_url)
        if "www2.eecs.berkeley.edu" in parsed.netloc:
            path_lower = parsed.path.lower()
            skip = False
            for ext in SKIP_EXTENSIONS:
                if path_lower.endswith(ext):
                    skip = True
                    break
            if not skip:
                links.add(full_url)
    return links


def fetch_page(session, url):
    """Fetch a page via archive.org."""
    wayback_url = WAYBACK_PREFIX + url
    try:
        resp = session.get(wayback_url, timeout=20, allow_redirects=True)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            title, text, soup = clean_html(resp.text)
            if len(text) >= 50:
                return {"url": url, "title": title, "text": text}, soup
    except Exception:
        pass
    return None, None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Phase 1: Fetch index/listing pages and collect links
    listing_pages = [
        "https://www2.eecs.berkeley.edu/Courses/CS/",
        "https://www2.eecs.berkeley.edu/Courses/EE/",
        "https://www2.eecs.berkeley.edu/Courses/EECS/",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
        "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule-draft.html",
        "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html",
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/",
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/",
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/",
    ]

    all_pages = []
    all_links = set()
    fetched_urls = set()

    print("Phase 1: Fetching listing pages...")
    for url in listing_pages:
        print(f"  Fetching: {url}")
        page, soup = fetch_page(session, url)
        if page:
            all_pages.append(page)
            fetched_urls.add(url)
            links = extract_www2_links(soup, url)
            all_links.update(links)
            print(f"    Got {len(links)} links")
        time.sleep(0.3)

    # Phase 2: Fetch linked pages (courses, faculty, tech reports)
    to_fetch = all_links - fetched_urls
    # Prioritize: faculty homepages and courses
    priority_urls = sorted([u for u in to_fetch if "/Faculty/Homepages/" in u or "/Courses/" in u])
    other_urls = sorted([u for u in to_fetch if u not in set(priority_urls)])

    # Cap to avoid too many requests
    fetch_list = priority_urls[:500] + other_urls[:200]

    print(f"\nPhase 2: Fetching {len(fetch_list)} linked pages...")
    count = 0
    for i, url in enumerate(fetch_list):
        if url in fetched_urls:
            continue
        page, soup = fetch_page(session, url)
        if page:
            all_pages.append(page)
            fetched_urls.add(url)
            count += 1
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(fetch_list)}, fetched: {count}")
        time.sleep(0.2)

    print(f"\nTotal www2 pages fetched: {len(all_pages)}")

    # Save and chunk
    with open(os.path.join(OUTPUT_DIR, "pages.jsonl"), "a") as f:
        for page in all_pages:
            f.write(json.dumps(page) + "\n")

    all_chunks = []
    for page in all_pages:
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

    total = sum(1 for _ in open(os.path.join(OUTPUT_DIR, "chunks.jsonl")))
    print(f"Added {len(all_chunks)} chunks. Total corpus: {total} chunks.")


if __name__ == "__main__":
    main()
