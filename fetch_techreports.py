#!/usr/bin/env python3
"""Fetch individual tech report HTML pages from web.archive.org."""

import re
import json
import os
import hashlib
import time
import requests
from bs4 import BeautifulSoup

WAYBACK_PREFIX = "https://web.archive.org/web/2025/"
OUTPUT_DIR = "corpus"


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(id=re.compile("wm-")):
        tag.decompose()
    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return title, "\n".join(lines)


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Get list of individual tech report URLs from our corpus
    report_urls = set()
    for year in ["2023", "2024", "2025"]:
        listing_url = f"https://www2.eecs.berkeley.edu/Pubs/TechRpts/{year}/"
        try:
            wb_url = WAYBACK_PREFIX + listing_url
            r = session.get(wb_url, timeout=20)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # Look for links to individual tech reports
                wb_match = re.match(r'/web/\d+/(https?://.*)', href)
                if wb_match:
                    href = wb_match.group(1)
                if "EECS-" in href and href.endswith(".html"):
                    if "www2.eecs.berkeley.edu" in href:
                        report_urls.add(href)
        except Exception as e:
            print(f"Error fetching {listing_url}: {e}")

    print(f"Found {len(report_urls)} individual tech report pages to fetch.")

    # Fetch individual pages
    pages = []
    for i, url in enumerate(sorted(report_urls)):
        try:
            wb_url = WAYBACK_PREFIX + url
            r = session.get(wb_url, timeout=15, allow_redirects=True)
            if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
                title, text = clean_html(r.text)
                if len(text) >= 50:
                    pages.append({"url": url, "title": title, "text": text})
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  Fetched {i+1}/{len(report_urls)}, got {len(pages)} pages")
        time.sleep(0.15)

    print(f"Fetched {len(pages)} tech report pages.")

    # Append to corpus
    with open(os.path.join(OUTPUT_DIR, "pages.jsonl"), "a") as f:
        for page in pages:
            f.write(json.dumps(page) + "\n")

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

    total = sum(1 for _ in open(os.path.join(OUTPUT_DIR, "chunks.jsonl")))
    print(f"Added {len(all_chunks)} chunks. Total: {total}")


if __name__ == "__main__":
    main()
