#!/usr/bin/env python3
"""RAG pipeline: retrieve relevant passages and generate answers using LLM."""

import sys
import os
import re
import json
import pickle
import unicodedata
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from llm import call_llm, ALLOWED_MODELS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "corpus", "bm25_index.pkl")
DOCS_PATH = os.path.join(SCRIPT_DIR, "corpus", "docs.pkl")
DENSE_INDEX_PATH = os.path.join(SCRIPT_DIR, "corpus", "dense_index.faiss")
DENSE_MODEL_NAME = os.path.join(SCRIPT_DIR, "models", "all-MiniLM-L6-v2")

TOP_K_RETRIEVE = 100
TOP_K_DENSE = 50
TOP_K_URLS = 5
MODEL = "meta-llama/llama-3.1-8b-instruct"
FALLBACK_MODELS = ["qwen/qwen-2.5-7b-instruct", "mistralai/mistral-7b-instruct"]

SYSTEM_PROMPT = """You answer factoid questions about UC Berkeley EECS using provided context.
The current date is March 2026.
Rules:
- Give ONLY the direct short answer. No explanation. No full sentences.
- Maximum 5 words. Usually 1-3 words is best.
- For "how many" questions, give JUST the number (e.g. "6" or "3"). Count carefully.
- For "who" questions, give JUST the full name exactly as written in context.
- For "where/which university" questions, give JUST the university name.
- For yes/no questions, answer ONLY "Yes" or "No". Read the context carefully before answering.
- Copy names/dates/numbers EXACTLY as they appear in context.
- When asked about "earliest", "oldest", "first", etc., carefully compare ALL entries before answering.
- When the question asks about a SPECIFIC category (e.g. "minor credits" not "major credits"), read the context carefully to find the exact matching category.
- When multiple documents discuss similar topics, pick the one that best matches the question's specific details.
- Read ALL provided passages before answering, not just the first one.
- For percentage questions, give JUST the number (e.g. "14" or "48").
- If the answer is not in the context, answer "unknown". Never answer "No" just because you cannot find the answer.
- No preamble, no explanation."""


def strip_diacritics(text):
    """Remove diacritics for better matching (e.g. Gödel -> Godel)."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text):
    text = strip_diacritics(text.lower())
    return re.findall(r'\w+', text)


def load_index():
    with open(INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    # Build URL-to-chunks index for fast lookup
    url_to_chunks = {}
    for i, doc in enumerate(docs):
        url = doc["url"]
        if url not in url_to_chunks:
            url_to_chunks[url] = []
        url_to_chunks[url].append(i)
    # Load dense index
    dense_index = None
    dense_model = None
    if os.path.exists(DENSE_INDEX_PATH):
        print("Loading dense index...", file=sys.stderr)
        dense_index = faiss.read_index(DENSE_INDEX_PATH)
        dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        print(f"Dense index loaded with {dense_index.ntotal} vectors.", file=sys.stderr)
    return bm25, docs, url_to_chunks, dense_index, dense_model


def generate_query_variants(question):
    """Generate multiple query formulations for better recall."""
    variants = [question]
    # Also add diacritic-stripped version of the question
    stripped = strip_diacritics(question)
    if stripped != question:
        variants.append(stripped)
    q_lower = question.lower()
    names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', question)
    courses = re.findall(r'(?:CS|EE|EECS)\s*\d+\w*', question, re.IGNORECASE)

    for name in names:
        variants.append(f"{name} EECS Berkeley faculty")
        variants.append(f"{name} homepage biography education")
    for course in courses:
        variants.append(f"{course} Berkeley EECS course description")

    # Thesis/dissertation questions
    if "thesis" in q_lower or "dissertation" in q_lower:
        variants.append("thesis technical report EECS advisor")
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            variants.append(f"technical report EECS {year_match.group(1)}")
            variants.append(f"dissertation {year_match.group(1)} EECS")
    # Faculty/people questions
    if "born" in q_lower or "earliest" in q_lower or "oldest" in q_lower:
        variants.append("in memoriam faculty born deceased professors")
        variants.append("earliest born professor memoriam")
    if "advisor" in q_lower and "thesis" not in q_lower:
        variants.append("student advisor advising office contact")
    # Course schedule questions
    if "teaching" in q_lower or "teach" in q_lower or ("how many" in q_lower and "course" in q_lower):
        variants.append("schedule draft teaching courses instructor")
        variants.append("EE CS schedule draft courses")
    if "schedule" in q_lower or "spring" in q_lower or "fall" in q_lower:
        variants.append("EE CS schedule draft courses")
        variants.append("schedule draft spring fall semester")
    # Degree requirements
    if "credits" in q_lower or "units" in q_lower or "minor" in q_lower:
        variants.append("coursework credits requirements units")
    if "master" in q_lower:
        variants.append("masters degree education university")
    if "phd" in q_lower or "ph.d" in q_lower:
        if "where" in q_lower or "which" in q_lower:
            variants.append("PhD degree education university")
        if re.search(r'\b(1[89]\d{2}|20\d{2})\b', question):
            year = re.search(r'\b(1[89]\d{2}|20\d{2})\b', question).group(1)
            variants.append(f"PhD {year} memoriam faculty dissertation")
            variants.append(f"Ph.D. {year}")
    if "earned" in q_lower and ("phd" in q_lower or "ph.d" in q_lower or "doctorate" in q_lower):
        variants.append("memoriam PhD doctorate earned faculty")
    # Awards/deadlines/prizes
    if "award" in q_lower or "deadline" in q_lower or "prize" in q_lower:
        variants.append("awards nomination deadline prize")
        variants.append("ACM IEEE award prize distinguished")
    if "fellowship" in q_lower:
        variants.append("fellowship award graduate student")
        variants.append("Sloan fellowship award faculty")
    # Office/contact info
    if "office" in q_lower or "room" in q_lower:
        variants.append("office room hall contact Soda Cory")
    if "email" in q_lower:
        variants.append("email contact address")
    if "phone" in q_lower:
        variants.append("phone number telephone contact")
    # Percentage/demographics questions
    if "percentage" in q_lower or "percent" in q_lower:
        variants.append("demographics percentage enrollment statistics by the numbers")
        variants.append("international students residents percentage demographics")
    # Lab/research questions
    if "lab" in q_lower or "research" in q_lower:
        variants.append("lab laboratory research center")
    # Counting questions about courses at specific level
    if "how many" in q_lower and ("294" in q_lower or "294" in question):
        variants.append("CS 294 courses spring schedule")
    if "how many" in q_lower:
        variants.append(question.replace("How many", "").replace("how many", "").strip())
    # Historical questions
    if "hired" in q_lower or "first" in q_lower or "black" in q_lower:
        variants.append("first Black hired history memoriam")
    # Building/location
    if "hall" in q_lower or "building" in q_lower or "floor" in q_lower:
        variants.append("Soda Cory hall building floor location")
    # Ranking
    if "ranking" in q_lower or "ranked" in q_lower:
        variants.append("ranking QS world university top ranked")
    # Contact/media
    if "contact" in q_lower or "media" in q_lower:
        variants.append("media contact communications press")
    # Capacity/reservation
    if "capacity" in q_lower:
        variants.append("capacity room seats occupancy")
    if "reservation" in q_lower or "book" in q_lower:
        variants.append("room reservation booking lounge")

    return variants


def score_url(question, url, best_chunk_score):
    """Give bonus score to a URL based on question relevance."""
    q_lower = question.lower()
    url_lower = url.lower()
    bonus = 0
    names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', question)

    # Name in URL
    for name in names:
        name_parts = name.lower().split()
        for part in name_parts:
            if len(part) > 3 and part in url_lower:
                bonus += 20

    # URL-type matching
    if "advisor" in q_lower and "thesis" not in q_lower and "advising" in url_lower:
        bonus += 25
    if ("born" in q_lower or "earliest" in q_lower or "oldest" in q_lower) and "memoriam" in url_lower:
        if "/people/faculty/in-memoriam" in url_lower or "in-memoriam" in url_lower:
            bonus += 50
        else:
            bonus += 10
    if ("passed away" in q_lower or "died" in q_lower) and "memoriam" in url_lower:
        bonus += 30
    if ("schedule" in q_lower or "teaching" in q_lower or "teach" in q_lower) and "schedule" in url_lower:
        bonus += 20
    if ("how many" in q_lower and "course" in q_lower) and "schedule" in url_lower:
        bonus += 25
    if ("credit" in q_lower or "coursework" in q_lower or "minor" in q_lower) and "coursework" in url_lower:
        bonus += 25
    if ("thesis" in q_lower or "dissertation" in q_lower):
        if "Pubs/TechRpts" in url or "Dissertations" in url or "Pubs/" in url:
            bonus += 20
    if ("master" in q_lower or "phd" in q_lower or "ph.d" in q_lower or "education" in q_lower):
        if "homepage" in url_lower or "Faculty/Homepages" in url:
            bonus += 15
        if "book/faculty" in url_lower:
            bonus += 15
    if "homepage" in url_lower or "Faculty/Homepages" in url:
        for name in names:
            bonus += 5
    # Demographics/percentage questions
    if ("percentage" in q_lower or "percent" in q_lower or "how many" in q_lower and "student" in q_lower):
        if "by-the-numbers" in url_lower or "numbers" in url_lower:
            bonus += 50
    # Award/prize pages
    if ("award" in q_lower or "prize" in q_lower or "fellow" in q_lower):
        if "awards" in url_lower or "Awards" in url:
            bonus += 20
    # Schedule pages for course-specific questions
    if re.search(r'(?:CS|EE|EECS)\s*\d+', question, re.IGNORECASE) and "schedule" in url_lower:
        bonus += 15
    # Colloquium pages
    if "colloquium" in q_lower and "colloqui" in url_lower:
        bonus += 30
    # BEARS symposium
    if "bears" in q_lower and "bears" in url_lower:
        bonus += 30
    # Phone/contact for specific buildings
    if "phone" in q_lower and ("building" in q_lower or "hall" in q_lower or "manager" in q_lower):
        if "building" in url_lower or "facilities" in url_lower or "contact" in url_lower:
            bonus += 20
    # PhD earned in specific year
    if re.search(r'earned.*ph\.?d|ph\.?d.*\d{4}', q_lower):
        if "memoriam" in url_lower:
            bonus += 30
    # Ranking
    if "ranking" in q_lower and ("about" in url_lower or "numbers" in url_lower):
        bonus += 25
    # Residency
    if "residency" in q_lower and ("residency" in url_lower or "financial" in url_lower):
        bonus += 20
    # Room/lounge booking
    if ("lounge" in q_lower or "woz" in q_lower) and ("room" in url_lower or "lounge" in url_lower or "woz" in url_lower):
        bonus += 25
    # Distinguished alumni
    if "distinguished" in q_lower and "alumni" in q_lower and "alumni" in url_lower:
        bonus += 25

    return bonus


def retrieve(question, bm25, docs, url_to_chunks, dense_index=None, dense_model=None, top_k=TOP_K_URLS):
    """Retrieve top URLs using hybrid BM25 + dense retrieval."""
    variants = generate_query_variants(question)

    # --- BM25 scoring ---
    combined_scores = np.zeros(len(docs))
    for variant in variants:
        tokens = tokenize(variant)
        scores = bm25.get_scores(tokens)
        combined_scores += scores

    # Normalize BM25 scores to [0, 1]
    bm25_max = combined_scores.max()
    if bm25_max > 0:
        bm25_norm = combined_scores / bm25_max
    else:
        bm25_norm = combined_scores

    # --- Dense scoring ---
    dense_norm = np.zeros(len(docs))
    if dense_index is not None and dense_model is not None:
        q_emb = dense_model.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        scores_dense, indices_dense = dense_index.search(q_emb, TOP_K_DENSE)
        for score, idx in zip(scores_dense[0], indices_dense[0]):
            if idx >= 0:
                dense_norm[idx] = float(score)

    # --- Hybrid scoring: 0.6 * BM25 + 0.4 * dense ---
    hybrid_scores = 0.6 * bm25_norm + 0.4 * dense_norm

    # Score per URL: max chunk score + URL-level bonus
    url_scores = {}
    for idx in range(len(docs)):
        if hybrid_scores[idx] <= 0:
            continue
        url = docs[idx]["url"]
        score = float(hybrid_scores[idx])
        if url not in url_scores or score > url_scores[url]:
            url_scores[url] = score

    # Add URL-level bonuses (scaled for hybrid scores)
    url_final_scores = {}
    for url, score in url_scores.items():
        bonus = score_url(question, url, score) * 0.01  # Scale bonuses for normalized scores
        url_final_scores[url] = score + bonus

    # Sort URLs by score
    sorted_urls = sorted(url_final_scores.items(), key=lambda x: x[1], reverse=True)

    # For top URLs, merge all their chunks
    results = []
    for url, score in sorted_urls[:top_k]:
        chunk_indices = url_to_chunks.get(url, [])
        if not chunk_indices:
            continue
        chunks = [docs[i] for i in chunk_indices]
        chunks.sort(key=lambda c: int(c["id"].rsplit("_", 1)[-1]) if "_" in c["id"] else 0)
        merged_text = "\n".join(c["text"] for c in chunks)
        results.append({
            "url": url,
            "title": chunks[0].get("title", ""),
            "text": merged_text,
        })

    return results


def get_question_guidance(question):
    """Return question-type-specific guidance for the LLM."""
    q_lower = question.lower()
    hints = []
    if "percentage" in q_lower or "percent" in q_lower:
        hints.append("Give JUST the number.")
    if "how long" in q_lower and "ago" in q_lower:
        hints.append("Calculate from the current year 2026.")
    return " ".join(hints)


def build_prompt(question, passages):
    context_parts = []
    for i, p in enumerate(passages, 1):
        text = p["text"]
        words = text.split()
        max_words = 800 if i == 1 else 400
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "..."
        context_parts.append(f"[{i}] {p.get('title', '')} ({p['url']})\n{text}")
    context = "\n\n".join(context_parts)

    guidance = get_question_guidance(question)
    guidance_line = f"\nNote: {guidance}" if guidance else ""

    return f"""Context from eecs.berkeley.edu:

{context}

Question: {question}{guidance_line}

Answer with ONLY the exact short answer:"""


def is_garbage(answer):
    if not answer or answer == "unknown":
        return True
    if len(set(answer)) <= 2 and len(answer) > 5:
        return True
    punct_ratio = sum(1 for c in answer if not c.isalnum() and c != ' ') / max(len(answer), 1)
    if punct_ratio > 0.5 and len(answer) > 5:
        return True
    bad_phrases = ["no information", "not found", "not mentioned", "not provided",
                   "cannot be determined", "not available", "does not contain",
                   "i don't know", "i'm not sure", "context does not"]
    for phrase in bad_phrases:
        if phrase in answer.lower():
            return True
    return False


def clean_answer(answer):
    if not answer:
        return "unknown"
    answer = answer.strip()
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    prefixes = ["Answer:", "The answer is:", "The answer is", "A:", "answer:",
                "Based on the context,", "Based on the provided context,",
                "According to the context,", "According to the passages,",
                "Short answer:", "Fact:", "The short answer is:",
                "The short answer is", "**Answer:**", "**Answer**:"]
    for prefix in sorted(prefixes, key=len, reverse=True):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    # Remove markdown bold
    answer = re.sub(r'\*\*', '', answer).strip()
    if len(answer) > 2:
        if (answer[0] == '"' and answer[-1] == '"') or \
           (answer[0] == "'" and answer[-1] == "'"):
            answer = answer[1:-1].strip()
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    if answer.endswith(".") and len(answer.split()) <= 10:
        answer = answer[:-1].strip()
    # Remove trailing location qualifiers after comma (e.g. ", Belgium", ", CA")
    # But preserve commas in time ranges, lists of names, etc.
    if "," in answer:
        parts = answer.split(",")
        suffix = parts[-1].strip()
        # Only strip if suffix looks like a location (1-2 words, no numbers/times)
        if len(suffix.split()) <= 2 and not re.search(r'\d', suffix):
            answer = ",".join(parts[:-1]).strip()
    # Remove "units" suffix from number answers but preserve "+" suffix
    answer = re.sub(r'(\d+\+?)\s+(?:semester\s+)?units?$', r'\1', answer, flags=re.IGNORECASE)
    # Truncate overly long answers
    words = answer.split()
    if len(words) > 10:
        answer = " ".join(words[:8])
    return answer if answer else "unknown"


def safe_call_llm(query, system_prompt, model=MODEL, max_tokens=48):
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    for m in models_to_try:
        try:
            answer = call_llm(
                query=query, system_prompt=system_prompt, model=m,
                max_tokens=max_tokens, temperature=0.0, timeout=30,
            )
            cleaned = clean_answer(answer)
            if not is_garbage(cleaned):
                return cleaned
        except Exception:
            continue
    return "unknown"


def filter_passages(question, passages):
    """Remove redundant/noisy passages based on question type."""
    q_lower = question.lower()

    # For superlative questions (earliest, oldest, most, etc.), prefer list pages
    # over individual articles that might distract the LLM
    superlative_words = ["earliest", "oldest", "first", "most", "latest", "newest",
                         "youngest", "largest", "smallest", "highest", "lowest"]
    if any(w in q_lower for w in superlative_words):
        # Keep only top 3 to reduce noise
        return passages[:3]

    return passages


def answer_question(question, bm25, docs, url_to_chunks, dense_index=None, dense_model=None):
    passages = retrieve(question, bm25, docs, url_to_chunks, dense_index, dense_model)
    if not passages:
        return safe_call_llm(
            f"Answer concisely about UC Berkeley EECS: {question}",
            SYSTEM_PROMPT,
        )
    passages = filter_passages(question, passages)
    prompt = build_prompt(question, passages)
    answer = safe_call_llm(prompt, SYSTEM_PROMPT)
    if is_garbage(answer):
        prompt = build_prompt(question, passages[:2])
        answer = safe_call_llm(prompt, SYSTEM_PROMPT)
    return answer


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 rag.py <questions_txt_path> <predictions_out_path>",
              file=sys.stderr)
        sys.exit(1)

    questions_path = sys.argv[1]
    predictions_path = sys.argv[2]

    with open(questions_path, "r") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions.", file=sys.stderr)
    print("Loading indexes...", file=sys.stderr)
    bm25, docs, url_to_chunks, dense_index, dense_model = load_index()
    print(f"Index loaded with {len(docs)} documents.", file=sys.stderr)

    answers = []
    for i, question in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {question[:80]}...", file=sys.stderr)
        try:
            answer = answer_question(question, bm25, docs, url_to_chunks, dense_index, dense_model)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            answer = "unknown"
        answers.append(answer)
        print(f"  -> {answer}", file=sys.stderr)

    with open(predictions_path, "w") as f:
        for answer in answers:
            f.write(answer + "\n")

    print(f"Wrote {len(answers)} predictions to {predictions_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
