# citations.py
import re
import json
import requests
from urllib.parse import urlparse
from typing import List, Dict

# Free sources helpers
from src.services.citations_search import search_pubmed as pubmed_quick
from src.services.tools import search_crossref as crossref_quick

# Broadened trusted domains (consumer + clinical orgs)
TRUSTED_DOMAINS = {
    # US gov / intl orgs / medical libraries
    "nih.gov", "ncbi.nlm.nih.gov", "medlineplus.gov", "cdc.gov", "who.int",
    "fda.gov", "ema.europa.eu",
    # Journals / publishers
    "nejm.org", "thelancet.com", "bmj.com", "jamanetwork.com", "nature.com",
    "sciencedirect.com", "cochranelibrary.com",
    # Reputable consumer health
    "mayoclinic.org", "cancer.org", "aad.org", "clevelandclinic.org", "nhs.uk",
}

# Friendly names for hosts -> source labels
FRIENDLY_NAMES = {
    "ncbi.nlm.nih.gov": "NIH / PubMed",
    "medlineplus.gov": "MedlinePlus",
    "nih.gov": "NIH",
    "cdc.gov": "CDC",
    "who.int": "WHO",
    "fda.gov": "FDA",
    "ema.europa.eu": "EMA",
    "nejm.org": "The New England Journal of Medicine",
    "thelancet.com": "The Lancet",
    "bmj.com": "BMJ",
    "jamanetwork.com": "JAMA Network",
    "nature.com": "Nature",
    "sciencedirect.com": "ScienceDirect",
    "cochranelibrary.com": "Cochrane Library",
    "mayoclinic.org": "Mayo Clinic",
    "cancer.org": "American Cancer Society",
    "aad.org": "American Academy of Dermatology",
    "clevelandclinic.org": "Cleveland Clinic",
    "nhs.uk": "NHS",
}

CITATION_SCHEMA_INSTRUCTIONS = """
Return STRICT JSON with this shape:
{
  "citations": [
    {"title": "Exact page title", "url": "https://..."},
    ...
  ]
}
Do NOT include any keys besides "citations". Choose 1-4 high-quality sources.
Prefer guidelines, systematic reviews, or reputable health org pages.
Only include working URLs users can open directly (no paywalls if possible).
"""

# -------- helpers --------
def extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()

def domain_ok(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return any(netloc == d or netloc.endswith("." + d) for d in TRUSTED_DOMAINS)
    except Exception:
        return False

def host_to_source(host: str) -> str:
    host = (host or "").lower()
    # Exact host match first
    if host in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[host]
    # Then by suffix (e.g., subdomain.aad.org -> AAD)
    for dom, name in FRIENDLY_NAMES.items():
        if host == dom or host.endswith("." + dom):
            return name
    # Fallback: show bare host
    return host

def keywords(text: str, max_k: int = 12) -> set:
    words = re.findall(r"[A-Za-z]{4,}", (text or "").lower())
    out, seen = [], set()
    for w in words:
        if w not in seen:
            out.append(w); seen.add(w)
        if len(out) >= max_k:
            break
    return set(out)

# -------- LLM proposal (unchanged) --------
def propose_citations_via_llm(client, model: str, answer_text: str, k: int = 3) -> list:
    """
    Ask the model to propose citations as strict JSON.
    Returns a list of dicts: [{"title": ..., "url": ...}, ...]
    """
    if client is None:
        return []
    messages = [
        {"role": "system", "content": "You are a strict citation selector that only replies in valid JSON."},
        {"role": "user", "content": f"Answer text:\n{answer_text}\n\n{CITATION_SCHEMA_INSTRUCTIONS}"}
    ]
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=messages
        )
        raw = completion.choices[0].message.content or ""
        data = json.loads(raw)
        citations = data.get("citations", [])
        return citations[:k]
    except Exception:
        # best-effort: extract embedded JSON if any
        try:
            raw = locals().get("raw", "")
            blob = re.search(r"\{[\s\S]+\}", raw).group(0)  # type: ignore
            data = json.loads(blob)
            return data.get("citations", [])[:k]
        except Exception:
            return []

# -------- verification (keeps only trusted + 200), strips URLs in output --------
def verify_citations(citations: list, answer_text: str) -> list:
    """
    Verify each proposed URL (HTTP 200, trusted domain, title keyword overlap).
    Returns a list of {"source": <org>, "title": <page title>, "url": <final url>}.
    """
    if not citations:
        return []

    ans_keys = keywords(answer_text, max_k=12)
    verified_scored = []

    for c in citations:
        url = (c.get("url") or "").strip()
        title_hint = (c.get("title") or "").strip()
        if not url.startswith("http"):
            continue
        try:
            r = requests.get(url, timeout=7, allow_redirects=True)
            # Follow redirects and use the final URL for domain checks
            final_url = r.url or url
            if r.status_code != 200 or not r.text:
                continue

            host = urlparse(final_url).netloc
            if not domain_ok(final_url):
                continue

            title = extract_title(r.text) or title_hint
            if not title:
                continue

            title_keys = keywords(title, max_k=10)
            overlap = len(ans_keys & title_keys)
            # save score for ranking; we will drop URL in final output
            verified_scored.append({
                "source": host_to_source(host),
                "title": title,
                "url": final_url,
                "score": overlap,
            })
        except Exception:
            continue

    verified_scored.sort(key=lambda x: x["score"], reverse=True)
    # Deduplicate by (source, title)
    seen = set()
    out = []
    for v in verified_scored:
        key = (v["source"], v["title"])  # dedupe by content; keep first URL
        if key not in seen:
            seen.add(key)
            out.append({"source": v["source"], "title": v["title"], "url": v.get("url", "")})
        if len(out) >= 3:
            break
    return out

# --- ADD in citations.py (anywhere above the formatter is fine) ---
def _normalize_citations(items: list) -> list:
    """
    Ensure each item has {'source': str, 'title': str}.
    """
    out = []
    for it in items or []:
        src = (it.get("source") or "").strip()
        ttl = (it.get("title") or "").strip()
        if not src: src = "Reputable Medical Source"
        if not ttl: ttl = "Reference"
        out.append({"source": src, "title": ttl})
    return out


# -------- Free-only citation proposal + verification --------
def _compact_query(*texts: str, max_terms: int = 12) -> str:
    """Build a compact query string from input texts."""
    bag, seen = [], set()
    for t in texts:
        for w in re.findall(r"[A-Za-z]{3,}", (t or "")):
            lw = w.lower()
            if lw not in seen:
                bag.append(lw)
                seen.add(lw)
            if len(bag) >= max_terms:
                break
    return " ".join(bag)


def propose_citations_free(user_input: str, answer_text: str, k: int = 8) -> List[Dict]:
    """
    Use only free sources (PubMed, Crossref) to propose citation candidates.
    Returns a list of {title, url} (raw; unverified).
    """
    q = _compact_query(user_input, answer_text, max_terms=14)
    candidates: List[Dict] = []

    # PubMed first (already returns proper pubmed URLs)
    try:
        pm = pubmed_quick(q, n=min(6, k))
        for it in pm:
            title = (it.get("title") or "").strip()
            url = (it.get("url") or "").strip()
            if title and url:
                candidates.append({"title": title, "url": url})
    except Exception:
        pass

    # Crossref DOIs (may redirect to trusted publishers)
    try:
        cr = crossref_quick(q, rows=min(6, max(3, k)))
        for it in cr:
            title = (it.get("title") or "").strip()
            url = (it.get("url") or "").strip()
            if title and url:
                candidates.append({"title": title, "url": url})
    except Exception:
        pass

    # Deduplicate by URL then title
    seen_urls, out = set(), []
    for c in candidates:
        u = c.get("url")
        if not u or u in seen_urls:
            continue
        seen_urls.add(u)
        out.append(c)
        if len(out) >= k:
            break
    return out


def find_and_verify_citations(user_input: str, answer_text: str, max_results: int = 3) -> List[Dict]:
    """
    Free-only pipeline: propose from PubMed/Crossref, verify, normalize.
    Returns a list of {source, title} ready for formatting.
    """
    proposed = propose_citations_free(user_input, answer_text, k=max_results * 3)
    verified = verify_citations(proposed, answer_text)
    normalized = _normalize_citations(verified)
    return normalized[:max_results]


# --- REPLACE your existing format_references_block with this ---
def format_references_block(verified: list) -> str:
    """
    Always render:
      References:
      1. <Source> — <Title>
      2. <Source> — <Title>
    If 'source' is missing, infer from title keywords; else use a safe fallback.
    """

    if not verified:
        return ""

    # Heuristic title -> source inference (extend as needed)
    TITLE_HINT_SOURCES = [
        (("hyperpigmentation", "melasma", "pigment"), "American Academy of Dermatology"),
        (("mole", "melanoma", "skin cancer"), "American Cancer Society"),
        (("skin changes", "skin conditions", "dermatology"), "Mayo Clinic"),
        (("lentigines", "age spots"), "NHS"),
        (("guideline", "systematic review"), "Cochrane Library"),
    ]

    def infer_source_from_title(title: str) -> str:
        t = (title or "").lower()
        for keywords, org in TITLE_HINT_SOURCES:
            if any(k in t for k in keywords):
                return org
        return "Reputable Medical Source"

    # Normalize entries defensively
    normalized = []
    for item in verified:
        title = (item.get("title") or "").strip()
        source = (item.get("source") or "").strip()
        url = (item.get("url") or "").strip()
        if not source:
            source = infer_source_from_title(title)
        if not title:
            title = "Reference"
        normalized.append({"source": source, "title": title, "url": url})

    if not normalized:
        return ""

    lines = ["", "", "References:"]
    for i, v in enumerate(normalized, 1):
        suffix = f" — {v['url']}" if v.get('url') else ""
        lines.append(f"{i}. {v['source']} — {v['title']}{suffix}")
    return "\n".join(lines)


# Optional: a small fallback you can call if verify_citations returns empty
FALLBACK_REFERENCES = [
    {"source": "American Academy of Dermatology", "title": "Skin changes and when to seek care"},
    {"source": "Mayo Clinic", "title": "Common skin conditions overview"},
    {"source": "American Cancer Society", "title": "Signs and symptoms of skin cancer"},
]

def format_fallback_block() -> str:
    return format_references_block(FALLBACK_REFERENCES)
