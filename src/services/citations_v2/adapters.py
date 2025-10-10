"""Source adapters for citations v2 (Phase 1)."""
from __future__ import annotations

from typing import List, Dict
from src.services.tools import search_pubmed, search_crossref


def fetch_pubmed(query: str, limit: int = 6) -> List[Dict]:
    try:
        return search_pubmed(query, retmax=limit)
    except Exception:
        return []


def fetch_crossref(query: str, limit: int = 6) -> List[Dict]:
    try:
        return search_crossref(query, rows=limit)
    except Exception:
        return []


def gather_raw(query_variants: List[str], per_source_limit: int = 5) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for q in query_variants:
        if not q:
            continue
        for r in fetch_pubmed(q, limit=per_source_limit):
            r["_src"] = "pubmed"
            key = (r.get("pmid"), r.get("title"))
            if key not in seen:
                seen.add(key); out.append(r)
        for r in fetch_crossref(q, limit=per_source_limit):
            r["_src"] = "crossref"
            key = (r.get("doi"), r.get("title"))
            if key not in seen:
                seen.add(key); out.append(r)
    return out


__all__ = ["gather_raw"]
