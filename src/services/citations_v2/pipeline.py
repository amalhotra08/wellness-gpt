"""Public pipeline entry for citations v2 (Phase 1)."""
from __future__ import annotations

from typing import List, Dict
from .query import build_query_bundle
from .adapters import gather_raw
from .scoring import normalize_and_score

def gather_verified_citations(user_input: str, answer_text: str, max_results: int = 3) -> List[Dict]:
    qb = build_query_bundle(user_input or "", answer_text or "")
    if not qb.primary:
        return []
    raw = gather_raw(qb.variants, per_source_limit=max(3, max_results))
    scored = normalize_and_score(raw, qb.focus_terms)
    top = scored[:max_results]
    out = []
    for it in top:
        out.append({
            "source": it.source_label,
            "title": it.title,
            "url": it.url,
            "evidence_level": it.evidence_level,
            "year": it.year,
        })
    return out

__all__ = ["gather_verified_citations"]
