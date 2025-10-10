"""Scoring & normalization for citations v2 (Phase 1)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

WORD_RE = re.compile(r"[A-Za-z]{3,}")

EVIDENCE_WEIGHTS = {
    "guideline": 1.0,
    "systematic_review": 0.95,
    "meta_analysis": 0.95,
    "rct": 0.85,
    "observational": 0.7,
    "consumer": 0.5,
    "other": 0.6,
}


@dataclass
class EvidenceItem:
    title: str
    url: str
    source_label: str
    year: str | None
    doi: str | None
    pmid: str | None
    journal: str | None
    evidence_level: str
    score: float


def _token_set(text: str) -> Set[str]:
    return {w.lower() for w in WORD_RE.findall(text or "")}


def classify_evidence(raw: Dict) -> str:
    title = (raw.get("title") or "").lower()
    if any(k in title for k in ["guideline", "consensus"]):
        return "guideline"
    if "systematic review" in title:
        return "systematic_review"
    if "meta-analysis" in title or "meta analysis" in title:
        return "meta_analysis"
    if any(k in title for k in ["randomized", "randomised", "trial"]):
        return "rct"
    return "other"


def compute_score(raw: Dict, focus_terms: Set[str]) -> float:
    tset = _token_set(raw.get("title") or "")
    overlap = len(tset & focus_terms)
    coverage = overlap / max(1, len(focus_terms))
    ev = classify_evidence(raw)
    ew = EVIDENCE_WEIGHTS.get(ev, 0.6)
    year = raw.get("year")
    try:
        y = int(str(year)[:4]) if year else None
    except Exception:
        y = None
    recency = 0.0
    if y:
        import datetime
        current = datetime.datetime.utcnow().year
        age = max(0, current - y)
        recency = max(0.0, 0.2 - 0.02 * age)
    return round((0.4 * coverage) + (0.4 * ew) + (0.2 * recency), 4)


def normalize_and_score(raw_items: List[Dict], focus_terms: Set[str]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    for r in raw_items:
        title = (r.get("title") or "").strip()
        if not title:
            continue
        url = r.get("url") or (f"https://doi.org/{r['doi']}" if r.get("doi") else None)
        if not url:
            continue
        ev = classify_evidence(r)
        score = compute_score(r, focus_terms)
        source_label = "NIH / PubMed" if r.get("_src") == "pubmed" else (r.get("journal") or "Journal")
        items.append(EvidenceItem(
            title=title,
            url=url,
            source_label=source_label,
            year=str(r.get("year")) if r.get("year") else None,
            doi=r.get("doi"),
            pmid=r.get("pmid"),
            journal=r.get("journal"),
            evidence_level=ev,
            score=score,
        ))
    seen_doi = set()
    seen_title = set()
    deduped: List[EvidenceItem] = []
    for it in sorted(items, key=lambda x: x.score, reverse=True):
        key_title = it.title.lower()
        if it.doi and it.doi in seen_doi:
            continue
        if not it.doi and key_title in seen_title:
            continue
        if it.doi:
            seen_doi.add(it.doi)
        else:
            seen_title.add(key_title)
        deduped.append(it)
    return deduped


__all__ = ["EvidenceItem", "normalize_and_score"]
