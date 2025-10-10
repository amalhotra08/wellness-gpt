"""Query synthesis utilities for citations v2.

Phase 1: lightweight term extraction & variant generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

STOPWORDS = {
    "the","and","with","for","that","this","from","your","have","been","will","into","about","when","into","you","are","was","were","can","could","should","would","may","might","need","find","please","help","just","like"
}

SYNONYM_MAP = {
    "mole": ["nevus"],
    "moles": ["nevi"],
    "eczema": ["atopic dermatitis"],
    "acne": ["acne vulgaris"],
    "heart": ["cardiac"],
}

WORD_RE = re.compile(r"[A-Za-z]{3,}")

@dataclass
class QueryBundle:
    primary: str
    variants: List[str]
    focus_terms: Set[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_focus_terms(*texts: str, max_terms: int = 12) -> Set[str]:
    bag = []
    seen = set()
    for t in texts:
        for w in WORD_RE.findall(t or ""):
            lw = w.lower()
            if lw in STOPWORDS or len(lw) < 3:
                continue
            if lw not in seen:
                bag.append(lw); seen.add(lw)
            if len(bag) >= max_terms:
                break
        if len(bag) >= max_terms:
            break
    return set(bag)


def build_query_bundle(user_input: str, answer_text: str) -> QueryBundle:
    focus = extract_focus_terms(user_input, answer_text)
    if not focus:
        return QueryBundle(primary="", variants=[], focus_terms=set())

    primary = " ".join(sorted(focus))
    variants = [primary]

    # Add synonym expanded variant
    expanded = []
    for term in focus:
        expanded.append(term)
        if term in SYNONYM_MAP:
            expanded.extend(SYNONYM_MAP[term])
    if len(expanded) != len(focus):
        variants.append(" ".join(sorted(set(expanded))))

    # Phrase variant (first 5 terms)
    variants.append(" ".join(list(focus)[:5]))

    # Deduplicate & trim
    dedup = []
    seen_q = set()
    for v in variants:
        v2 = _normalize(v)
        if v2 and v2 not in seen_q:
            seen_q.add(v2)
            dedup.append(v2)

    return QueryBundle(primary=dedup[0] if dedup else "", variants=dedup[:5], focus_terms=focus)


__all__ = ["QueryBundle", "build_query_bundle"]
