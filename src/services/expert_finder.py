"""Expert Finder service.

Incremental (Phase 1) implementation of an agentic "expert finder" skill:
 - Lightweight intent detection (heuristic) for provider lookups.
 - Maps user phrasing to a normalized provider label + (example) NUCC taxonomy code.
 - Performs location-based search using OpenStreetMap Overpass API (no key required).
 - Ranks & returns compact provider payloads (distance + basic metadata).
 - Provides in-memory caching & privacy safeguards (rounded coordinates; no persistence).

Future phases (not yet implemented here):
 - NPI Registry verification & fuzzy match scoring.
 - Google Places / commercial index integration (if API key provided).
 - Auto outreach drafting & approval workflow.
 - Telehealth fallback and follow-up reminders.
"""

from __future__ import annotations

import math
import os
import time
import re
import threading
from typing import Dict, List, Optional, Tuple, Any
import requests

# ---------------------- Taxonomy & Category Mapping ----------------------

# Minimal illustrative mapping (label -> NUCC taxonomy code).
# These codes are examples; extend/adjust for production coverage.
TAXONOMY_MAP: Dict[str, str] = {
    "dermatologist": "207N00000X",
    "cardiologist": "207RC0000X",
    "clinical psychologist": "103T00000X",
    "psychologist": "103T00000X",
    "therapist": "225X00000X",  # Occupational / generic therapist placeholder
    "dentist": "122300000X",
    "nutritionist": "133V00000X",
    "psychiatrist": "2084P0800X",
    "primary care": "261QP2300X",  # Clinic/Center, Primary Care
}

# Mapping from normalized label to Overpass amenity / healthcare tags.
# Each entry is a list of (key, value) pairs to OR together.
OSM_TAGS: Dict[str, List[Tuple[str, str]]] = {
    "dermatologist": [("healthcare:speciality", "dermatology"), ("amenity", "doctors")],
    "cardiologist": [("healthcare:speciality", "cardiology"), ("amenity", "doctors")],
    "clinical psychologist": [("healthcare:speciality", "psychology"), ("amenity", "clinic")],
    "psychologist": [("healthcare:speciality", "psychology"), ("amenity", "clinic")],
    "therapist": [("healthcare", "psychotherapist"), ("amenity", "clinic")],
    "dentist": [("amenity", "dentist")],
    "nutritionist": [("healthcare:speciality", "nutrition"), ("amenity", "clinic")],
    "psychiatrist": [("healthcare:speciality", "psychiatry"), ("amenity", "clinic")],
    "primary care": [("amenity", "doctors"), ("amenity", "clinic")],
}


# ---------------------- In-memory Cache ----------------------

class _TTLCache:
    def __init__(self, ttl_seconds: int = 900):  # 15 min
        self.ttl = ttl_seconds
        self._data: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        now = time.time()
        with self._lock:
            if key in self._data:
                ts, val = self._data[key]
                if now - ts < self.ttl:
                    return val
                else:
                    del self._data[key]
        return None

    def set(self, key: str, val: Any):
        with self._lock:
            self._data[key] = (time.time(), val)


CACHE = _TTLCache()


# ---------------------- Intent Detection (Robust Heuristic) ----------------------

TRIGGER_WORDS = {
    "find", "need", "near", "nearby", "recommend", "locate", "get", "book", "looking", "search", "see"
}

# Map specialty canonical label -> list of keyword fragments that imply it.
SPECIALTY_SYNONYMS: Dict[str, List[str]] = {
    "dermatologist": ["dermatolog", "derm", "skin"],
    "cardiologist": ["cardiolog", "cardio", "heart"],
    "clinical psychologist": ["clinical psychologist"],
    "psychologist": ["psycholog", "mental health", "anxiety", "depress"],
    "therapist": ["therap", "counselor", "counsellor"],
    "dentist": ["dentist", "tooth", "teeth"],
    "nutritionist": ["nutrition", "dietician", "dietitian"],
    "psychiatrist": ["psychiat", "bipolar"],
    "primary care": ["primary care", "gp", "family doctor", "general practitioner"],
}

# Pre-build reverse lookup list of (fragment, canonical_label) sorted by fragment length desc
_FRAG_INDEX: List[Tuple[str, str]] = []
for lbl, frags in SPECIALTY_SYNONYMS.items():
    for f in frags:
        _FRAG_INDEX.append((f.lower(), lbl))
_FRAG_INDEX.sort(key=lambda x: -len(x[0]))  # longest first to avoid short fragment swallowing


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def detect_expert_intent(user_text: str) -> Optional[Dict[str, str]]:
    """Heuristic intent classifier for provider search.

    Strategy (robust to punctuation / ordering):
      1. Tokenize & check if ANY trigger word appears OR phrase starts with 'find a/an'.
      2. Scan for specialty fragments (longest-first) and pick first hit.
      3. Fallback to qualitative clues (heart/skin/anxiety/etc.).
      4. Return structured payload; optionally include debug if EXPERT_DEBUG=1.
    """
    raw = (user_text or "").strip()
    if not raw:
        return None
    low = raw.lower()
    toks = _tokenize(raw)

    # Trigger detection
    triggered = bool(TRIGGER_WORDS & set(toks)) or low.startswith("find a") or low.startswith("find an")
    if not triggered:
        # Loose fallback: question form "can you find" etc.
        if not ("can you find" in low or "help me find" in low):
            return None

    # Specialty detection
    chosen_label: Optional[str] = None
    matched_fragment: Optional[str] = None
    for frag, label in _FRAG_INDEX:
        if frag in low:
            # Use canonical key as stored in TAXONOMY_MAP (guard against accidental substring anomalies)
            canonical = next((k for k in TAXONOMY_MAP.keys() if k == label), label)
            chosen_label = canonical
            matched_fragment = frag
            break

    # Additional fallback heuristics
    if chosen_label is None:
        if "skin" in low:
            chosen_label = "dermatologist"; matched_fragment = "skin"
        elif "heart" in low or "cardio" in low:
            chosen_label = "cardiologist"; matched_fragment = "heart"
        elif any(x in low for x in ["anxiety", "depress", "mental health", "panic"]):
            chosen_label = "psychologist"; matched_fragment = "anxiety/depress"
        elif any(x in low for x in ["tooth", "teeth"]):
            chosen_label = "dentist"; matched_fragment = "tooth/teeth"
        else:
            chosen_label = "primary care"; matched_fragment = "fallback"

    taxonomy = TAXONOMY_MAP.get(chosen_label, "")
    payload = {
        "intent": "expert_lookup",
        "raw": raw,
        "label": chosen_label,
        "taxonomy": taxonomy,
    }
    if os.getenv("EXPERT_DEBUG") == "1":  # type: ignore[name-defined]
        payload["_debug"] = {
            "triggered": triggered,
            "matched_fragment": matched_fragment,
            "tokens": toks[:40],
        }
    return payload

# Simple self-test utility (optional)
def _selftest_examples():  # pragma: no cover (manual aid)
    samples = [
        "Can you find me a cardiologist?",
        "Need a good dermatologist nearby",
        "Looking to book a therapist for anxiety",
        "Find a nutrition expert",
        "I want to see a heart doctor",
        "Help me find primary care",
    ]
    return {s: detect_expert_intent(s) for s in samples}


# ---------------------- Geospatial Helpers ----------------------

def _round_coord(x: float, places: int = 3) -> float:
    return round(x, places)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------- OSM / Overpass Query ----------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _build_overpass_query(lat: float, lng: float, radius_m: int, label: str) -> str:
    """Build an Overpass QL query for the given label using OSM_TAGS mapping."""
    tag_pairs = OSM_TAGS.get(label, [("amenity", "clinic")])
    # Build OR group
    ors = []
    for k, v in tag_pairs:
        # Some tags like healthcare:speciality require quoting
        ors.append(f'node["{k}"="{v}"](around:{radius_m},{lat},{lng});')
        ors.append(f'way["{k}"="{v}"](around:{radius_m},{lat},{lng});')
        ors.append(f'relation["{k}"="{v}"](around:{radius_m},{lat},{lng});')
    blocks = "\n".join(ors)
    return f"[out:json][timeout:15];\n({blocks}\n);\nout center 40;"


def _query_overpass(query: str, timeout_s: int = 20) -> Optional[dict]:
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout_s)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def search_providers(lat: float, lng: float, label: str, taxonomy: str, radius_km: float = 10.0, limit: int = 8) -> Dict[str, Any]:
    """Search nearby providers via Overpass. Returns JSON-safe dict.

    Privacy: coordinates are rounded before caching.
    Ranking: distance + rudimentary score heuristic (closer is better, presence of contact info yields minor boost).
    """
    radius_km = max(1.0, min(radius_km, 50.0))
    radius_m = int(radius_km * 1000)
    rlat, rlng = _round_coord(lat), _round_coord(lng)
    cache_key = f"{rlat}:{rlng}:{label}:{int(radius_km)}"
    cached = CACHE.get(cache_key)
    if cached:
        return cached

    query = _build_overpass_query(lat, lng, radius_m, label)
    data = _query_overpass(query)

    elements: List[Dict[str, Any]] = (data or {}).get("elements", []) if data else []

    # --- New: bucket results so true specialty beats generic clinic/doctor hits ---
    specialized_results: List[Dict[str, Any]] = []
    generic_results: List[Dict[str, Any]] = []

    def _is_specialized(tags: Dict[str, str], target_label: str) -> bool:
        """Return True if tags explicitly indicate the requested specialty."""
        spec = tags.get("healthcare:speciality") or tags.get("healthcare")
        if not spec:
            return False
        norm = spec.lower()
        # loose contains to catch combined specialties e.g. "dermatology;allergy"
        return any(tok in norm for tok in target_label.split())

    results: List[Dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags", {})
        if el.get("type") == "node":
            plat = el.get("lat"); plon = el.get("lon")
        else:
            center = el.get("center", {})
            plat = center.get("lat"); plon = center.get("lon")
        if plat is None or plon is None:
            continue
        dist = haversine_km(lat, lng, plat, plon)
        if dist > radius_km * 1.2:
            continue

        name = tags.get("name") or tags.get("operator") or "(Unnamed Provider)"
        phone = tags.get("phone") or tags.get("contact:phone")
        website = tags.get("website") or tags.get("contact:website")
        addr_parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")]
        address = ", ".join([p for p in addr_parts if p]) or tags.get("addr:full") or "(No address)"
        score = max(0.0, 1.0 - dist / radius_km) + (0.05 if phone else 0) + (0.05 if website else 0)

        rec = {
            "name": name,
            "specialty": label.title(),
            "distance_km": round(dist, 2),
            "address": address,
            "phone": phone,
            "website": website,
            "booking_link": None,
            "npi_match_status": "unverified",
            "lat": plat,
            "lng": plon,
            "score": round(score, 3),
            "notes": "OSM derived; verify details before clinical decisions.",
            "_raw_tags": tags if os.getenv("EXPERT_DEBUG") == "1" else None,
        }

        if _is_specialized(tags, label):
            specialized_results.append(rec)
        else:
            generic_results.append(rec)

    # Prefer specialized; fallback to generic if empty
    if specialized_results:
        results = specialized_results
    else:
        results = generic_results

    # Fallback sample if still empty
    if not results:
        results = [
            {
                "name": f"Example {label.title()} Clinic",
                "specialty": label.title(),
                "distance_km": 3.2,
                "address": "(Sample) 123 Wellness Way",
                "phone": None,
                "website": None,
                "booking_link": None,
                "npi_match_status": "unverified",
                "lat": lat + 0.01,
                "lng": lng + 0.01,
                "score": 0.5,
                "notes": "Sample placeholder due to no live results.",
            }
        ]

    results.sort(key=lambda r: (-r["score"], r["distance_km"]))
    trimmed = [r for r in results if "_raw_tags" in r and r["_raw_tags"] is not None] if os.getenv("EXPERT_DEBUG") == "1" else results
    trimmed = trimmed[:limit]

    payload = {
        "taxonomy": taxonomy,
        "label": label,
        "radius_km": radius_km,
        "results": trimmed,
        "disclaimer": (
            "Informational only; always verify licensing & availability. Not a substitute for professional medical advice."
        ),
    }
    CACHE.set(cache_key, payload)
    return payload


__all__ = [
    "detect_expert_intent",
    "search_providers",
    "TAXONOMY_MAP",
]

