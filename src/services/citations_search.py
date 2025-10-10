# citations_search.py
import requests
from urllib.parse import urlencode

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def search_pubmed(query: str, n: int = 5) -> list[dict]:
    """
    Returns: [{"title": "...", "url": "https://pubmed.ncbi.nlm.nih.gov/<pmid>/"}...]
    No API key required. Keep requests modest to respect NCBI usage.
    """
    if not query.strip():
        return []
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(n),
        "sort": "relevance",
    }
    r = requests.get(PUBMED_ESEARCH, params=params, timeout=6)
    r.raise_for_status()
    js = r.json()
    pmids = js.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    # Fetch summaries (titles)
    params2 = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    r2 = requests.get(PUBMED_ESUMMARY, params=params2, timeout=6)
    r2.raise_for_status()
    js2 = r2.json()
    result = []
    for pmid in pmids:
        rec = js2.get("result", {}).get(pmid, {})
        title = rec.get("title") or f"PubMed {pmid}"
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        result.append({"title": title, "url": url})
    return result
