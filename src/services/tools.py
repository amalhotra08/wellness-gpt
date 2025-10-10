import requests, math
from typing import List, Dict

CROSSREF = "https://api.crossref.org/works"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
HEADERS = {"User-Agent": "HMS-Chatbot/1.0 (mailto:you@example.com)"}

def search_crossref(query: str, rows: int = 5) -> List[Dict]:
    r = requests.get(CROSSREF, params={"query.bibliographic": query, "rows": rows}, headers=HEADERS, timeout=10)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    out = []
    for it in items:
        out.append({
            "title": (it.get("title") or [""])[0],
            "doi": it.get("DOI"),
            "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "journal": (it.get("container-title") or [""])[0],
            "url": f"https://doi.org/{it.get('DOI')}" if it.get("DOI") else None,
        })
    return out

def search_pubmed(query: str, retmax: int = 5) -> List[Dict]:
    ids = requests.get(PUBMED_ESEARCH, params={"db":"pubmed","term":query,"retmode":"json","retmax":retmax}, timeout=10)\
            .json()["esearchresult"].get("idlist", [])
    if not ids: return []
    summ = requests.get(PUBMED_ESUMMARY, params={"db":"pubmed","retmode":"json","id":",".join(ids)}, timeout=10)\
            .json()["result"]
    out = []
    for pmid in ids:
        it = summ.get(pmid, {})
        out.append({
            "pmid": pmid,
            "title": it.get("title"),
            "year": it.get("pubdate", "").split(" ")[0],
            "journal": it.get("fulljournalname"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })
    return out

def relative_risk(a: int, b: int, c: int, d: int) -> Dict:
    # 2x2: exposed cases=a, exposed noncases=b, unexposed cases=c, unexposed noncases=d
    rr = (a / (a+b)) / (c / (c+d))
    import math
    se = math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)))
    lo = math.exp(math.log(rr) - 1.96*se)
    hi = math.exp(math.log(rr) + 1.96*se)
    return {"rr": rr, "ci95": [lo, hi]}
