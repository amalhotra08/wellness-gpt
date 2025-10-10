from legacy.compare_rsids import compare_gene_conditions
import pandas as pd
from typing import List, Dict

def run_genomics(tsv_path: str, out_csv_path: str) -> List[Dict]:
    compare_gene_conditions(tsv_path, out_csv_path)
    df = pd.read_csv(out_csv_path)
    # Normalize to a stable schema the chat can consume
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": str(row.get("ID", "")),
            "gene": str(row.get("Gene", "")),
            "genotype": str(row.get("Genotype", "")),
            "summary": str(row.get("Summary", "")),
        })
    return records