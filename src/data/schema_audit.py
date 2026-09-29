import pandas as pd

CONCEPTS = {
    "survey_weight": ["weight", "factor", "elev", "ponder"],
    "origin": ["origin", "origen"],
    "destination": ["destination", "destino", "desti"],
    "start_time": ["start time", "departure", "hora", "inici"],
    "mode": ["mode", "modo", "mitja", "transport"],
    "duration": ["duration", "duracion", "durada"],
    "purpose": ["purpose", "motivo", "motiu"],
}

def audit_dictionary(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    text = df.astype(str).agg(" | ".join, axis=1).str.lower()
    out = []
    for concept, terms in CONCEPTS.items():
        hits = df[text.apply(lambda s: any(t in s for t in terms))]
        out.append({
            "concept": concept,
            "candidate_rows": len(hits),
            "status": "REVIEW_REQUIRED" if len(hits) else "NOT_FOUND",
            "candidate_text": " || ".join(
                hits.astype(str).agg(" | ".join, axis=1).head(5)
            ),
        })
    return pd.DataFrame(out)
