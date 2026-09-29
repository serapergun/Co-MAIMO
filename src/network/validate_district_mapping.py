EXPECTED_BCN_DISTRICTS={
"Ciutat Vella","Eixample","Sants-Montjuïc","Les Corts",
"Sarrià-Sant Gervasi","Gràcia","Horta-Guinardó","Nou Barris",
"Sant Andreu","Sant Martí"}

def validate_mapping(df):
    present=set(df["district"].dropna().astype(str))
    missing=EXPECTED_BCN_DISTRICTS-present
    counts=df.groupby("district")["edge_id"].nunique().to_dict()
    return {"pass":not missing and all(v>0 for v in counts.values()),
            "missing_districts":sorted(missing),"edges_per_district":counts}
