import pandas as pd, glob, os, re, sys

DATA_DIR = "data/processed"
OUT_CSV  = os.path.join(DATA_DIR, "metric_stems_2024.csv")

files = glob.glob(os.path.join(DATA_DIR, "*_weekly_2024.csv")) + glob.glob(os.path.join(DATA_DIR, "*_team_2024.csv"))

stems = set()
cols_by_stem = {}

def add_stem(col):
    base = re.sub(r"(_off|_def|_team)$", "", col)
    stems.add(base)
    cols_by_stem.setdefault(base, set()).add(col)

for f in files:
    try:
        df = pd.read_csv(f, nrows=5)
    except Exception as e:
        print(f"[WARN] Could not read {f}: {e}", file=sys.stderr)
        continue
    for col in df.columns:
        if col in {"season","week","team","side","season_type","game_id","opponent","home_away"}:
            continue
        add_stem(col)

rows = []
for s in sorted(stems):
    cols = sorted(cols_by_stem.get(s, []))
    rows.append({"stem": s, "columns_found": ", ".join(cols)})

out = pd.DataFrame(rows)
os.makedirs(DATA_DIR, exist_ok=True)
out.to_csv(OUT_CSV, index=False)
print(f"Found {len(out)} unique stems. Saved to: {OUT_CSV}")
print(out.head(20).to_string(index=False))
