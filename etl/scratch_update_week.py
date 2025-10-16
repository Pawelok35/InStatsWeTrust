#!/usr/bin/env python3
import argparse, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Update PBP to a chosen week (idempotent).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True, help="Target week to include (1..18)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    season, week, out_dir = args.season, args.week, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"pbp_clean_{season}.parquet"

    # 1) Import danych PBP (nfl_data_py)
    try:
        import nfl_data_py as nfl
        import pandas as pd
    except Exception:
        sys.exit("❌ Brak zależności. Zainstaluj: pip install nfl-data-py pandas pyarrow")

    print(f"📥 Importuję PBP dla {season}… (to może chwilę trwać)")
    df = nfl.import_pbp_data([season], downcast=True)

    # 2) Minimalne czyszczenie/typy i filtr do week <= N
    import pandas as pd
    need_cols = [
        "season","week","posteam","defteam","epa","success","down","yardline_100",
        "is_pass","is_rush","interception","fumble","yards_gained","st_play"
    ]
    for c in need_cols:
        if c not in df.columns:
            # kolumny, które mogą nie wystąpić uzupełniamy zerami/NA (zabezpieczenie)
            if c in ("is_pass","is_rush","interception","fumble","st_play"):
                df[c] = 0
            else:
                df[c] = pd.NA

    # Typy liczbowe gdzie ma sens
    num_cols = ["epa","success","down","yardline_100","is_pass","is_rush","interception","fumble","yards_gained","st_play"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Tylko do podanego tygodnia
    df = df[(df["season"] == season) & (df["week"] <= week)].copy()

    # 3) (opcjonalnie) wywal zupełnie puste wiersze ofensywne/defensywne
    #    (zostawiamy special teams, bo część metryk ich używa — st_play mamy jako 0/1)
    df = df.reset_index(drop=True)

    # 4) Zapis
    try:
        df.to_parquet(outp, index=False)
    except Exception:
        # Fallback do CSV, gdy brak pyarrow/fastparquet
        outp = outp.with_suffix(".csv")
        df.to_csv(outp, index=False)

    # 5) Podsumowanie
    weeks = sorted(pd.Series(df["week"].dropna().unique()).astype(int).tolist())
    print(f"✅ Zapisano: {outp}  | rows={len(df)}  | weeks={weeks}")

if __name__ == "__main__":
    main()
