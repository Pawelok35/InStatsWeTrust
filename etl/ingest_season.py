from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

SEASON = 2024
WEEKS = list(range(1, 19))  # 1..18 (regular season)

def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Importuję PBP dla sezonu {SEASON} (bez cache)…")
    pbp = nfl.import_pbp_data([SEASON], downcast=True, cache=False)

    # Zapis per tydzień
    frames = []
    for wk in WEEKS:
        wk_df = pbp[pbp["week"].eq(wk)].copy()
        out_w = raw_dir / f"pbp_{SEASON}_week_{wk}.csv"
        wk_df.to_csv(out_w, index=False)
        frames.append(wk_df)
        print(f"✅ Week {wk:02d}: zapisano {len(wk_df):,} wierszy -> {out_w.name}")

    # Zapis scalony (all weeks)
    all_df = pd.concat(frames, ignore_index=True)
    out_all = raw_dir / f"pbp_{SEASON}_all.csv"
    all_df.to_csv(out_all, index=False)
    print(f"🎉 Done. Scalony plik: {out_all} (wierszy: {len(all_df):,})")

if __name__ == "__main__":
    main()
