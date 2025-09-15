# etl/build_third_down.py
import argparse
import sys
import pandas as pd
from metrics.third_down import third_down_weekly, third_down_season

REQUIRED_COLS = {
    "season", "week", "posteam", "defteam",
    "down", "ydstogo", "epa",
    # opcjonalne, ale zalecane:
    "qb_dropback", "first_down", "first_down_penalty",
    "touchdown", "qb_spike", "qb_kneel"
}

def load_df(path: str) -> pd.DataFrame:
    """Wczytuje CSV lub Parquet na podstawie rozszerzenia."""
    if path.endswith(".parquet"):
        try:
            # wymaga: pip install pyarrow
            return pd.read_parquet(path)
        except Exception as e:
            print(f"❌ Błąd przy czytaniu Parquet: {e}")
            print("   Upewnij się, że masz zainstalowane 'pyarrow' (pip install pyarrow).")
            sys.exit(1)
    else:
        return pd.read_csv(path)

def validate_cols(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print("⚠️  Brakuje kolumn w PBP potrzebnych do 3rd down:")
        for m in sorted(missing):
            print(f"   - {m}")
        print("   Skrypt policzy co może, ale wyniki mogą być niepełne.")
        # Nie przerywamy – metrics.third_down używa _safe_col tam, gdzie się da.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=str, default="data/processed/pbp_{season}.csv")
    ap.add_argument("--out_weekly", type=str, default="data/processed/third_down_weekly_{season}.csv")
    ap.add_argument("--out_team", type=str, default="data/processed/third_down_team_{season}.csv")
    args = ap.parse_args()

    in_pbp = args.in_pbp.format(season=args.season)
    out_weekly = args.out_weekly.format(season=args.season)
    out_team = args.out_team.format(season=args.season)

    # 1) Wczytanie danych
    pbp = load_df(in_pbp)

    # 2) Walidacja kolumn (informacyjnie)
    validate_cols(pbp)

    # 3) Obliczenia
    weekly = third_down_weekly(pbp)
    season_df = third_down_season(weekly)

    # 4) Zapis
    weekly.to_csv(out_weekly, index=False)
    season_df.to_csv(out_team, index=False)
    print(f"✅ Zapisano weekly: {out_weekly}")
    print(f"🏆 Zapisano sezon: {out_team}")

if __name__ == "__main__":
    main()
