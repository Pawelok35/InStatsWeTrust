# etl/ingest.py
import argparse
import sys
from typing import List, Optional
import pandas as pd

# ====== ZESTAWY KOLUMN ======
KEEP_COLS_FULL = [
    # meta
    "season", "week", "game_id", "game_date",
    "posteam", "defteam", "home_team", "away_team",
    # sytuacyjne
    "down", "ydstogo", "yardline_100", "qtr", "time",
    # play id / opis
    "play_id", "play_type", "yards_gained", "desc",
    # modele
    "epa", "wp", "wpa",
    # 3rd down / Bonus9
    "first_down", "first_down_rush", "first_down_pass", "first_down_penalty",
    "qb_dropback", "qb_spike", "qb_kneel",
    "touchdown", "pass", "rush",
    # kary
    "no_play", "penalty", "penalty_team",
]

KEEP_COLS_MIN = [
    "season","week","game_id","posteam","defteam",
    "down","ydstogo","yards_gained","epa","play_type"
]

BOOL_LIKE = [
    "first_down", "first_down_rush", "first_down_pass", "first_down_penalty",
    "qb_dropback", "qb_spike", "qb_kneel",
    "touchdown", "pass", "rush",
    "no_play", "penalty"
]

# ====== UTILS ======
def write_df(df: pd.DataFrame, out_path: str):
    if out_path.lower().endswith(".parquet"):
        try:
            import pyarrow  # noqa: F401
        except Exception as e:
            raise SystemExit("❌ Brak pakietu 'pyarrow'. Zainstaluj: pip install pyarrow") from e
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

def coerce_bool01(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
        else:
            df[c] = 0
    return df

def ensure_first_down(df: pd.DataFrame) -> pd.DataFrame:
    """Jeżeli brak 'first_down', zbuduj go z *_rush/pass/penalty lub z warunku yards_gained>=ydstogo."""
    if "first_down" not in df.columns:
        fr   = df.get("first_down_rush", 0)
        fp   = df.get("first_down_pass", 0)
        fpen = df.get("first_down_penalty", 0)
        gained_fd = (df.get("yards_gained", 0).fillna(0) >= df.get("ydstogo", 9999).fillna(9999)).astype(int)
        df["first_down"] = (fr.fillna(0).astype(int) | fp.fillna(0).astype(int) | fpen.fillna(0).astype(int) | gained_fd).astype(int)
    return df

def filter_weeks(df: pd.DataFrame, weeks: Optional[List[int]]) -> pd.DataFrame:
    if weeks:
        df = df[df["week"].isin(weeks)].reset_index(drop=True)
    return df

def import_pbp(seasons: List[int]) -> pd.DataFrame:
    try:
        from nfl_data_py import import_pbp_data
    except Exception as e:
        raise SystemExit(
            "❌ Brak pakietu nfl-data-py. Zainstaluj:\n"
            "    pip install nfl-data-py\n"
            f"Szczegóły: {e}"
        )
    return import_pbp_data(seasons)

def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    # kluczowe typy liczbowe
    for c in ("season","week","down"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("ydstogo","epa","yards_gained","yardline_100","wp","wpa"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ====== GŁÓWNA FUNKCJA INGEST ======
def ingest_pbp(
    seasons: List[int],
    out_path: str,
    mode: str = "full",
    weeks: Optional[List[int]] = None,
):
    assert mode in {"full","min"}, "mode musi być 'full' albo 'min'"

    print(f"⬇️  Pobieram PBP dla sezonów: {', '.join(map(str, seasons))} …")
    pbp = import_pbp(seasons)

    keep_cols = KEEP_COLS_FULL if mode == "full" else KEEP_COLS_MIN
    keep = [c for c in keep_cols if c in pbp.columns]
    df = pbp[keep].copy()

    # Uzupełnij brakujące kolumny z definicji (przewidywalny schema)
    for c in keep_cols:
        if c not in df.columns:
            df[c] = 0 if c in BOOL_LIKE else pd.NA

    df = clean_types(df)
    df = coerce_bool01(df, BOOL_LIKE)
    df = ensure_first_down(df)

    # Usuń rekordy bez down (np. kickoff-only)
    if "down" in df.columns:
        df = df[df["down"].notna()].reset_index(drop=True)

    # Filtr tygodni (opcjonalnie)
    if weeks:
        df = filter_weeks(df, weeks)

    print(f"💾 Zapisuję: {out_path}")
    write_df(df, out_path)
    print(f"✅ Gotowe: {out_path} (rows={len(df):,})")

# ====== CLI ======
def parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Ingest danych PBP (full/min) do CSV/Parquet w zależności od rozszerzenia.")
    ap.add_argument("--season", type=int, nargs="+", required=True, help="Sezon(y) np. 2024 2025")
    ap.add_argument("--out", type=str, required=True, help="Ścieżka wyjściowa (.parquet lub .csv)")
    ap.add_argument("--mode", type=str, choices=["full","min"], default="full", help="full = bogaty schema (Bonus9), min = lekki")
    ap.add_argument("--weeks", type=str, help="Lista tygodni, np. 1,2,3 (opcjonalnie)")
    args = ap.parse_args()

    seasons = args.season
    weeks = parse_int_list(args.weeks)
    try:
        ingest_pbp(seasons=seasons, out_path=args.out, mode=args.mode, weeks=weeks)
    except Exception as e:
        print(f"❌ Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
