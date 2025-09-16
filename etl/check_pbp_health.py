# etl/check_pbp_health.py
from __future__ import annotations
import sys
import pandas as pd

def compute_drive_starts_coverage(df: pd.DataFrame) -> float | None:
    """
    Szacuje pokrycie startów drive’ów:
    - preferuje metadane 'drive_start_yard_line' (jeśli są),
    - w przeciwnym razie bierze pierwszą akcję w drive’ie z nie-NaN 'yardline_100'.
    Zwraca ułamek [0..1] albo None, jeśli brak kolumn drive/drive_id.
    """
    key = "drive_id" if "drive_id" in df.columns else ("drive" if "drive" in df.columns else None)
    if key is None:
        return None

    # 1) jeśli mamy kolumnę startową z metadanych — liczymy samo pokrycie (czy jest wartość)
    if "drive_start_yard_line" in df.columns:
        starts = (
            df.sort_values(["game_id", "posteam", key, "qtr", "time", "play_id"])
              .drop_duplicates(["game_id", "posteam", key], keep="first")["drive_start_yard_line"]
        )
        return float(starts.notna().mean())

    # 2) fallback: „pierwsza nie-NaN yardline_100” w obrębie drive’u
    ordered = df.sort_values(["game_id", "posteam", key, "qtr", "time", "play_id"])
    # pick pierwsze wiersze drive’ów:
    firsts = ordered.drop_duplicates(["game_id", "posteam", key], keep="first")
    has = firsts["yardline_100"].notna().mean()

    # jeżeli pierwszy wiersz często ma NaN (kickoff/penalty), spróbujmy wziąć pierwszą NIE-NaN w obrębie drive’u
    if has < 0.95:
        nonan = ordered[ordered["yardline_100"].notna()]
        pick = nonan.drop_duplicates(["game_id", "posteam", key], keep="first")
        idx = ["game_id", "posteam", key]
        merged = firsts[idx].merge(pick[idx + ["yardline_100"]], on=idx, how="left")
        return float(merged["yardline_100"].notna().mean())

    return float(has)

def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/pbp_clean_2024_new.parquet"
    df = pd.read_parquet(path)

    print("FILE:", path)

    # Podstawowa kontrola yardline_100
    if "yardline_100" not in df.columns:
        print("ERROR: brak kolumny 'yardline_100' w pliku — sprawdź fetch.")
        return

    na = df["yardline_100"].isna().mean()
    ymin = float(df["yardline_100"].min())
    ymax = float(df["yardline_100"].max())
    print("yardline_100 null%:", round(na * 100, 3), "%")
    print("yardline_100 min/max:", ymin, ymax)

    # Pokrycie startów drive’ów
    cov = compute_drive_starts_coverage(df)
    if cov is None:
        print("drive starts coverage: n/a (brak kolumn drive_id/drive)")
    else:
        print("drive starts coverage:", f"{round(cov * 100, 2)} %")

if __name__ == "__main__":
    main()
