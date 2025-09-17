# tests/verify_penalty_discipline.py
# Szybka walidacja metryki Penalty Discipline (OFF/DEF)
# Użycie (przykład):
#   python tests/verify_penalty_discipline.py \
#       --season 2024 \
#       --in_pbp data/processed/pbp_clean_2024.parquet \
#       --team DET --side off --week 1 --show-rows 10

import argparse
import sys
from pathlib import Path
import pandas as pd

# import z naszego modułu
sys.path.append(str(Path(__file__).resolve().parents[1]))  # dodaje root repo do PYTHONPATH
from etl.build_penalty_discipline import build  # noqa: E402

EXPECTED_WEEKLY_COLS = [
    "game_id","week","team","side",
    "plays","penalties","penalty_yds",
    "pen_per_100_plays","presnap_pen_rate","rz_pen_rate",
    "third_fourth_pen","auto_fd_allowed","dpi_yds"
]

EXPECTED_TEAM_COLS = [
    "team","side","games","plays",
    "penalties","penalties_pg",
    "penalty_yds","penalty_yds_pg",
    "pen_per_100_plays",
    "presnap_pen_rate","rz_pen_rate",
    "third_fourth_pen","third_fourth_pen_pg",
    "auto_fd_allowed","def_auto_fd_allowed_pg",
    "dpi_yds","def_dpi_yds_pg"
]

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=Path, required=True)
    ap.add_argument("--team", type=str, default=None)
    ap.add_argument("--side", type=str, choices=["off","def"], default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--show-rows", type=int, default=8)
    args = ap.parse_args()

    print(f"📥 Load PBP: {args.in_pbp}")
    df = _read_any(args.in_pbp)

    print("🔧 Build weekly/team frames…")
    weekly, team = build(df, season=args.season)

    # --------------- Struktura kolumn
    missing_w = [c for c in EXPECTED_WEEKLY_COLS if c not in weekly.columns]
    missing_t = [c for c in EXPECTED_TEAM_COLS if c not in team.columns]
    assert not missing_w, f"Brak kolumn w weekly: {missing_w}"
    assert not missing_t, f"Brak kolumn w team: {missing_t}"

    # --------------- Podstawowe zakresy i typy
    assert set(weekly["side"].unique()) <= {"off","def"}, "Nieznane wartości w 'side'"
    for col in ["plays","penalties","penalty_yds","third_fourth_pen","auto_fd_allowed","dpi_yds"]:
        assert (weekly[col].fillna(0) >= 0).all(), f"Ujemne wartości w weekly.{col}"
    for col in ["presnap_pen_rate","rz_pen_rate"]:
        s = weekly[col].fillna(0)
        assert ((s >= 0) & (s <= 1)).all(), f"Poza zakresem [0,1]: weekly.{col}"
    s = weekly["pen_per_100_plays"].fillna(0)
    assert ((s >= 0) & (s <= 300)).all(), "Poza zakresem: weekly.pen_per_100_plays"

    # --------------- Spójność: sumy weekly vs team
    wk_sum = (weekly
              .groupby(["team","side"], dropna=False)
              .agg(plays=("plays","sum"),
                   penalties=("penalties","sum"),
                   penalty_yds=("penalty_yds","sum"))
              .reset_index())
    merged = team.merge(wk_sum, on=["team","side"], suffixes=("_team","_wk"))
    assert (merged["plays_team"] == merged["plays_wk"]).all(), "plays team ≠ suma weekly"
    assert (merged["penalties_team"] == merged["penalties_wk"]).all(), "penalties team ≠ suma weekly"
    assert (merged["penalty_yds_team"] == merged["penalty_yds_wk"]).all(), "penalty_yds team ≠ suma weekly"

    # --------------- Podgląd (opcjonalny)
    if args.team:
        filt = (weekly["team"] == args.team)
        if args.side:
            filt &= (weekly["side"] == args.side)
        if args.week:
            filt &= (weekly["week"] == args.week)

        sample = weekly.loc[filt].sort_values(["week","side","team"]).head(args.show_rows)
        print("\n📄 Weekly sample:")
        if sample.empty:
            print(" (brak wierszy dla podanego filtra)")
        else:
            print(sample.to_string(index=False))

        trow = team[(team["team"] == args.team) & ((team["side"] == args.side) if args.side else True)]
        print("\n📊 Team row(s):")
        if trow.empty:
            print(" (brak wierszy w team dla podanego filtra)")
        else:
            print(trow.to_string(index=False))

    print("\n✅ Wszystkie podstawowe testy przeszły pomyślnie.")

if __name__ == "__main__":
    main()
