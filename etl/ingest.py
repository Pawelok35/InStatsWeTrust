from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

def get_latest_completed_week(season: int) -> int:
    # games: jeden rekord na mecz – bierzemy max(week) z zakończonych spotkań
    games = nfl.import_schedules([season])
    played = games[games['game_type'].eq('REG') & games['result'].notna()]
    if played.empty:
        # przed sezonem – fallback na Week 1
        return 1
    return int(played['week'].max())

def main():
    # parametry
    SEASON = 2024  # na start 2023 (stabilne); potem możesz zmienić na 2024/2025
    latest_week = get_latest_completed_week(SEASON)

    # ścieżki
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Pobieram PBP dla sezonu {SEASON} ...")
    pbp = nfl.import_pbp_data([SEASON], downcast=True, cache=False)

    print(f"🪚 Filtrowanie tygodnia: {latest_week}")
    pbp_week = pbp[pbp["week"].eq(latest_week)].copy()

    out_csv = raw_dir / f"pbp_{SEASON}_week_{latest_week}.csv"
    pbp_week.to_csv(out_csv, index=False)
    print(f"✅ Zapisano {len(pbp_week):,} wierszy do: {out_csv}")

    # szybki podgląd
    cols = [
        "season","week","game_id","home_team","away_team","qtr","time",
        "down","ydstogo","yardline_100","play_type","yards_gained",
        "epa","posteam","defteam","desc"
    ]
    preview_cols = [c for c in cols if c in pbp_week.columns]
    if not pbp_week.empty:
        print("📋 Podgląd wierszy:")
        print(pbp_week.head(3)[preview_cols].to_string(index=False))
    else:
        print("⚠️ Brak danych dla tego tygodnia – sprawdź czy sezon/tydzień istnieją.")

if __name__ == "__main__":
    main()
