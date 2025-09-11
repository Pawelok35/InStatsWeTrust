# etl/build_schedule.py
import pandas as pd
from pathlib import Path
import nfl_data_py as nfl

SEASON = 2024
OUT = Path("data/processed/schedule_2024.csv")

def main():
    df = nfl.import_schedules([SEASON])[["season","week","game_id","home_team","away_team"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Zapisano: {OUT}")

if __name__ == "__main__":
    main()
