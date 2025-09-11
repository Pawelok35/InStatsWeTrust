import pandas as pd
from pathlib import Path

TEAM_WEEK = Path("data/processed/team_core12_weekly_2024.csv")
SCHEDULE  = Path("data/processed/schedule_2024.csv")  # kolumny: season, week, home_team, away_team, game_id
OUT       = Path("data/processed/game_features_2024.csv")

HOME_PREFIX = "home_"
AWAY_PREFIX = "away_"

def suffix(df: pd.DataFrame, prefix: str, team_col: str) -> pd.DataFrame:
    df = df.rename(columns={"team": team_col})
    keep = [c for c in df.columns if c not in {team_col}]
    return df.rename(columns={c: prefix+c for c in keep})

def main():
    if not TEAM_WEEK.exists():
        raise SystemExit("❌ Brak team_core12_weekly_2024.csv (uruchom weekly ETL).")
    if not SCHEDULE.exists():
        raise SystemExit("❌ Brak schedule_2024.csv (uruchom build_schedule.py).")

    tw = pd.read_csv(TEAM_WEEK)
    sched = pd.read_csv(SCHEDULE)

    home = suffix(tw, HOME_PREFIX, "home_team")
    away = suffix(tw, AWAY_PREFIX, "away_team")

    gf = sched.merge(
        home, left_on=["season","week","home_team"],
        right_on=[HOME_PREFIX+"season", HOME_PREFIX+"week", "home_team"], how="left"
    )
    gf = gf.merge(
        away, left_on=["season","week","away_team"],
        right_on=[AWAY_PREFIX+"season", AWAY_PREFIX+"week", "away_team"], how="left"
    )

    diff_cols = ["net_epa","off_epa_per_play","def_epa_per_play_allowed","momentum_3w"]
    for c in diff_cols:
        gf[f"diff_{c}"] = gf[f"{HOME_PREFIX}{c}"] - gf[f"{AWAY_PREFIX}{c}"]

    drop_cols = [HOME_PREFIX+"season", HOME_PREFIX+"week", AWAY_PREFIX+"season", AWAY_PREFIX+"week"]
    gf = gf.drop(columns=[c for c in drop_cols if c in gf.columns])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gf.to_csv(OUT, index=False)
    print(f"✅ Zapisano game features: {OUT}")

if __name__ == "__main__":
    main()
