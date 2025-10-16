# etl/build_core12_weekly.py
import pandas as pd
from pathlib import Path

SEASON = 2025
IN_DIR = Path("data/processed")
OFF = IN_DIR / f"epa_offense_summary_{SEASON}_weekly.csv"
DEF = IN_DIR / f"epa_defense_summary_{SEASON}_weekly.csv"
OUT = IN_DIR / f"team_core12_weekly_{SEASON}.csv"

def rename_off(off: pd.DataFrame) -> pd.DataFrame:
    return off.rename(columns={
        "plays": "plays_off",
        "avg_epa": "off_epa_per_play",
        "median_epa": "off_median_epa",
        "success_rate": "success_rate_off",
        "explosive_plays": "explosive_plays_off",
        "explosive_rate": "explosive_rate_off",
        "total_yards": "total_yards_off",
        "early_down_epa": "early_down_epa_off",
        "late_down_epa": "late_down_epa_off",
        "third_down_sr": "third_down_sr_off",
        "fourth_down_sr": "fourth_down_sr_off",
        "red_zone_epa": "red_zone_epa_off",
        "pass_epa_per_play": "pass_epa_per_play_off",
        "rush_epa_per_play": "rush_epa_per_play_off",
        "turnover_epa": "turnover_epa_off",
        "avg_start_yardline_100": "avg_start_yardline_100_off",
    })

def rename_def(deff: pd.DataFrame) -> pd.DataFrame:
    return deff.rename(columns={
        "plays_allowed": "plays_def",
        "avg_epa_allowed": "def_epa_per_play_allowed",
        "median_epa_allowed": "def_median_epa_allowed",
        "success_rate_allowed": "success_rate_allowed",
        "explosive_allowed": "explosive_plays_allowed",
        "explosive_rate_allowed": "explosive_rate_allowed",
        "yards_allowed": "yards_allowed",
        "early_down_epa_allowed": "early_down_epa_allowed",
        "late_down_epa_allowed": "late_down_epa_allowed",
        "third_down_sr_allowed": "third_down_sr_allowed",
        "fourth_down_sr_allowed": "fourth_down_sr_allowed",
        "red_zone_epa_allowed": "red_zone_epa_allowed",
        "pass_epa_per_play_allowed": "pass_epa_per_play_allowed",
        "rush_epa_per_play_allowed": "rush_epa_per_play_allowed",
        "turnover_epa_forced": "turnover_epa_forced",
        "avg_start_yardline_100_faced": "avg_start_yardline_100_faced",
    })

def add_net_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["net_epa"] = df["off_epa_per_play"] - df["def_epa_per_play_allowed"]
    df["net_early_down_epa"] = df["early_down_epa_off"] - df["early_down_epa_allowed"]
    df["net_late_down_epa"] = df["late_down_epa_off"] - df["late_down_epa_allowed"]
    df["net_third_down_sr"] = df["third_down_sr_off"] - df["third_down_sr_allowed"]
    df["net_fourth_down_sr"] = df["fourth_down_sr_off"] - df["fourth_down_sr_allowed"]
    df["net_explosive_rate"] = df["explosive_rate_off"] - df["explosive_rate_allowed"]
    df["net_red_zone_epa"] = df["red_zone_epa_off"] - df["red_zone_epa_allowed"]
    df["pass_rush_delta_off"] = df["pass_epa_per_play_off"] - df["rush_epa_per_play_off"]
    df["field_pos_advantage"] = -df["avg_start_yardline_100_off"] + df["avg_start_yardline_100_faced"]
    df["turnover_epa_net"] = df["turnover_epa_off"] - df["turnover_epa_forced"]
    return df

def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team","week"]).copy()
    for col in ["net_epa","off_epa_per_play","def_epa_per_play_allowed"]:
        df[f"{col}_roll3"] = df.groupby("team")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df[f"{col}_delta3"] = df.groupby("team")[col].transform(lambda s: s.diff(3))
    df["momentum_3w"] = (
        0.6*df["net_epa_delta3"].fillna(0)
        + 0.25*df["off_epa_per_play_delta3"].fillna(0)
        - 0.15*df["def_epa_per_play_allowed_delta3"].fillna(0)
    )
    return df

def main():
    if not OFF.exists() or not DEF.exists():
        raise SystemExit("❌ Brak weekly summaries. Najpierw uruchom: python etl/build_weekly_summaries.py --season 2024")
    off = pd.read_csv(OFF)
    deff = pd.read_csv(DEF)
    off = rename_off(off)
    deff = rename_def(deff)

    need = {"season","week","team"}
    if not need.issubset(off.columns) or not need.issubset(deff.columns):
        raise SystemExit("❌ Wejście musi zawierać kolumny season, week, team.")

    df = off.merge(deff, on=["season","week","team"], how="inner")

    # — zabezpieczenie: team to zawsze string
    df["team"] = df["team"].astype(str)

    # — wypełniamy TYLKO kolumny numeryczne
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    df = add_net_metrics(df)
    df = add_momentum(df)


    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Zapisano weekly Core12 + momentum: {OUT}")

if __name__ == "__main__":
    main()
