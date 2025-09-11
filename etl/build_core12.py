"""
Build a single Team Core12 table by merging offense/defense season summaries
and computing derived "net" metrics + ranking.

Usage:
    python etl/build_core12.py --season 2024 --in_dir data/processed --out data/processed/team_core12_2024.csv

Inputs (expected in --in_dir):
    epa_offense_summary_<season>_season.csv
    epa_defense_summary_<season>_season.csv

Outputs:
    team_core12_<season>.csv (detailed metrics)
    power_ranking_<season>.csv (compact ranking by net_epa)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# ---- Required input columns (raw) ----
REQ_OFF = {
    "team",
    # season is optional in source; we'll fill from CLI if missing
    "plays", "avg_epa", "median_epa", "success_rate",
    "explosive_plays", "explosive_rate", "total_yards",
    "early_down_epa", "late_down_epa",
    "third_down_sr", "fourth_down_sr",
    "red_zone_epa", "pass_epa_per_play", "rush_epa_per_play",
    "turnover_epa", "avg_start_yardline_100",
}

REQ_DEF = {
    "team",
    "plays_allowed", "avg_epa_allowed", "median_epa_allowed",
    "success_rate_allowed", "explosive_allowed", "explosive_rate_allowed",
    "yards_allowed", "early_down_epa_allowed", "late_down_epa_allowed",
    "third_down_sr_allowed", "fourth_down_sr_allowed",
    "red_zone_epa_allowed", "pass_epa_per_play_allowed", "rush_epa_per_play_allowed",
    "turnover_epa_forced", "avg_start_yardline_100_faced",
}


def load_tables(season: int, in_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    off_p = in_dir / f"epa_offense_summary_{season}_season.csv"
    def_p = in_dir / f"epa_defense_summary_{season}_season.csv"
    if not off_p.exists():
        raise FileNotFoundError(off_p)
    if not def_p.exists():
        raise FileNotFoundError(def_p)
    off = pd.read_csv(off_p)
    deff = pd.read_csv(def_p)
    return off, deff


def _validate(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] Missing columns: {missing}")


def rename_off(off: pd.DataFrame) -> pd.DataFrame:
    # season optional; ensure present column (filled later)
    if "season" not in off.columns:
        off["season"] = np.nan
    _validate(off, REQ_OFF, "offense")
    mapping = {
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
    }
    return off.rename(columns=mapping)


def rename_def(deff: pd.DataFrame) -> pd.DataFrame:
    if "season" not in deff.columns:
        deff["season"] = np.nan
    _validate(deff, REQ_DEF, "defense")
    mapping = {
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
    }
    return deff.rename(columns=mapping)


def build_core12(off_raw: pd.DataFrame, def_raw: pd.DataFrame, season: int) -> pd.DataFrame:
    off = rename_off(off_raw.copy())
    deff = rename_def(def_raw.copy())

    # Ensure keys exist and proper dtypes
    for df in (off, deff):
        if "team" not in df.columns:
            raise KeyError("Both offense and defense tables must have 'team' column.")
        # fill season if missing in source
        if df["season"].isna().any():
            df.loc[df["season"].isna(), "season"] = season
        df["season"] = df["season"].astype(int)

    # Merge
    df = off.merge(deff, on=["season", "team"], how="inner")

    # Fill NaNs on critical numeric cols to avoid propagating NaN in net_* metrics
    safe_cols = [
        "off_epa_per_play","def_epa_per_play_allowed",
        "early_down_epa_off","early_down_epa_allowed",
        "late_down_epa_off","late_down_epa_allowed",
        "third_down_sr_off","third_down_sr_allowed",
        "fourth_down_sr_off","fourth_down_sr_allowed",
        "explosive_rate_off","explosive_rate_allowed",
        "red_zone_epa_off","red_zone_epa_allowed",
        "pass_epa_per_play_off","rush_epa_per_play_off",
        "turnover_epa_off","turnover_epa_forced",
        "avg_start_yardline_100_off","avg_start_yardline_100_faced",
        "plays_off","plays_def","total_yards_off","yards_allowed",
    ]
    for c in safe_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Derived net metrics
    df["net_epa"] = df["off_epa_per_play"] - df["def_epa_per_play_allowed"]
    df["net_early_down_epa"] = df["early_down_epa_off"] - df["early_down_epa_allowed"]
    df["net_late_down_epa"] = df["late_down_epa_off"] - df["late_down_epa_allowed"]
    df["net_third_down_sr"] = df["third_down_sr_off"] - df["third_down_sr_allowed"]
    df["net_fourth_down_sr"] = df["fourth_down_sr_off"] - df["fourth_down_sr_allowed"]
    df["net_explosive_rate"] = df["explosive_rate_off"] - df["explosive_rate_allowed"]
    df["net_red_zone_epa"] = df["red_zone_epa_off"] - df["red_zone_epa_allowed"]
    df["pass_rush_delta_off"] = df["pass_epa_per_play_off"] - df["rush_epa_per_play_off"]
    # Field position advantage: offense closer (lower yardline_100) and defense facing farther (higher)
    df["field_pos_advantage"] = -df["avg_start_yardline_100_off"] + df["avg_start_yardline_100_faced"]
    # Turnover balance: (off turnovers cost) vs (def turnovers gained)
    df["turnover_epa_net"] = df["turnover_epa_off"] - df["turnover_epa_forced"]

    # Rank by net_epa (desc)
    df = df.sort_values(["season", "net_epa"], ascending=[True, False])
    df["rank_net_epa"] = df.groupby("season")["net_epa"].rank(method="first", ascending=False).astype(int)

    # Order columns for readability
    col_order = [
        "season", "team", "rank_net_epa",
        # headline
        "net_epa", "off_epa_per_play", "def_epa_per_play_allowed",
        # conversion
        "net_third_down_sr", "third_down_sr_off", "third_down_sr_allowed",
        "net_fourth_down_sr", "fourth_down_sr_off", "fourth_down_sr_allowed",
        # early/late
        "net_early_down_epa", "early_down_epa_off", "early_down_epa_allowed",
        "net_late_down_epa", "late_down_epa_off", "late_down_epa_allowed",
        # explosive
        "net_explosive_rate", "explosive_rate_off", "explosive_rate_allowed",
        # red zone
        "net_red_zone_epa", "red_zone_epa_off", "red_zone_epa_allowed",
        # style
        "pass_rush_delta_off", "pass_epa_per_play_off", "rush_epa_per_play_off",
        # turnover & field position
        "turnover_epa_net", "turnover_epa_off", "turnover_epa_forced",
        "field_pos_advantage", "avg_start_yardline_100_off", "avg_start_yardline_100_faced",
        # volume
        "plays_off", "plays_def", "total_yards_off", "yards_allowed",
    ]
    final_cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[final_cols]

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    off, deff = load_tables(args.season, args.in_dir)
    core12 = build_core12(off, deff, args.season)

    out_path = args.out or (args.in_dir / f"team_core12_{args.season}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core12.to_csv(out_path, index=False)

    # compact ranking
    ranking = core12[[
        "season", "team", "rank_net_epa",
        "net_epa", "off_epa_per_play", "def_epa_per_play_allowed"
    ]].sort_values(["season", "net_epa"], ascending=[True, False])
    ranking_path = args.in_dir / f"power_ranking_{args.season}.csv"
    ranking.to_csv(ranking_path, index=False)

    print(f"✅ Zapisano Core12: {out_path}")
    print(f"🏆 Zapisano ranking: {ranking_path}")


if __name__ == "__main__":
    main()
