#!/usr/bin/env python3
"""
Build 1st Down Success (Offense/Defense) from PBP.

Definition (aligned with standard SR):
- A 1st-down play is a SUCCESS if yards_gained >= 40% of yards_to_go at snap.
- Exclusions: penalties-only (no-play), spikes, kneels, timeouts, 2pt tries,
  laterals/fumble returns on ST, obvious non-scrimmage plays.

Outputs (CSV):
1) weekly-level per team & side (off/def):
   [season, week, team, side, plays, successes, sr, epa_per_play]
2) season team aggregate per side:
   [season, team, side, plays, successes, sr, epa_per_play]

CLI
----
python etl/build_first_down_success.py \
  --season 2024 \
  --in_pbp data/processed/pbp_clean_2024.parquet \
  --out_weekly data/processed/first_down_success_weekly_2024.csv \
  --out_team data/processed/first_down_success_team_2024.csv

Notes
-----
- The script is defensive to column presence; it will infer reasonable flags
  if some helper columns are missing in pbp_clean.
- If a precomputed success flag for 1st down exists (e.g. `success` or
  `success_play`), we ignore it to keep the 40/60/100 definition consistent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


EXCLUDE_PLAY_TYPES = {
    "no_play",  # penalty-only rows in nflfastR style
    "timeout",
    "qb_spike",
    "qb_kneel",
    "extra_point",
    "field_goal",
    "kickoff",
    "punt",
}


def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a boolean Series for flag column if present else False."""
    if name in df.columns:
        s = df[name]
        if s.dtype != bool:
            # Treat 1/0 or 'True'/'False'
            return s.fillna(0).astype(int).astype(bool)
        return s.fillna(False)
    return pd.Series(False, index=df.index)


def load_and_filter_pbp(path: str | Path, season: int) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # --- Alias/normalize common column name variants ---
    alias_map = {
        "yards_to_go": ["ydstogo", "yds_to_go", "distance"],
        "posteam": ["pos_team", "posteam"],
        "defteam": ["def_team", "defteam"],
        "play_type": ["play_type", "playtype"],
        "yards_gained": ["yards_gained", "yds_gained", "gain"],
        "week": ["week", "game_week"],
        "season": ["season", "year"],
        "down": ["down"],
        "epa": ["epa"],
        "yardline_100": ["yardline_100", "yrdln100", "yardline"],
    }

    for target, candidates in alias_map.items():
        if target not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[target] = df[c]
                    break

    # Basic filters: season and 1st down scrimmage plays.
    if "season" in df.columns:
        df = df[df["season"].astype(int) == int(season)]

    # Down == 1 only
    df = df[df.get("down").fillna(0).astype(int) == 1]

    # Exclude obvious non-scrimmage types
    play_type = df.get("play_type").astype(str).str.lower()
    EXCLUDE = EXCLUDE_PLAY_TYPES
    # Some datasets label penalties as 'penalty' or 'no_play'; exclude both
    EXCLUDE = EXCLUDE.union({"penalty"})
    df = df[~play_type.isin(EXCLUDE)]

    # Exclude two-point attempts
    two_pt = _bool_col(df, "two_point_attempt")
    df = df[~two_pt]

    # Exclude kneels/spikes if encoded via flags
    df = df[~_bool_col(df, "qb_kneel")]
    df = df[~_bool_col(df, "qb_spike")]

    # Minimal required columns
    required = ["yardline_100", "yards_gained", "yards_to_go", "posteam", "defteam", "epa"]
    for col in required:
        if col not in df.columns:
            if col == "yardline_100":
                df[col] = pd.NA
            elif col == "epa":
                df[col] = 0.0
            else:
                raise ValueError(f"Missing required column in PBP: {col}")

    # Keep only rows with non-null yards_to_go (protect against weird rows)
    df = df[df["yards_to_go"].notna()]

    return df


def add_success_flag(df: pd.DataFrame) -> pd.DataFrame:
    ytg = df["yards_to_go"].astype(float).clip(lower=0)
    yg = df["yards_gained"].astype(float).fillna(0)
    success_1st = yg >= 0.4 * ytg
    out = df.copy()
    out["success_1st"] = success_1st.astype(int)
    return out


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure week/season present
    if "week" not in df.columns:
        raise ValueError("PBP is missing 'week' column; required for weekly output.")
    if "season" not in df.columns:
        raise ValueError("PBP is missing 'season' column; required for outputs.")

    base_cols = ["season", "week"]

    # OFFENSE aggregation
    off = (
        df.groupby(base_cols + ["posteam"], dropna=False)
        .agg(plays=("success_1st", "size"), successes=("success_1st", "sum"), epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off["side"] = "off"
    off["sr"] = (off["successes"] / off["plays"]).round(6)

    # DEFENSE aggregation (what teams allowed on opponent 1st downs)
    deff = (
        df.groupby(base_cols + ["defteam"], dropna=False)
        .agg(plays=("success_1st", "size"), successes=("success_1st", "sum"), epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    deff["side"] = "def"
    deff["sr"] = (deff["successes"] / deff["plays"]).round(6)

    out = pd.concat([off, deff], ignore_index=True)
    # Order columns
    out = out[["season", "week", "team", "side", "plays", "successes", "sr", "epa_per_play"]]
    return out.sort_values(["season", "week", "team", "side"]).reset_index(drop=True)


def aggregate_team(df: pd.DataFrame) -> pd.DataFrame:
    # Season-level per team & side
    if "season" not in df.columns:
        raise ValueError("PBP is missing 'season' column; required for outputs.")

    base_cols = ["season"]

    off = (
        df.groupby(base_cols + ["posteam"], dropna=False)
        .agg(plays=("success_1st", "size"), successes=("success_1st", "sum"), epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off["side"] = "off"
    off["sr"] = (off["successes"] / off["plays"]).round(6)

    deff = (
        df.groupby(base_cols + ["defteam"], dropna=False)
        .agg(plays=("success_1st", "size"), successes=("success_1st", "sum"), epa_per_play=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    deff["side"] = "def"
    deff["sr"] = (deff["successes"] / deff["plays"]).round(6)

    out = pd.concat([off, deff], ignore_index=True)
    out = out[["season", "team", "side", "plays", "successes", "sr", "epa_per_play"]]
    return out.sort_values(["season", "team", "side"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Build 1st Down Success (O/D) from PBP.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=str, required=True)
    ap.add_argument("--out_weekly", type=str, required=True)
    ap.add_argument("--out_team", type=str, required=True)
    args = ap.parse_args()

    print(f"📥 Wczytuję PBP: {args.in_pbp}")
    try:
        df = load_and_filter_pbp(args.in_pbp, args.season)
    except Exception as e:
        print(f"❌ Błąd wczytywania/filtra PBP: {e}")
        sys.exit(2)

    df = add_success_flag(df)

    # WEEKLY
    weekly = aggregate_weekly(df)
    Path(args.out_weekly).parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out_weekly, index=False)
    print(
        f"✅ Zapisano weekly: {args.out_weekly} (rows={len(weekly)})\n"
        f"   sample:\n{weekly.head(6).to_string(index=False)}"
    )

    # TEAM
    team = aggregate_team(df)
    Path(args.out_team).parent.mkdir(parents=True, exist_ok=True)
    team.to_csv(args.out_team, index=False)
    print(
        f"🏁 Zapisano team:   {args.out_team} (rows={len(team)})\n"
        f"   sample:\n{team.head(6).to_string(index=False)}"
    )

    # Quick leaderboard preview (top/bottom by offensive sr)
    try:
        off_lead = (
            team[team["side"] == "off"]
            .sort_values("sr", ascending=False)
            .head(5)
            [["team", "sr", "epa_per_play", "plays"]]
        )
        def_tail = (
            team[team["side"] == "def"]
            .sort_values("sr", ascending=True)
            .head(5)
            [["team", "sr", "epa_per_play", "plays"]]
        )
        print("\n🏆 OFF – Top 1st-Down SR (so far):")
        print(off_lead.to_string(index=False))
        print("\n🛡️  DEF – Best at preventing 1st-Down success (lowest allowed SR):")
        print(def_tail.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
