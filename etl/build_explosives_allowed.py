"""
Build Explosive Plays Allowed (Defense) – weekly & team aggregates.

Definition (configurable via CLI):
- Explosive rush: rush for >= R yards (default 10)
- Explosive pass: pass for >= P yards (default 15)
Optionally also track "chunk 20+" (both rush & pass >= 20)

Input:
  Cleaned PBP parquet (consistent with our other ETL steps), one season.

Outputs:
  - weekly CSV:   defteam,week,season,def_plays,expl_rush_allowed,expl_pass_allowed,
                  expl_total_allowed,expl_rate_allowed,chunk20_allowed,chunk20_rate
  - team CSV:     defteam,season,def_plays,expl_rush_allowed,expl_pass_allowed,
                  expl_total_allowed,expl_rate_allowed,chunk20_allowed,chunk20_rate

Usage example:
  python etl/build_explosives_allowed.py ^
    --season 2024 ^
    --in_pbp data/processed/pbp_clean_2024.parquet ^
    --out_weekly data/processed/explosives_allowed_weekly_2024.csv ^
    --out_team data/processed/explosives_allowed_team_2024.csv ^
    --rush_yards 10 ^
    --pass_yards 15 ^
    --track_chunk20
"""

import argparse
import pandas as pd
from pathlib import Path

# --- Helpers -----------------------------------------------------------------

RUSH_PLAY_TYPES = {"run", "rush"}
PASS_PLAY_TYPES = {"pass"}

def is_play_from_pbp(row) -> bool:
    """
    Heurystyka spójna z innymi metrykami w projekcie:
    - bierzemy tylko 'play' gdzie to nie jest karny-only/no_play
    - liczymy rush / pass z realnym 'yards_gained'
    """
    # Jeśli w Twoim pbp_clean masz kolumny 'play', 'no_play', 'penalty', dopasuj poniżej.
    # Zakładamy, że pbp_clean już wycina karne-only i posiada standardowe kolumny:
    # 'play_type', 'yards_gained', 'posteam', 'defteam', 'week', 'season'
    return (
        pd.notna(row.get("play_type"))
        and row.get("play_type") in (RUSH_PLAY_TYPES | PASS_PLAY_TYPES)
        and pd.notna(row.get("yards_gained"))
    )

def build(df: pd.DataFrame, rush_yards: int, pass_yards: int, track_chunk20: bool):
    # Filtr "real plays"
    df = df[df.apply(is_play_from_pbp, axis=1)].copy()

    # Ograniczamy do pól, na których pracujemy
    needed_cols = ["season", "week", "defteam", "play_type", "yards_gained"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in PBP: {missing}")

    # Tag rush/pass
    df["is_rush"] = df["play_type"].isin(RUSH_PLAY_TYPES)
    df["is_pass"] = df["play_type"].isin(PASS_PLAY_TYPES)

    # Explosive allowed (Defense)
    df["expl_rush_allowed"] = (df["is_rush"] & (df["yards_gained"] >= rush_yards)).astype(int)
    df["expl_pass_allowed"] = (df["is_pass"] & (df["yards_gained"] >= pass_yards)).astype(int)
    df["expl_total_allowed"] = df["expl_rush_allowed"] + df["expl_pass_allowed"]

    if track_chunk20:
        df["chunk20_allowed"] = (df["yards_gained"] >= 20).astype(int)
    else:
        df["chunk20_allowed"] = 0

    # Defensive plays (denominator) – bierzemy wszystkie zdefiniowane rush/pass na defteam
    df["def_plays"] = 1

    # --- Weekly agg ---
    grp = df.groupby(["season", "week", "defteam"], as_index=False).agg(
        def_plays=("def_plays", "sum"),
        expl_rush_allowed=("expl_rush_allowed", "sum"),
        expl_pass_allowed=("expl_pass_allowed", "sum"),
        expl_total_allowed=("expl_total_allowed", "sum"),
        chunk20_allowed=("chunk20_allowed", "sum"),
    )
    grp["expl_rate_allowed"] = grp["expl_total_allowed"] / grp["def_plays"].where(grp["def_plays"] != 0, pd.NA)
    grp["chunk20_rate"] = grp["chunk20_allowed"] / grp["def_plays"].where(grp["def_plays"] != 0, pd.NA)

    weekly = grp[[
        "defteam","week","season","def_plays",
        "expl_rush_allowed","expl_pass_allowed","expl_total_allowed",
        "expl_rate_allowed","chunk20_allowed","chunk20_rate"
    ]].sort_values(["season","week","defteam"]).reset_index(drop=True)

    # --- Team (season) agg ---
    team = grp.groupby(["season","defteam"], as_index=False).agg(
        def_plays=("def_plays","sum"),
        expl_rush_allowed=("expl_rush_allowed","sum"),
        expl_pass_allowed=("expl_pass_allowed","sum"),
        expl_total_allowed=("expl_total_allowed","sum"),
        chunk20_allowed=("chunk20_allowed","sum"),
    )
    team["expl_rate_allowed"] = team["expl_total_allowed"] / team["def_plays"].where(team["def_plays"] != 0, pd.NA)
    team["chunk20_rate"] = team["chunk20_allowed"] / team["def_plays"].where(team["def_plays"] != 0, pd.NA)

    team = team[[
        "defteam","season","def_plays",
        "expl_rush_allowed","expl_pass_allowed","expl_total_allowed",
        "expl_rate_allowed","chunk20_allowed","chunk20_rate"
    ]].sort_values(["season","defteam"]).reset_index(drop=True)

    return weekly, team

# --- CLI ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build Explosive Plays Allowed (Defense).")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--in_pbp", type=Path, required=True)
    p.add_argument("--out_weekly", type=Path, required=True)
    p.add_argument("--out_team", type=Path, required=True)
    p.add_argument("--rush_yards", type=int, default=10, help="Threshold for explosive rush (default 10).")
    p.add_argument("--pass_yards", type=int, default=15, help="Threshold for explosive pass (default 15).")
    p.add_argument("--track_chunk20", action="store_true", help="Also compute 20+ chunk plays.")
    args = p.parse_args()

    print(f"📥 Wczytuję PBP: {args.in_pbp}")
    df = pd.read_parquet(args.in_pbp)
    df = df[df["season"] == args.season].copy()

    weekly, team = build(
        df,
        rush_yards=args.rush_yards,
        pass_yards=args.pass_yards,
        track_chunk20=args.track_chunk20,
    )

    # Zapisy
    args.out_weekly.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out_weekly, index=False)
    print(f"✅ Zapisano weekly: {args.out_weekly} (rows={len(weekly):,})")

    args.out_team.parent.mkdir(parents=True, exist_ok=True)
    team.to_csv(args.out_team, index=False)
    print(f"🏁 Zapisano team:   {args.out_team} (rows={len(team):,})")

    # Krótki podgląd
    top = team.sort_values("expl_rate_allowed", ascending=False).head(5)
    print("📊 Top-5 (najgorsze D wg Explosive Rate Allowed):")
    print(top[["defteam","expl_total_allowed","def_plays","expl_rate_allowed"]].to_string(index=False))

if __name__ == "__main__":
    main()
