"""
Season-level EPA & Core 12 metrics aggregation
Project: In Stats We Trust – Main

CLI:
    python etl/transform_epa_season.py <input_clean_pbp.(parquet|csv)> <out_dir>

We assume input is the CLEAN play-by-play from etl/transform.py with columns like:
- game_id, play_id, season, week, game_date
- posteam, defteam, home_team, away_team
- down, ydstogo, yardline_100, yards_gained, epa, success
- is_pass, is_rush, is_dropback, touchdown, interception, fumble, st_play

This script computes season-level team summaries for offense/defense and writes:
- epa_offense_summary_<season>_season.csv
- epa_defense_summary_<season>_season.csv
- power_signal_<season>_season.csv (simple signal = off_epa/play − def_epa_allowed/play)
- matchups_<season>_season.csv (unique season matchups extracted from PBP)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------ Helpers ------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        # more robust CSV read
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input: {path.suffix}")


def _save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    df.to_csv(p, index=False)
    return p


def _ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _explosive_mask(df: pd.DataFrame) -> pd.Series:
    yd = pd.to_numeric(df.get("yards_gained"), errors="coerce").fillna(0)
    is_pass = pd.to_numeric(df.get("is_pass"), errors="coerce").fillna(0).astype(int)
    is_rush = pd.to_numeric(df.get("is_rush"), errors="coerce").fillna(0).astype(int)
    return ((is_pass == 1) & (yd >= 20)) | ((is_rush == 1) & (yd >= 10))


def _group_offense(df: pd.DataFrame) -> pd.DataFrame:
    # Filter: exclude special teams for offensive profile (optional, safer)
    st = pd.to_numeric(df.get("st_play"), errors="coerce").fillna(0).astype(int)
    off = df.loc[st == 0].copy()

    # Masks
    epa = pd.to_numeric(off.get("epa"), errors="coerce")
    success = pd.to_numeric(off.get("success"), errors="coerce")
    down = pd.to_numeric(off.get("down"), errors="coerce")
    y100 = pd.to_numeric(off.get("yardline_100"), errors="coerce")

    is_pass = pd.to_numeric(off.get("is_pass"), errors="coerce").fillna(0).astype(int)
    is_rush = pd.to_numeric(off.get("is_rush"), errors="coerce").fillna(0).astype(int)
    inter = pd.to_numeric(off.get("interception"), errors="coerce").fillna(0).astype(int)
    fumble = pd.to_numeric(off.get("fumble"), errors="coerce").fillna(0).astype(int)

    explosive = _explosive_mask(off)

    grp = off.groupby(["season", "posteam"], dropna=False)

    def agg_fn(g: pd.DataFrame) -> pd.Series:
        e = pd.to_numeric(g["epa"], errors="coerce")
        s = pd.to_numeric(g["success"], errors="coerce")
        d = pd.to_numeric(g["down"], errors="coerce")
        y = pd.to_numeric(g["yardline_100"], errors="coerce")
        ip = pd.to_numeric(g.get("is_pass"), errors="coerce").fillna(0).astype(int)
        ir = pd.to_numeric(g.get("is_rush"), errors="coerce").fillna(0).astype(int)
        inter = pd.to_numeric(g.get("interception"), errors="coerce").fillna(0).astype(int)
        fmb = pd.to_numeric(g.get("fumble"), errors="coerce").fillna(0).astype(int)
        yards = pd.to_numeric(g.get("yards_gained"), errors="coerce").fillna(0)
        expl = _explosive_mask(g)

        out = {
            "plays": int(len(g)),
            "avg_epa": float(e.mean(skipna=True)),
            "median_epa": float(e.median(skipna=True)),
            "success_rate": float(s.mean(skipna=True)),
            "explosive_plays": int(expl.sum()),
            "explosive_rate": float(expl.mean(skipna=True)),
            "total_yards": float(yards.sum()),
            # Core 12 pieces
            "early_down_epa": float(e[d.isin([1, 2])].mean(skipna=True)),
            "late_down_epa": float(e[d.isin([3, 4])].mean(skipna=True)),
            "third_down_sr": float(s[d == 3].mean(skipna=True)),
            "fourth_down_sr": float(s[d == 4].mean(skipna=True)),
            "red_zone_epa": float(e[y <= 20].mean(skipna=True)),
            "pass_epa_per_play": float(e[ip == 1].mean(skipna=True)),
            "rush_epa_per_play": float(e[ir == 1].mean(skipna=True)),
            # Turnover EPA (sum EPA on plays flagged as turnover proxies)
            "turnover_epa": float(e[(inter == 1) | (fmb == 1)].sum(skipna=True)),
            # Field position (lower is better; offense closer to EZ)
            "avg_start_yardline_100": float(y.mean(skipna=True)),
        }
        return pd.Series(out)

    out = grp.apply(agg_fn).reset_index().rename(columns={"posteam": "team"})
    return out


def _group_defense(df: pd.DataFrame) -> pd.DataFrame:
    st = pd.to_numeric(df.get("st_play"), errors="coerce").fillna(0).astype(int)
    deff = df.loc[st == 0].copy()

    grp = deff.groupby(["season", "defteam"], dropna=False)

    def agg_fn(g: pd.DataFrame) -> pd.Series:
        e = pd.to_numeric(g["epa"], errors="coerce")
        s = pd.to_numeric(g["success"], errors="coerce")
        d = pd.to_numeric(g["down"], errors="coerce")
        y = pd.to_numeric(g["yardline_100"], errors="coerce")
        ip = pd.to_numeric(g.get("is_pass"), errors="coerce").fillna(0).astype(int)
        ir = pd.to_numeric(g.get("is_rush"), errors="coerce").fillna(0).astype(int)
        inter = pd.to_numeric(g.get("interception"), errors="coerce").fillna(0).astype(int)
        fmb = pd.to_numeric(g.get("fumble"), errors="coerce").fillna(0).astype(int)
        yards = pd.to_numeric(g.get("yards_gained"), errors="coerce").fillna(0)
        expl = _explosive_mask(g)

        out = {
            "plays_allowed": int(len(g)),
            "avg_epa_allowed": float(e.mean(skipna=True)),
            "median_epa_allowed": float(e.median(skipna=True)),
            "success_rate_allowed": float(s.mean(skipna=True)),
            "explosive_allowed": int(expl.sum()),
            "explosive_rate_allowed": float(expl.mean(skipna=True)),
            "yards_allowed": float(yards.sum()),
            # Core 12 (allowed)
            "early_down_epa_allowed": float(e[d.isin([1, 2])].mean(skipna=True)),
            "late_down_epa_allowed": float(e[d.isin([3, 4])].mean(skipna=True)),
            "third_down_sr_allowed": float(s[d == 3].mean(skipna=True)),
            "fourth_down_sr_allowed": float(s[d == 4].mean(skipna=True)),
            "red_zone_epa_allowed": float(e[y <= 20].mean(skipna=True)),
            "pass_epa_per_play_allowed": float(e[ip == 1].mean(skipna=True)),
            "rush_epa_per_play_allowed": float(e[ir == 1].mean(skipna=True)),
            # Turnovers forced (EPA from opponent turnovers — negative for OFF, positive benefit DEF)
            "turnover_epa_forced": float(e[(inter == 1) | (fmb == 1)].sum(skipna=True)),
            # Field position faced (higher is worse for DEF because OFF farther from EZ)
            "avg_start_yardline_100_faced": float(y.mean(skipna=True)),
        }
        return pd.Series(out)

    out = grp.apply(agg_fn).reset_index().rename(columns={"defteam": "team"})
    return out


def _build_matchups(df: pd.DataFrame) -> pd.DataFrame:
    games = df[["game_id", "home_team", "away_team", "season"]].drop_duplicates()
    games = games.sort_values(["season", "home_team", "away_team", "game_id"])  # stable
    return games


# ------------------------------ Main ---------------------------------------

def run(input_path: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _read_any(input_path)
    need = [
        "season", "posteam", "defteam", "home_team", "away_team",
        "down", "ydstogo", "yardline_100", "yards_gained", "epa",
        "success", "is_pass", "is_rush", "interception", "fumble", "st_play",
    ]
    df = _ensure_cols(df, need)

    # Coerce crucial types once
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    # OFF & DEF tables
    off = _group_offense(df)
    deff = _group_defense(df)

    # Power signal (simple version)
    join = off.merge(deff[["season", "team", "avg_epa_allowed"]], on=["season", "team"], how="left")
    join["power_signal"] = join["avg_epa"] - join["avg_epa_allowed"]
    power = join[["season", "team", "avg_epa", "avg_epa_allowed", "power_signal"]].sort_values(
        ["season", "power_signal"], ascending=[True, False]
    )

    # Matchups
    matchups = _build_matchups(df)

    # Persist (derive season for filenames)
    seasons = sorted(off["season"].dropna().unique().tolist())
    if len(seasons) == 1:
        season = int(seasons[0])
        off_p = _save_csv(off, out_dir, f"epa_offense_summary_{season}_season.csv")
        deff_p = _save_csv(deff, out_dir, f"epa_defense_summary_{season}_season.csv")
        pow_p = _save_csv(power, out_dir, f"power_signal_{season}_season.csv")
        mat_p = _save_csv(matchups, out_dir, f"matchups_{season}_season.csv")
        print(f"📥 Wczytuję PBP scalone dla {season}…")
        print(f"✅ Zapisano: {off_p}")
        print(f"✅ Zapisano: {deff_p}")
        print(f"✅ Zapisano: {mat_p}")
        print(f"🏆 Zapisano ranking: {pow_p}")
    else:
        # Multi-season safety: write generic filenames
        off_p = _save_csv(off, out_dir, "epa_offense_summary_seasons.csv")
        deff_p = _save_csv(deff, out_dir, "epa_defense_summary_seasons.csv")
        pow_p = _save_csv(power, out_dir, "power_signal_seasons.csv")
        mat_p = _save_csv(matchups, out_dir, "matchups_seasons.csv")
        print("📥 Wczytano wiele sezonów…")
        print(f"✅ Zapisano: {off_p}")
        print(f"✅ Zapisano: {deff_p}")
        print(f"✅ Zapisano: {mat_p}")
        print(f"🏆 Zapisano ranking: {pow_p}")

    return off, deff, power


def _main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("Usage: python etl/transform_epa_season.py <input_clean_pbp.(parquet|csv)> <out_dir>")
        return 2

    in_path = Path(argv[0])
    out_dir = Path(argv[1])
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    run(in_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
