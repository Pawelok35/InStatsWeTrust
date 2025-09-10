"""
ETL • Transform step for NFL play-by-play (PBP) data
Project: In Stats We Trust – Main
Day 4 (Sprint 1): Clean & standardize PBP for metric computation.

Usage:
    python etl/transform.py data/pbp_sample.parquet data/pbp_clean.parquet
    # or CSV
    python etl/transform.py data/pbp_sample.csv data/pbp_clean.parquet

Notes:
- Input is the raw dataframe produced by etl/ingest.py (nfl_data_py / nflfastR schema).
- Output is a slim, standardized dataframe ready for metric layers (Core 12, etc.).
- The script is conservative: it keeps rows even if EPA is NaN, but tags them.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ---- Configuration ---------------------------------------------------------
# Minimal set of columns we expect from nflfastR/nfl_data_py. The code guards
# against missing columns and fills them if absent.
REQUIRED_COLS: List[str] = [
    # identifiers
    "game_id", "play_id", "season", "week",
    # teams & context
    "posteam", "defteam", "home_team", "away_team", "game_date",
    # play descriptors
    "play_type", "down", "ydstogo", "yardline_100", "yards_gained",
    "qtr", "time", "desc",
    # advanced
    "epa", "success", "air_yards", "qb_dropback", "pass", "rush",
    # results / flags
    "touchdown", "pass_touchdown", "rush_touchdown",
    "interception", "fumble",
    # penalties & special teams (often used to filter)
    "penalty", "kickoff", "punt", "field_goal_attempt",
]

# Columns we keep in the slim table. Order matters for readability.
OUTPUT_COLS: List[str] = [
    "game_id", "play_id", "season", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam",
    "qtr", "time", "down", "ydstogo", "yardline_100",
    "play_type", "yards_gained", "epa", "success",
    "is_pass", "is_rush", "is_dropback",
    "air_yards", "touchdown", "pass_touchdown", "rush_touchdown",
    "interception", "fumble", "penalty", "st_play",
    "desc",
]


# ---- Utilities -------------------------------------------------------------
def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Ensure all columns exist in df; if missing, create with NaN/False.
    Returns df (view) with all requested columns present.
    """
    for c in cols:
        if c not in df.columns:
            if c in {"pass", "rush", "qb_dropback", "touchdown",
                     "pass_touchdown", "rush_touchdown", "interception",
                     "fumble", "penalty", "kickoff", "punt",
                     "field_goal_attempt", "success"}:
                df[c] = np.nan  # will be coerced later
            else:
                df[c] = np.nan
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types to reasonable dtypes and normalize booleans (0/1)."""
    boolish = [
        "pass", "rush", "qb_dropback", "touchdown", "pass_touchdown",
        "rush_touchdown", "interception", "fumble", "penalty",
        "kickoff", "punt", "field_goal_attempt",
    ]
    for c in boolish:
        if c in df.columns:
            df[c] = df[c].astype("float").fillna(0.0).astype("int8")

    # numeric
    num_cols = [
        "down", "ydstogo", "yardline_100", "yards_gained", "air_yards",
        "epa", "qtr", "week", "play_id", "season",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # strings
    str_cols = [
        "game_id", "posteam", "defteam", "home_team", "away_team",
        "play_type", "time", "desc",
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    return df


def _derive_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add convenient boolean flags and special-teams indicator."""
    # In some schemas, column names vary; keep defensive.
    if "pass" in df.columns:
        df["is_pass"] = (df["pass"] == 1).astype("int8")
    else:
        df["is_pass"] = (df.get("play_type", "").fillna("")
                          .str.lower().str.contains("pass")).astype("int8")

    if "rush" in df.columns:
        df["is_rush"] = (df["rush"] == 1).astype("int8")
    else:
        df["is_rush"] = (df.get("play_type", "").fillna("")
                          .str.lower().str.contains("run|rush")).astype("int8")

    if "qb_dropback" in df.columns:
        df["is_dropback"] = (df["qb_dropback"] == 1).astype("int8")
    else:
        df["is_dropback"] = 0

    st_cols = ["kickoff", "punt", "field_goal_attempt"]
    st_present = [c for c in st_cols if c in df.columns]
    if st_present:
        df["st_play"] = (df[st_present].sum(axis=1) > 0).astype("int8")
    else:
        df["st_play"] = 0

    return df


def _compute_success_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill/compute `success` if absent or all NA using classic success rules.

    Rule of thumb (success if yards gained >= threshold):
      - 1st down: >= 50% of ydstogo
      - 2nd down: >= 70% of ydstogo
      - 3rd/4th down: >= 100% of ydstogo
    Penalized/No-play rows are marked unsuccessful.
    """
    need = False
    if "success" not in df.columns:
        df["success"] = np.nan
        need = True
    if df["success"].isna().all():
        need = True

    if not need:
        # normalize to 0/1
        df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype("int8")
        return df

    # Compute
    gains = pd.to_numeric(df.get("yards_gained"), errors="coerce")
    togo = pd.to_numeric(df.get("ydstogo"), errors="coerce")
    down = pd.to_numeric(df.get("down"), errors="coerce").fillna(0).astype("Int64")

    thr = (
        (down == 1) * (0.5 * togo) +
        (down == 2) * (0.7 * togo) +
        (down >= 3) * (1.0 * togo)
    )

    base_success = (gains >= thr).astype("int8")

    # Penalized or special teams → mark 0
    penalty_flag = pd.to_numeric(df.get("penalty"), errors="coerce").fillna(0).astype(int)
    st_flag = pd.to_numeric(df.get("st_play"), errors="coerce").fillna(0).astype(int)
    df["success"] = (base_success & (penalty_flag == 0) & (st_flag == 0)).astype("int8")
    return df


def _select_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order output columns; add any missing ones as NA/0."""
    for c in OUTPUT_COLS:
        if c not in df.columns:
            if c in {"is_pass", "is_rush", "is_dropback", "st_play", "success"}:
                df[c] = 0
            else:
                df[c] = pd.NA
    return df[OUTPUT_COLS].copy()


def transform_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """Main transform function.

    Steps:
    1) Ensure required columns exist
    2) Coerce types
    3) Derive flags (is_pass, is_rush, etc.)
    4) Compute success if needed
    5) Select/Order output columns
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("transform_pbp expects a pandas DataFrame")

    df = df.copy()
    df = _ensure_columns(df, REQUIRED_COLS)
    df = _coerce_types(df)
    df = _derive_flags(df)
    df = _compute_success_if_missing(df)

    # Optional data hygiene: drop obviously invalid rows
    # e.g., missing teams or down outside 1–4, but keep conservative by default
    # df = df[df["posteam"].notna() & df["defteam"].notna()]

    out = _select_output(df)

    # Tag rows where EPA is NaN (upstream may exclude some plays)
    out["epa_is_nan"] = out["epa"].isna().astype("int8")

    return out


# ---- IO Helpers ------------------------------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


# ---- CLI -------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("Usage: python etl/transform.py <input.(csv|parquet)> <output.(parquet|csv)>")
        return 2

    in_path = Path(argv[0])
    out_path = Path(argv[1])

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    print(f"[transform] Reading: {in_path}")
    df_raw = _read_any(in_path)
    print(f"[transform] Rows in: {len(df_raw):,}")

    df_clean = transform_pbp(df_raw)

    # Basic validation
    assert set(["game_id", "play_id"]).issubset(df_clean.columns), "Missing key IDs in output"

    print(f"[transform] Rows out: {len(df_clean):,}")
    print(f"[transform] Null EPA rows: {int(df_clean['epa_is_nan'].sum()):,}")

    _write_any(df_clean, out_path)
    print(f"[transform] Saved → {out_path}")

    # Emit a tiny summary JSON next to the output (helps in tests/CI)
    summary = {
        "rows_in": int(len(df_raw)),
        "rows_out": int(len(df_clean)),
        "null_epa": int(df_clean["epa_is_nan"].sum()),
        "columns": list(df_clean.columns),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[transform] Summary → {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
