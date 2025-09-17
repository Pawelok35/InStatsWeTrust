# build_penalty_discipline.py
# ---------------------------------
# ETL: Penalty Discipline (OFF/DEF) — weekly & team aggregates
# Usage (example):
#   python etl/build_penalty_discipline.py \
#       --season 2024 \
#       --in_pbp data/processed/pbp_clean_2024.parquet \
#       --out_weekly data/processed/penalty_discipline_weekly_2024.csv \
#       --out_team   data/processed/penalty_discipline_team_2024.csv
#
# Output columns:
#   weekly: game_id, week, team, side(off/def), plays, penalties, penalty_yds,
#           pen_per_100_plays, presnap_pen_rate, rz_pen_rate,
#           third_fourth_pen, auto_fd_allowed, dpi_yds
#   team:   team, side, games, plays, penalties, penalties_pg,
#           penalty_yds, penalty_yds_pg, pen_per_100_plays,
#           presnap_pen_rate, rz_pen_rate,
#           third_fourth_pen, third_fourth_pen_pg,
#           auto_fd_allowed, def_auto_fd_allowed_pg,
#           dpi_yds, def_dpi_yds_pg

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

def _safe_div(a: pd.Series | float | int, b: pd.Series | float | int) -> pd.Series:
    """Safe division that returns 0 where denominator==0."""
    return (a / b).where(b != 0, 0)


# Basic parser for penalty description lines.
# It attempts to infer:
#  - penalty type string
#  - yardage assessed (int)
#  - whether it's automatic first down
#  - whether it's a pre-snap foul
#  - whether it's Defensive Pass Interference (DPI)
_PRESNAP_PAT = re.compile(
    r"(false\s*start|offside|neutral\s*zone\s*infraction|encroachment|delay\s*of\s*game|"
    r"illegal\s*shift|illegal\s*motion|illegal\s*formation)",
    re.IGNORECASE,
)
_DPI_PAT = re.compile(r"(defensive\s*pass\s*interference|DPI\b)", re.IGNORECASE)
_AUTO_FD_PAT = re.compile(r"automatic\s+first\s+down", re.IGNORECASE)
_YDS_PAT = re.compile(r"(\d+)\s*yard(s)?", re.IGNORECASE)

def parse_penalty_from_desc(desc: str) -> Tuple[str, int, int, int, int]:
    """
    Returns: (penalty_type, yards, auto_first_down (0/1), is_presnap (0/1), is_dpi (0/1))
    """
    if not isinstance(desc, str):
        desc = ""

    # crude type: take first phrase until '(' or ' - ' or ' for '
    ptype = ""
    # Try to capture well-known tokens
    if _DPI_PAT.search(desc):
        ptype = "Defensive Pass Interference"
    elif _PRESNAP_PAT.search(desc):
        # normalize a bit
        m = _PRESNAP_PAT.search(desc)
        ptype = m.group(1).strip().title() if m else "Pre-snap Foul"
    else:
        # fallback: take the part before " for " / "(" / " - "
        cut = re.split(r"\s*(?:\(|\s-\s| for )", desc, maxsplit=1)
        ptype = cut[0].strip().title() if cut else ""

    yds = 0
    m_yds = _YDS_PAT.search(desc)
    if m_yds:
        try:
            yds = int(m_yds.group(1))
        except ValueError:
            yds = 0

    auto_fd = 1 if _AUTO_FD_PAT.search(desc) else 0
    is_presnap = 1 if _PRESNAP_PAT.search(desc) else 0
    is_dpi = 1 if _DPI_PAT.search(desc) else 0

    return ptype, yds, auto_fd, is_presnap, is_dpi


# ---------------------------
# Core builder
# ---------------------------

def build(df: pd.DataFrame, season: int):
    # Filtr sezonu
    if "season" in df.columns:
        df = df.loc[df["season"] == season].copy()

    # Filtr tylko REG (jeśli nie ma season_type → fallback na week 1–18)
    if "season_type" in df.columns:
        df = df.loc[df["season_type"] == "REG"].copy()
    else:
        if "week" in df.columns:
            df = df.loc[df["week"].between(1, 18)].copy()


# Najpierw usuń wiersze z brakiem drużyny (prawdziwe NaN, jeszcze przed astype(str))
    df = df[df.get("posteam").notna() & df.get("defteam").notna()].copy()

    # Ujednolicenie kodów drużyn
    TEAM_FIX = {
        "LA": "LAR",   # Rams
        "JAC": "JAX",  # Jaguars
        "WAS": "WSH",  # Commanders
        "OAK": "LV",   # Raiders (historycznie)
        "SD": "LAC",   # Chargers (historycznie)
        "STL": "LAR",  # Rams (historycznie)
    }
    for col in ("posteam", "defteam"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map(lambda x: TEAM_FIX.get(x, x))

    # Usuń sztuczny team 'NONE' (po mapowaniu)
    df = df[(df["posteam"] != "NONE") & (df["defteam"] != "NONE")].copy()




    # Normalizacje bazowe
    if "penalty" not in df.columns:
        df["penalty"] = 0
    df["penalty"] = df["penalty"].fillna(0).astype(int)

    df["desc"] = df.get("desc", pd.Series("", index=df.index)).fillna("").astype(str)
    df["down"] = df.get("down").fillna(0).astype(int)
    df["yardline_100"] = df.get("yardline_100").fillna(100).astype(float)

    # Utwórz brakujące kolumny
    if "penalty_type" not in df.columns:
        df["penalty_type"] = ""
    if "penalty_yards" not in df.columns:
        df["penalty_yards"] = 0
    if "automatic_first_down" not in df.columns:
        df["automatic_first_down"] = 0

    # Parsowanie kar z desc
    ptype_list, yards_list, auto_fd_list, presnap_list, dpi_list = [], [], [], [], []
    for is_pen, d in zip(df["penalty"].astype(int), df["desc"]):
        if is_pen == 1:
            ptype, yards, auto_fd, presnap, dpi = parse_penalty_from_desc(d)
        else:
            ptype, yards, auto_fd, presnap, dpi = "", 0, 0, 0, 0
        ptype_list.append(ptype)
        yards_list.append(yards)
        auto_fd_list.append(auto_fd)
        presnap_list.append(presnap)
        dpi_list.append(dpi)

    # Uzupełnianie braków z parsingu
    ptype_series = pd.Series(ptype_list, index=df.index)
    df["penalty_type"] = (
        df["penalty_type"].fillna("").astype(str).str.strip()
        .mask(lambda s: s.eq(""), ptype_series)
    )
    df["penalty_type"] = df["penalty_type"].fillna("").astype(str)

    yards_series = pd.Series(yards_list, index=df.index)
    df["penalty_yards"] = (
        pd.to_numeric(df["penalty_yards"], errors="coerce").fillna(0).astype(int)
        .mask(lambda s: (s == 0) & (yards_series != 0), yards_series)
    )

    auto_fd_series = pd.Series(auto_fd_list, index=df.index).astype(int)
    if "first_down_penalty" in df.columns:
        df["automatic_first_down"] = (
            pd.to_numeric(df["automatic_first_down"], errors="coerce").fillna(0).astype(int)
            .mask(lambda s: (s == 0) & (df["first_down_penalty"].fillna(0).astype(int) == 1), 1)
            .mask(lambda s: (s == 0) & (auto_fd_series == 1), 1)
        )
    else:
        df["automatic_first_down"] = (
            pd.to_numeric(df["automatic_first_down"], errors="coerce").fillna(0).astype(int)
            .mask(lambda s: (s == 0) & (auto_fd_series == 1), 1)
        )

    # Flagi kontekstowe
    df["is_presnap"] = pd.Series(presnap_list, index=df.index).astype(int)
    df["is_dpi"] = pd.Series(dpi_list, index=df.index).astype(int)
    df["is_rz"] = (df["yardline_100"] <= 20).astype(int)
    df["is_3rd4th"] = df["down"].isin([3, 4]).astype(int)

    # Dla OFF/DEF zliczamy na bazie posteam/defteam
    base_cols = ["game_id", "week"]

    def _agg_side(side: str, team_col: str, plays_group_col: str):
        snaps = (
            df.groupby(base_cols + [plays_group_col], dropna=False)
              .size()
              .rename("plays")
              .reset_index()
              .rename(columns={plays_group_col: "team"})
        )

        pen = (
            df[df["penalty"] == 1]
            .groupby(base_cols + [team_col], dropna=False)
            .agg(
                penalties=("penalty", "sum"),
                penalty_yds=("penalty_yards", "sum"),
                presnap_pen=("is_presnap", "sum"),
                rz_pen=("is_rz", "sum"),
                third_fourth_pen=("is_3rd4th", "sum"),
                auto_fd_allowed=("automatic_first_down", "sum"),
                dpi_yds=("penalty_yards", lambda s: s[df.loc[s.index, "is_dpi"] == 1].sum()),
            )
            .reset_index()
            .rename(columns={team_col: "team"})
        )

        out = snaps.merge(pen, on=base_cols + ["team"], how="left")
        num_cols = ["penalties","penalty_yds","presnap_pen","rz_pen",
                    "third_fourth_pen","auto_fd_allowed","dpi_yds"]
        for c in num_cols:
            if c not in out.columns:
                out[c] = 0
        out[num_cols] = out[num_cols].fillna(0)

        out["pen_per_100_plays"] = _safe_div(out["penalties"] * 100, out["plays"])
        out["presnap_pen_rate"] = _safe_div(out["presnap_pen"], out["plays"])
        out["rz_pen_rate"] = _safe_div(out["rz_pen"], out["plays"])
        out["side"] = side

        keep = [
            "game_id","week","team","side",
            "plays","penalties","penalty_yds",
            "pen_per_100_plays","presnap_pen_rate","rz_pen_rate",
            "third_fourth_pen","auto_fd_allowed","dpi_yds"
        ]
        return out[keep]

    off = _agg_side("off", "posteam", "posteam")
    deff = _agg_side("def", "defteam", "defteam")
    weekly = pd.concat([off, deff], ignore_index=True)

    # Agregat drużynowy
    team = (
        weekly.groupby(["team","side"], dropna=False)
              .agg(
                  plays=("plays","sum"),
                  penalties=("penalties","sum"),
                  penalty_yds=("penalty_yds","sum"),
                  presnap_pen_rate=("presnap_pen_rate","mean"),   # średnia tygodniowa
                  rz_pen_rate=("rz_pen_rate","mean"),
                  pen_per_100_plays=("pen_per_100_plays","mean"),
                  third_fourth_pen=("third_fourth_pen","sum"),
                  auto_fd_allowed=("auto_fd_allowed","sum"),
                  dpi_yds=("dpi_yds","sum"),
                )
                .reset_index()
    )

    # --- WARIANTY WAŻONE LICZBĄ SNAPÓW (weighted by plays) ---
    # policz liczniki ważone po weekly i złącz
    _w = weekly.assign(
        presnap_num = weekly["presnap_pen_rate"] * weekly["plays"],
        rz_num      = weekly["rz_pen_rate"]       * weekly["plays"],
        per100_num  = weekly["pen_per_100_plays"] * weekly["plays"],
    ).groupby(["team","side"], dropna=False).agg(
        plays_w=("plays","sum"),
        presnap_num=("presnap_num","sum"),
        rz_num=("rz_num","sum"),
        per100_num=("per100_num","sum"),
    ).reset_index()

    team = team.merge(_w, on=["team","side"], how="left")
    team["presnap_pen_rate_w"] = _safe_div(team["presnap_num"], team["plays_w"])
    team["rz_pen_rate_w"] = _safe_div(team["rz_num"], team["plays_w"])
    team["pen_per_100_plays_w"] = _safe_div(team["per100_num"], team["plays_w"])


    games = (
        weekly.groupby(["team","side"])["week"].nunique().rename("games").reset_index()
    )
    team = team.merge(games, on=["team","side"], how="left")

    team["penalties_pg"] = _safe_div(team["penalties"], team["games"])
    team["penalty_yds_pg"] = _safe_div(team["penalty_yds"], team["games"])
    team["third_fourth_pen_pg"] = _safe_div(team["third_fourth_pen"], team["games"])
    team["def_auto_fd_allowed_pg"] = _safe_div(team["auto_fd_allowed"], team["games"])
    team["def_dpi_yds_pg"] = _safe_div(team["dpi_yds"], team["games"])

    team = team[[
        "team","side","games","plays",
        "penalties","penalties_pg",
        "penalty_yds","penalty_yds_pg",
        "pen_per_100_plays","pen_per_100_plays_w",
        "presnap_pen_rate","presnap_pen_rate_w",
        "rz_pen_rate","rz_pen_rate_w",
        "third_fourth_pen","third_fourth_pen_pg",
        "auto_fd_allowed","def_auto_fd_allowed_pg",
        "dpi_yds","def_dpi_yds_pg"
    ]]


    return weekly.sort_values(["week","team","side"]), team.sort_values(["team","side"])


# ---------------------------
# CLI
# ---------------------------

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

def main():
    ap = argparse.ArgumentParser(description="Build Penalty Discipline metrics (OFF/DEF).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=Path, required=True)
    ap.add_argument("--out_weekly", type=Path, required=True)
    ap.add_argument("--out_team", type=Path, required=True)
    args = ap.parse_args()

    print(f"📥 Wczytuję PBP: {args.in_pbp}")
    df = _read_any(args.in_pbp)

    weekly, team = build(df, season=args.season)

        # Spodziewana liczba teamów/wierszy
    assert team["team"].nunique() == 32, f"Spodziewano 32 teamy, jest {team['team'].nunique()}"
    assert len(team) == 64, f"Spodziewano 64 wiersze w team, jest {len(team)}"

        # Kolumny weighted (jeśli włączone)
    for col in ["presnap_pen_rate_w","rz_pen_rate_w","pen_per_100_plays_w"]:
        assert col in team.columns, f"Brak kolumny {col} (weighted)"
        s = team[col].fillna(0)
        assert (s >= 0).all(), f"Ujemne wartości w {col}"


    args.out_weekly.parent.mkdir(parents=True, exist_ok=True)
    args.out_team.parent.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(args.out_weekly, index=False)
    team.to_csv(args.out_team, index=False)

    print(f"✅ Zapisano weekly: {args.out_weekly} (rows={len(weekly)})")
    print(f"🏁 Zapisano team:   {args.out_team} (rows={len(team)})")

if __name__ == "__main__":
    main()
