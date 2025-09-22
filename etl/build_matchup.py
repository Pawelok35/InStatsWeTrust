#!/usr/bin/env python3
"""
Robust head-to-head matchup comparison (season-to-date through week-1 of target).
Fixes:
- Handles files with only 'team' (no 'side') without groupby unpack errors
- Accepts common team column aliases (posteam/defteam/club/abbr/Team)
- Creates parent directory even when --out is provided
- Quiets noisy pandas FutureWarnings
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List
import re
import sys


import numpy as np
import pandas as pd

ID_COLS = {"season", "week", "team", "side", "season_type", "game_id", "opponent", "home_away"}
TEAM_ALIASES = ["team", "posteam", "defteam", "club", "club_code", "abbr", "Team"]

RATE_PAT = re.compile(r"(_rate$|_sr$|_pct$|_per_\d+$|success|_epa$|epa_|_prob$|_ratio$)", re.I)
COUNT_PAT = re.compile(r"(_count$|_tds?$|_fgs?$|_ints?$|_sacks?$|_penalties$|_takeaways$|_attempts$|_plays$)", re.I)
AVG_PAT = re.compile(r"(_avg$|_mean$|avg_|mean_|_yds$|yardline|distance|time_sec|seconds)", re.I)

METRIC_OVERRIDES: Dict[str, Dict[str, str]] = {
    # 3rd Down Efficiency
    "third_down_weekly_2024.csv": {
        "conv_rate": "weighted_mean",
        "plays": "sum",
        "conv": "sum",
    },

    # Red Zone Efficiency
    "redzone_weekly_2024.csv": {
        "rz_td_rate": "weighted_mean",
        "rz_drives": "sum",
        "td_drives": "sum",
        "fg_drives": "sum",
        "to_drives": "sum",
    },

    # 1st Down Success
    "first_down_success_weekly_2024.csv": {
        "sr": "weighted_mean",
        "plays": "sum",
        "success": "sum",
    },

    # Penalty Discipline
    "penalty_discipline_weekly_2024.csv": {
        "pen_per_100_plays_w": "weighted_mean",
        "penalties": "sum",
        "plays": "sum",
        "penalty_yds_pg": "weighted_mean",
    },

    # Drive Efficiency
    "drive_efficiency_weekly_2024.csv": {
         "__weight__": "drives",
        # Podstawowe sumy po drive’ach / akcjach
        "drives": "sum",
        "plays": "sum",
        "plays_h1": "sum",
        "plays_h2": "sum",
        "yds": "sum",
        "yds_h1": "sum",
        "yds_h2": "sum",
        # Typy drive’ów — sumy
        "td_drives": "sum",
        "fg_drives": "sum",
        "rz_drives": "sum",
        "score_drives": "sum",
        "punt_drives": "sum",
        "to_drives": "sum",
        # Punkty / PPD
        "ppd_basic": "weighted_mean",   # średnia ważona per drive
        "ppd_basic_sum": "sum",         # suma punktów/PPD
        # Średnie per drive
        "yds_per_drive": "weighted_mean",
        "time_per_drive": "weighted_mean",
    },

    # Explosives Allowed (Defense)
    "explosives_allowed_weekly_2024.csv": {
        "explosive_rate_allowed": "weighted_mean",
        "explosive_pass_20+": "sum",
        "explosive_rush_10+": "sum",
        "explosive_plays_allowed": "sum",
        "plays": "sum",
    },

    # Second-Half Adjustments
    "second_half_weekly_2024.csv": {
        "adj_epa_all": "weighted_mean",
        "adj_epa_pass": "weighted_mean",
        "adj_epa_rush": "weighted_mean",
        "plays_h1": "sum",
        "plays_h2": "sum",
        "plays": "sum",
    },

    # Points off Turnovers
    "points_off_turnovers_weekly_2024.csv": {
        "takeaways": "sum",
        "points_after_to": "sum",
        "avg_pts_per_takeaway": "weighted_mean",
        "plays": "sum",
    },

    # Special Teams Impact
    "special_teams_weekly_2024.csv": {
        "ko_tb_rate": "weighted_mean",
        "fg_rate": "weighted_mean",
        "xp_rate": "weighted_mean",
        "punt_plays": "sum",
        "plays": "sum",
    },
}


def _list_candidate_files(in_dir: Path) -> List[Path]:
    """
    Zwraca tylko pliki potrzebne do naszych 21 metryk (Core12 + rozszerzenia).
    """
    keep = [
        # Core12
        f"core12_team_{{season}}.csv",            # Off EPA/play, Def EPA/play, SR, Tempo, 4th down, SFP, Hidden Yards, Penalty EPA, QB pressure, Run/Pass block, Rolling Form + SoS

        # Rozszerzenia (Bonus 9)
        f"third_down_weekly_{{season}}.csv",
        f"redzone_weekly_{{season}}.csv",
        f"points_off_turnovers_weekly_{{season}}.csv",
        f"explosives_allowed_weekly_{{season}}.csv",
        f"drive_efficiency_weekly_{{season}}.csv",
        f"second_half_weekly_{{season}}.csv",
        f"first_down_success_weekly_{{season}}.csv",
        f"penalty_discipline_weekly_{{season}}.csv",
        f"special_teams_weekly_{{season}}.csv",
    ]

    files: List[Path] = []
    for pat in keep:
        # podmieniamy {season} na 2024 / 2025
        for year in (2024, 2025):
            p = in_dir / pat.format(season=year)
            if p.exists():
                files.append(p)

    return files


def _weighted_mean(series: pd.Series, weights: pd.Series | None) -> float:
    s = series.astype(float)
    if weights is None or weights.isna().all() or (weights <= 0).all():
        return float(s.mean()) if len(s) else np.nan
    w = weights.astype(float).clip(lower=0)
    try:
        return float(np.average(s, weights=w))
    except ZeroDivisionError:
        return float(s.mean()) if len(s) else np.nan


def _infer_agg(col: str) -> str:
    if RATE_PAT.search(col):
        return "weighted_mean"
    if COUNT_PAT.search(col):
        return "sum"
    if AVG_PAT.search(col):
        return "weighted_mean"
    return "mean"


def _aggregate_weekly(df: pd.DataFrame, overrides: Dict[str, str]) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        return pd.DataFrame()

    agg_plan: Dict[str, str] = {}
    for c in metric_cols:
        agg_plan[c] = overrides.get(c, _infer_agg(c))
        # wybór kolumny wagi: najpierw z overrides["__weight__"], inaczej fallback na "plays"
    weight_col = overrides.get("__weight__")
    if weight_col and weight_col not in df.columns:
        # jeśli podana waga nie istnieje, spadamy na "plays" (jeśli jest)
        weight_col = "plays" if "plays" in df.columns else None
    if not weight_col:
        weight_col = "plays" if "plays" in df.columns else None

    parts = []
    group_keys = [k for k in ["team", "side"] if k in df.columns]
    if not group_keys:
        df = df.copy()
        df["side"] = "team"
        group_keys = ["team", "side"]

    for keys, g in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"team": None, "side": "team"}
        if "team" in group_keys:
            row["team"] = keys[group_keys.index("team")]
        if "side" in group_keys:
            row["side"] = keys[group_keys.index("side")]

        w = g[weight_col] if (weight_col and weight_col in g.columns) else None
        for c, how in agg_plan.items():
            if how == "sum":
                val = float(g[c].sum())
            elif how == "weighted_mean":
                val = _weighted_mean(g[c], w)
            else:
                val = float(g[c].mean())
            row[c] = val
        # zapisz sumę wybranej wagi (np. drives), a dla zgodności wstecz zachowaj też "plays" jeśli istnieje
        if weight_col and weight_col in g.columns:
            row[weight_col] = int(g[weight_col].sum())
        if "plays" in g.columns and weight_col != "plays":
            row["plays"] = int(g["plays"].sum())
        parts.append(row)
    return pd.DataFrame(parts)


def _pivot_off_def(wide: pd.DataFrame) -> pd.DataFrame:
    if "side" not in wide.columns:
        wide = wide.copy()
        wide["side"] = "team"
    if "team" not in wide.columns:
        raise ValueError("Expected a 'team' column after aggregation.")

    sides = set(str(x) for x in wide["side"].unique())
    if sides == {"team"}:
        suffix_map = {"team": "_team"}
    else:
        suffix_map = {"off": "_off", "def": "_def", "team": "_team"}

    frames = []
    for side, suf in suffix_map.items():
        sub = wide.loc[wide["side"] == side].drop(columns=["side"], errors="ignore").copy()
        metric_cols = [c for c in sub.columns if c != "team"]
        sub = sub.rename(columns={c: f"{c}{suf}" for c in metric_cols})
        frames.append(sub)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="team", how="outer")
    return out


def load_and_aggregate_file(path: Path, season: int, week: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure 'team'
    if "team" not in df.columns:
        alias = next((a for a in TEAM_ALIASES if a in df.columns), None)
        if alias is None:
            raise KeyError("team")
        df = df.rename(columns={alias: "team"})

    if "season" in df.columns:
        df = df.loc[df["season"] == season]
    if "season_type" in df.columns:
        df = df.loc[df["season_type"] == "REG"]
    if "week" in df.columns:
        df = df.loc[df["week"].astype(int) < int(week)]

    if df.empty:
        return pd.DataFrame()

    overrides = METRIC_OVERRIDES.get(path.name, {})
    agg = _aggregate_weekly(df, overrides)
    if agg.empty:
        return pd.DataFrame()
    return _pivot_off_def(agg)


def build_team_feature_table(in_dir: Path, season: int, week: int) -> pd.DataFrame:
    files = _list_candidate_files(in_dir)
    tables = []
    for f in files:
        try:
            t = load_and_aggregate_file(f, season=season, week=week)
            if not t.empty:
                tables.append(t)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
    if not tables:
        raise RuntimeError("No usable tables found. Check --in_dir or week cutoff (week must be > 1).")

    base = tables[0]
    for t in tables[1:]:
        base = base.merge(t, on="team", how="outer")

    cols = ["team"] + sorted([c for c in base.columns if c != "team"])
    return base[cols]


def build_matchup_row(team_features: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    th = team_features.loc[team_features["team"] == home]
    ta = team_features.loc[team_features["team"] == away]
    if th.empty or ta.empty:
        missing = home if th.empty else away
        raise ValueError(f"Team not found in features: {missing}")

    th = th.reset_index(drop=True).iloc[0]
    ta = ta.reset_index(drop=True).iloc[0]

    def_cols = [c for c in team_features.columns if c.endswith("_def")]
    off_cols = [c for c in team_features.columns if c.endswith("_off")]
    team_cols = [c for c in team_features.columns if c.endswith("_team")]

    data: Dict[str, float | str] = {"home": home, "away": away}

    for c in off_cols + def_cols + team_cols:
        data[f"home.{c}"] = float(th.get(c)) if pd.notna(th.get(c)) else np.nan
        data[f"away.{c}"] = float(ta.get(c)) if pd.notna(ta.get(c)) else np.nan

    for c in off_cols:
        stem = c[:-4]
        opp = stem + "_def"
        if opp in ta.index:
            data[f"edge.off_vs_def.{stem}"] = (th.get(c, np.nan) or np.nan) - (ta.get(opp, np.nan) or np.nan)
    for c in def_cols:
        stem = c[:-4]
        opp = stem + "_off"
        if opp in ta.index:
            data[f"edge.def_vs_off.{stem}"] = (th.get(c, np.nan) or np.nan) - (ta.get(opp, np.nan) or np.nan)
    for c in team_cols:
        stem = c[:-5]
        data[f"edge.team_vs_team.{stem}"] = (th.get(c, np.nan) or np.nan) - (ta.get(c, np.nan) or np.nan)

    return pd.DataFrame([data])


def _edge_direction_tag(x: float) -> str:
    if pd.isna(x):
        return ""
    return "HOME ↑" if x > 0 else ("AWAY ↑" if x < 0 else "EVEN")

def _zscore_series(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return (s - m)  # unikamy dzielenia przez 0; wszystkie 0 ⇒ równe rangi
    return (s - m) / sd
import re as _re

def build_edges_table(matchup: pd.DataFrame,
                      team_features: pd.DataFrame,
                      include_substr: str | None = None,
                      include_regex: str | None = None,
                      sort_mode: str = "abs",
                      top_n: int = 20,
                      min_abs: float = 0.0) -> pd.DataFrame:
    # zbierz edge.*
    edge_cols = [c for c in matchup.columns if c.startswith("edge.")]
    if include_substr:
        edge_cols = [c for c in edge_cols if include_substr.lower() in c.lower()]
    if include_regex:
        try:
            rx = re.compile(include_regex, flags=re.I)
        except re.error as e:
            print(f"[WARN] Invalid include_regex '{include_regex}': {e}. Ignoring filter.", file=sys.stderr)
            rx = None
        if rx:
            edge_cols = [c for c in edge_cols if rx.search(c)]


    if not edge_cols:
        return pd.DataFrame(columns=["edge_name", "value", "zscore"])

    s = matchup[edge_cols].iloc[0].dropna()
    if s.empty:
        return pd.DataFrame(columns=["edge_name", "value", "zscore"])

    df = s.rename("value").to_frame()
    df["edge_name"] = df.index

    # policz z-score edge’ów względem ligi
    # przygotuj mapy rozkładów dla _off/_def/_team
    def get_league_z(col: str) -> float | None:
        # col jest np. 'edge.off_vs_def.yds'
        base = col.replace("edge.", "")
        if "." not in base:
            return None
        prefix, stem = base.split(".", 1)
        if prefix == "off_vs_def":
            a = f"{stem}_off"
            b = f"{stem}_def"
            if a in team_features.columns and b in team_features.columns:
                za = _zscore_series(team_features[a].dropna())
                zb = _zscore_series(team_features[b].dropna())
                # wartości zespołów z matchupu:
                home = matchup[f"home.{a}"].iloc[0] if f"home.{a}" in matchup.columns else np.nan
                away = matchup[f"away.{b}"].iloc[0] if f"away.{b}" in matchup.columns else np.nan
                # osadź w rozkładach: najbliższa aproksymacja — po prostu standaryzujemy względem ligi (mean/std); 
                # przekształcamy skalą i przesunięciem:
                za_val = (home - team_features[a].mean()) / (team_features[a].std(ddof=0) or 1)
                zb_val = (away - team_features[b].mean()) / (team_features[b].std(ddof=0) or 1)
                return za_val - zb_val
        elif prefix == "def_vs_off":
            a = f"{stem}_def"
            b = f"{stem}_off"
            if a in team_features.columns and b in team_features.columns:
                # wyciągamy SKALARY z matchupu (iloc[0]) zamiast Series z .get(...)
                home = matchup[f"home.{a}"].iloc[0] if f"home.{a}" in matchup.columns else np.nan
                away = matchup[f"away.{b}"].iloc[0] if f"away.{b}" in matchup.columns else np.nan
                za_val = (home - team_features[a].mean()) / (team_features[a].std(ddof=0) or 1)
                zb_val = (away - team_features[b].mean()) / (team_features[b].std(ddof=0) or 1)
                return za_val - zb_val
            else:  # team_vs_team
                a = f"{stem}_team"
                if a in team_features.columns:
                    home = matchup[f"home.{a}"].iloc[0] if f"home.{a}" in matchup.columns else np.nan
                    away = matchup[f"away.{a}"].iloc[0] if f"away.{a}" in matchup.columns else np.nan
                    za_val = (home - team_features[a].mean()) / (team_features[a].std(ddof=0) or 1)
                    zb_val = (away - team_features[a].mean()) / (team_features[a].std(ddof=0) or 1)
                    return za_val - zb_val

        return None

    df["zscore"] = df["edge_name"].apply(get_league_z)
    df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce")


    # filtrowanie po progu
    df["abs"] = df["value"].abs()
    df = df[df["abs"] >= min_abs]

        # friendly hint when nothing passes filters
    if df.empty:
        reason = []
        if include_regex:
            reason.append(f"regex='{include_regex}'")
        if min_abs is not None:
            reason.append(f"min_abs={min_abs}")
        hint = " & ".join(reason) or "no qualifying edges"
        print(f"[INFO] No edges to show ({hint}). Try relaxing filters or lowering --min_abs.", file=sys.stderr)
        return df

        # drop rows that have no data on both sides
    df = df.dropna(subset=["value"])

             # sort
    if sort_mode == "zscore" and df["zscore"].notna().any():
        # stabilne sortowanie: najpierw |z|, potem |value| (abs), potem nazwa edge
        df["_zkey"] = df["zscore"].abs().fillna(-np.inf)
        df = df.sort_values(["_zkey", "abs", "edge_name"], ascending=[False, False, True])
        df = df.drop(columns="_zkey")
    else:
        # stabilne sortowanie: najpierw |value| (abs), potem nazwa edge
        df = df.sort_values(["abs", "edge_name"], ascending=[False, True])




    return df[["edge_name", "value", "zscore"]].head(top_n).reset_index(drop=True)

# === NEW: full comparison table (home, away, edge, dir, z) ===

def _league_z_for_edge(edge_name: str, matchup: pd.DataFrame, team_features: pd.DataFrame) -> float | None:
    # edge_name np. "off_vs_def:net_third_down_sr"
    try:
        prefix, stem = edge_name.split(":", 1)
    except ValueError:
        return None

    # lokalny helper — wyciąga skalara z matchupu (pierwszy wiersz)
    def _scalar(col: str) -> float:
        if col in matchup.columns:
            val = matchup[col].iloc[0]
            try:
                return float(val)
            except Exception:
                return np.nan
        return np.nan

    # z-score po kolumnie w team_features
    def _z(col: str, value: float) -> float:
        if col not in team_features.columns:
            return np.nan
        s = team_features[col].astype(float)
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return np.nan
        return (value - mu) / sd

    try:
        if prefix == "off_vs_def":
            a = f"{stem}_off"
            b = f"{stem}_def"
            home_val = _scalar(f"home.{a}")
            away_val = _scalar(f"away.{b}")
            return _z(a, home_val) - _z(b, away_val)

        elif prefix == "def_vs_off":
            a = f"{stem}_def"
            b = f"{stem}_off"
            home_val = _scalar(f"home.{a}")
            away_val = _scalar(f"away.{b}")
            return _z(a, home_val) - _z(b, away_val)

        else:  # "team_vs_team"
            a = f"{stem}_team"
            home_val = _scalar(f"home.{a}")
            away_val = _scalar(f"away.{a}")
            return _z(a, home_val) - _z(a, away_val)
    except Exception:
        return None


def build_comparison_table(matchup: pd.DataFrame,
                           team_features: pd.DataFrame,
                           include_substr: str | None = None,
                           include_regex: str | None = None,
                           sort_mode: str = "abs",
                           top_n: int = 9999) -> pd.DataFrame:
    """
    Zwraca pełną tabelę: Metric | Home | Away | Edge | Dir | Z
    Oparta o kolumny edge.* + raw home./away. wyliczone ze stemów.
    """
    if matchup.empty:
        return pd.DataFrame(columns=["metric", "home", "away", "edge", "dir", "z"])

    edge_cols = [c for c in matchup.columns if c.startswith("edge.")]
    if include_substr:
        edge_cols = [c for c in edge_cols if include_substr.lower() in c.lower()]
    if include_regex:
        try:
            rx = re.compile(include_regex, flags=re.I)
        except re.error as e:
            print(f"[WARN] Invalid include_regex '{include_regex}': {e}. Ignoring filter.", file=sys.stderr)
            rx = None
        if rx:
            edge_cols = [c for c in edge_cols if rx.search(c)]


    if not edge_cols:
        return pd.DataFrame(columns=["metric", "home", "away", "edge", "dir", "z"])

    rows = []
    for col in edge_cols:
        base = col.replace("edge.", "")
        if "." in base:
            prefix, stem = base.split(".", 1)
        else:
            prefix, stem = "edge", base

        # dobierz sufiksy do raw
        if prefix == "off_vs_def":
            hsfx, asfx, label_prefix = "_off", "_def", "Offense vs Opponent Defense"
        elif prefix == "def_vs_off":
            hsfx, asfx, label_prefix = "_def", "_off", "Defense vs Opponent Offense"
        else:
            hsfx, asfx, label_prefix = "_team", "_team", "Team vs Team"

        hcol = f"home.{stem}{hsfx}"
        acol = f"away.{stem}{asfx}"
        hval = matchup[hcol].iloc[0] if hcol in matchup.columns else np.nan
        aval = matchup[acol].iloc[0] if acol in matchup.columns else np.nan
        edge_val = matchup[col].iloc[0] if col in matchup.columns else np.nan

        label = f"{label_prefix} · {METRIC_LABELS.get(stem, stem)}"
        rows.append({
            "metric": label,
            "home": hval,
            "away": aval,
            "edge": edge_val,
            "dir": _edge_direction_tag(edge_val),
            "z": _league_z_for_edge(col, matchup, team_features)
        })

    df = pd.DataFrame(rows)

    # sortowanie
    if sort_mode == "zscore" and df["z"].notna().any():
        df = df.sort_values("z", key=lambda s: s.abs().fillna(-np.inf), ascending=False)
    else:
        df = df.assign(abs=lambda d: d["edge"].abs()).sort_values("abs", ascending=False).drop(columns=["abs"])

    return df.head(top_n).reset_index(drop=True)


def print_comparison_table(df: pd.DataFrame, max_rows: int = 120) -> None:
    if df is None or df.empty:
        print("\n(no comparison table)\n")
        return
    shown = df.head(max_rows).copy()
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 140):
        print("\nFull comparison table (top {} rows):\n".format(len(shown)))
        print(shown.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else str(x)))


def export_table_csv(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"📄 Saved comparison table CSV: {p}")



def pretty_print_edges(
    matchup: pd.DataFrame,
    team_features: pd.DataFrame,
    top_n: int = 20,
    min_abs: float = 0.5,
    include: str | None = None,
    include_regex: str | None = None,
    sort_mode: str = "abs",
) -> None:
    # Zbuduj tabelę edge’ów (value + zscore + filtrowanie/sortowanie)
    df = build_edges_table(
        matchup=matchup,
        team_features=team_features,
        include_substr=include,
        include_regex=include_regex,
        sort_mode=sort_mode,
        top_n=top_n,
        min_abs=min_abs,
    )

    if df is None or df.empty:
        print("No edges above threshold.")
        return

    # Dodaj kierunek przewagi
    df["dir"] = df["value"].apply(_edge_direction_tag)
    df["value"]  = pd.to_numeric(df["value"], errors="coerce").round(3)
    df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce").round(3)

    print("\nTop edges (sorted by {}):\n".format("z-score" if sort_mode == "zscore" else "|value|"))
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 140):
        print(df[["edge_name", "value", "zscore", "dir"]].to_string(index=False))


PREFIX_LABELS = {
    "off_vs_def": "Offense vs Opponent Defense",
    "def_vs_off": "Defense vs Opponent Offense",
    "team_vs_team": "Team vs Team",
}

def build_scorecard_df(matchup: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    if matchup.empty:
        return pd.DataFrame(columns=["metric", "edge", "dir"])

    edge_cols = [c for c in matchup.columns if c.startswith("edge.")]
    if not edge_cols:
        return pd.DataFrame(columns=["metric", "edge", "dir"])

    # Heurystyka wyboru „kluczowych” metryk (tak jak wcześniej)
    KEY_HINTS = [
        "epa_play_all", "epa_all", "epa/play", "epa_play",
        "epa_play_run", "epa_run", "epa_rush", "epa_play_pass", "epa_pass",
        "success", "sr", "first_down",
        "third_down", "3rd_down", "conv_rate",
        "rz_td", "redzone", "rz_",
        "explosive", "xpl", "xpass_20", "xrush_10",
        "pen_per_100", "penalty_epa", "penalty_yds",
        "hidden", "field_pos", "avg_start_yardline_100", "st_",
        "second_half", "h2_", "adj",
        "ppd", "td_drives", "rz_drives", "yds_per_drive", "time_per_drive",
    ]

    s = matchup[edge_cols].iloc[0].dropna()
    if s.empty:
        return pd.DataFrame(columns=["metric", "edge", "dir"])

    keep = [idx for idx in s.index if any(h in idx.lower() for h in KEY_HINTS)]
    if not keep:
        keep = list(s.index)  # fallback: jeśli nie ma trafień, pokaż cokolwiek

    df = (
        s.loc[keep].rename("edge").to_frame()
         .assign(abs=lambda d: d["edge"].abs())
         .sort_values("abs", ascending=False)
         .drop(columns=["abs"])
         .head(max_rows)
         .copy()
    )

    def _label(col: str) -> str:
        base = col.replace("edge.", "")
        if "." in base:
            prefix, stem = base.split(".", 1)
        else:
            prefix, stem = "edge", base
        prefix_nice = PREFIX_LABELS.get(prefix, prefix)
        stem_nice = METRIC_LABELS.get(stem, stem)
        return f"{prefix_nice} · {stem_nice}"

    df.insert(0, "metric", df.index.map(_label))
    df["dir"] = df["edge"].apply(_edge_direction_tag)
    df = df[["metric", "edge", "dir"]].reset_index(drop=True)
    return df

def export_scorecard_md(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        content = "_(no scorecard data)_\n"
    else:
        # prosta tabela Markdown
        content = "| Metric | Edge | Dir |\n|---|---:|:---|\n"
        for _, r in df.iterrows():
            content += f"| {r['metric']} | {r['edge']:.3f} | {r['dir']} |\n"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def export_scorecard_html(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        html = "<p><em>(no scorecard data)</em></p>"
    else:
        # prosta tabela HTML
        rows = "".join(
            f"<tr><td>{r['metric']}</td><td style='text-align:right'>{r['edge']:.3f}</td><td>{r['dir']}</td></tr>"
            for _, r in df.iterrows()
        )
        html = (
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<thead><tr><th>Metric</th><th>Edge</th><th>Dir</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

def print_raw_for_stems(matchup: pd.DataFrame, stems_csv: str | None) -> None:
    if not stems_csv:
        return
    stems = [s.strip() for s in stems_csv.split(",") if s.strip()]
    if not stems:
        return

    def _val(col: str) -> float:
        if col in matchup.columns:
            v = matchup[col].iloc[0]
            try:
                return float(v)
            except Exception:
                return np.nan
        return np.nan

    rows = []
    for stem in stems:
        for prefix, asuf, bsuf, label in [
            ("off_vs_def", "_off", "_def", "Off vs Def"),
            ("def_vs_off", "_def", "_off", "Def vs Off"),
            ("team_vs_team", "_team", "_team", "Team vs Team"),
        ]:
            home_col = f"home.{stem}{asuf}"
            away_col = f"away.{stem}{bsuf}" if prefix != "team_vs_team" else f"away.{stem}{asuf}"
            edge_col = f"edge.{prefix}.{stem}"
            if edge_col in matchup.columns:
                rows.append({
                    "metric": f"{label} · {stem}",
                    "home": _val(home_col),
                    "away": _val(away_col),
                    "edge": _val(edge_col),
                })

    if rows:
        df = pd.DataFrame(rows)
        print("\nRaw values for selected metrics:\n")
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 140):
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))



SCORECARD_LEGEND = (
    "Legend: HOME ↑ means advantage to the home team, AWAY ↑ to the away team. "
    "Prefix: Offense vs Opponent Defense / Defense vs Opponent Offense / Team vs Team. "
    "If --normalize per_game is used, sum-like metrics are per-game averages."
)


def print_scorecard(matchup: pd.DataFrame, max_rows: int = 12, legend: bool = False) -> None:
    if legend:
        print("\n" + SCORECARD_LEGEND + "\n")
    # Zbierz wszystkie edge.* i wybierz podzbiór kluczowych
    if matchup.empty:
        return
    edge_cols = [c for c in matchup.columns if c.startswith("edge.")]
    if not edge_cols:
        print("\n(no edges to scorecard)\n")
        return

    # Kluczowe „stems” – dopasujemy przez substring (bezpiecznie przy różnych nazwach)
    KEY_HINTS = [
        # efektywność ogólna
        "epa_play_all", "epa_all", "epa/play", "epa_play",
        # run/pass
        "epa_play_run", "epa_run", "epa_rush", "epa_play_pass", "epa_pass",
        # SR i 1st down
        "success", "sr", "first_down",
        # 3rd down / conversion
        "third_down", "3rd_down", "conv_rate",
        # red zone
        "rz_td", "redzone", "rz_",
        # explosive plays
        "explosive", "xpl", "xpass_20", "xrush_10",
        # kary / dyscyplina
        "pen_per_100", "penalty_epa", "penalty_yds",
        # special teams / hidden yards / field position
        "hidden", "field_pos", "avg_start_yardline_100", "st_",
        # second-half adjustments
        "second_half", "h2_", "adj",
        # drive efficiency
        "ppd", "td_drives", "rz_drives", "yds_per_drive", "time_per_drive",
    ]

    s = matchup[[c for c in edge_cols]].iloc[0].dropna()
    if s.empty:
        print("\n(no edges to scorecard)\n")
        return

    # wybieramy tylko kolumny z którymkolwiek hintem
    keep = [idx for idx in s.index if any(h in idx.lower() for h in KEY_HINTS)]
    if not keep:
        print("\nScorecard: no key metrics found by hints.\n")
        return

    df = (
        s.loc[keep].rename("edge").to_frame()
         .assign(abs=lambda d: d["edge"].abs())
         .sort_values("abs", ascending=False)
         .drop(columns=["abs"])
    )
    df = df.head(max_rows).copy()
    df["dir"] = df["edge"].apply(_edge_direction_tag)

    # Czytelne etykiety
    def _label(col: str) -> str:
        base = col.replace("edge.", "")
        if "." in base:
            prefix, stem = base.split(".", 1)
        else:
            prefix, stem = "edge", base
        prefix_nice = PREFIX_LABELS.get(prefix, prefix)
        stem_nice = METRIC_LABELS.get(stem, stem)
        return f"{prefix_nice} · {stem_nice}"



    df.insert(0, "metric", df.index.map(_label))
    df = df[["metric", "edge", "dir"]]

    print("\nScorecard (key edges):\n")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
        print(df.to_string(index=False))


def _compute_games_played(in_dir: Path, season: int, week: int) -> pd.DataFrame:
    """
    Spróbuj policzyć liczbę rozegranych meczów (REG) do week-1 na podstawie jednego z weekly CSV.
    Kolejność prób: drive_efficiency -> redzone -> third_down.
    Zwraca DataFrame: team, games_played
    """
    candidates = [
        f"drive_efficiency_weekly_{season}.csv",
        f"redzone_weekly_{season}.csv",
        f"third_down_weekly_{season}.csv",
    ]
    for name in candidates:
        p = in_dir / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "team" not in df.columns:
            # spróbuj aliasów
            for a in ["posteam", "defteam", "club", "club_code", "abbr", "Team"]:
                if a in df.columns:
                    df = df.rename(columns={a: "team"})
                    break
        if "team" not in df.columns:
            continue
        if "season" in df.columns:
            df = df.loc[df["season"] == season]
        if "season_type" in df.columns:
            df = df.loc[df["season_type"] == "REG"]
        if "week" in df.columns:
            df = df.loc[df["week"].astype(int) < int(week)]
        if df.empty:
            continue
        gp = df.groupby("team")["week"].nunique().reset_index(name="games_played") if "week" in df.columns else \
             df.groupby("team").size().reset_index(name="games_played")
        return gp
    # fallback: brak danych → przyjmij 1 (żeby nie dzielić przez 0), ale poinformuj
    return pd.DataFrame({"team": [], "games_played": []})


# jakie kolumny traktujemy jako "sumowalne" (po stemie nazwy, PRZED sufiksami _off/_def/_team)
SUM_NORMALIZE_STEMS = {
    "yds", "yds_h1", "yds_h2",
    "plays", "plays_h1", "plays_h2",
    "drives", "td_drives", "fg_drives", "rz_drives", "score_drives", "punt_drives", "to_drives",
    "ppd_basic_sum",
    "total_yards_off", "yards_allowed",
    "punt_plays",
}


# Czytelne etykiety metryk
METRIC_LABELS = {
    "ppd_basic_sum": "PPD (Points per Drive, sum)",
    "ppd_basic": "PPD (Points per Drive)",
    "yds": "Total Yards",
    "plays": "Plays",
    "plays_h1": "Plays 1H",
    "plays_h2": "Plays 2H",
    "rz_drives": "Red Zone Drives",
    "td_drives": "TD Drives",
    "fg_drives": "FG Drives",
    "to_drives": "Turnover Drives",
    "score_drives": "Scoring Drives",
    "punt_drives": "Punt Drives",
    "sr": "Success Rate",
    "conv_rate": "3rd Down Conv Rate",
    "rz_td_rate": "Red Zone TD%",
    "explosive_plays_off": "Explosive Plays (O)",
    "explosive_plays_allowed": "Explosive Plays Allowed (D)",
    "pen_per_100_plays_w": "Penalties / 100 plays",
    "penalty_yds_pg": "Penalty Yards (pg)",
    "st_score": "Special Teams Score",
    "field_pos_advantage": "Field Position Edge",
    "avg_start_yardline_100_off": "Avg Starting Yards (O)",
}

# Dodatkowe etykiety — uzupełnienia
METRIC_LABELS.update({
    "adj_epa": "Adj EPA",
    "adj_sr": "Adj Success Rate",
    "auto_fd_allowed": "Auto Fd Allowed",
    "avg_start_yardline_100": "Avg Starting Yards (O)",
    "avg_start_yardline_100_faced": "Avg Starting Yards Faced",
    "chunk20_allowed": "Chunk 20+ Allowed",
    "chunk20_rate": "Chunk 20+ Rate",
    "def_3d_allowed_conv": "Defense · 3rd Down Allowed Conversions",
    "def_3d_allowed_rate": "Defense · 3rd Down Allowed Rate",
    "def_3d_att_faced": "Defense · 3rd Down Attempts Faced",
    "def_3d_avg_togo_faced": "Defense · 3rd Down Avg Togo Faced",
    "def_3d_epa_allowed_per_play": "Defense · 3rd Down EPA Allowed Per Play",
    "def_3d_pass_rate_faced": "Defense · 3rd Down Pass Rate Faced",
    "def_3d_sr_allowed": "Defense · 3rd Down Success Rate Allowed",
    "def_auto_fd_allowed_pg": "Defense · Auto Fd Allowed (pg)",
    "def_dpi_yds_pg": "Defense · Dpi Yards (pg)",
    "def_epa_per_play_allowed": "Defense · EPA Per Play Allowed",
    "def_epa_per_play_allowed_delta3": "Defense · EPA Per Play Allowed delta3",
    "def_epa_per_play_allowed_roll3": "Defense · EPA Per Play Allowed roll3",
    "def_median_epa_allowed": "Defense · Median EPA Allowed",
    "def_plays": "Defense · Plays",
    "defteam": "Defteam",
    "dpi_yds": "Dpi Yards",
    "drives": "Drives",
    "early_down_epa": "Early Down EPA",
    "early_down_epa_allowed": "Early Down EPA Allowed",
    "epa_avg_h1": "EPA Avg h1",
    "epa_avg_h2": "EPA Avg h2",
    "epa_h1": "EPA h1",
    "epa_h2": "EPA h2",
    "epa_per_play": "EPA Per Play",
    "expl_pass_allowed": "Expl Pass Allowed",
    "expl_rate_allowed": "Expl Rate Allowed",
    "expl_rush_allowed": "Expl Rush Allowed",
    "expl_total_allowed": "Expl Total Allowed",
    "explosive_plays": "Explosive Plays",
    "explosive_plays_allowed": "Explosive Plays Allowed",
    "explosive_rate": "Explosive Rate",
    "explosive_rate_allowed": "Explosive Rate Allowed",
    "fg_att": "FG Attempts",
    "fg_drives": "FG Drives",
    "fg_epa_per_att": "FG EPA Per Attempts",
    "fg_made": "FG Made",
    "fg_pct": "FG Pct",
    "fg_rate": "FG Rate",
    "field_pos_advantage": "Field Position Edge",
    "fourth_down_sr": "Fourth Down Success Rate",
    "fourth_down_sr_allowed": "Fourth Down Success Rate Allowed",
    "games": "Games",
    "hidden_yards_per_drive": "Hidden Yards Per Drive",
    "ko_opp_start_yd100": "Kickoff Opp Start yd100",
    "ko_plays": "Kickoff Plays",
    "ko_tb_rate": "Kickoff Tb Rate",
    "late_down_epa": "Late Down EPA",
    "late_down_epa_allowed": "Late Down EPA Allowed",
    "momentum_3w": "Momentum 3w",
    "net_early_down_epa": "Net · Early Down EPA",
    "net_epa": "Net · EPA",
    "net_epa_delta3": "Net · EPA delta3",
    "net_epa_roll3": "Net · EPA roll3",
    "net_explosive_rate": "Net · Explosive Rate",
    "net_fourth_down_sr": "Net · Fourth Down Success Rate",
    "net_late_down_epa": "Net Late Down EPA",
    "net_red_zone_epa": "Net Red Zone EPA",
    "net_third_down_sr": "Net 3rd Down SR",
    "off_3d_att": "Offense · 3rd Down Attempts",
    "off_3d_avg_togo": "Offense · 3rd Down Avg Togo",
    "off_3d_conv": "Offense · 3rd Down Conversions",
    "off_3d_epa_per_play": "Offense · 3rd Down EPA Per Play",
    "off_3d_pass_rate": "Offense · 3rd Down Pass Rate",
    "off_3d_rate": "Offense · 3rd Down Rate",
    "off_3d_sr": "Offense · 3rd Down Success Rate",
    "off_epa_per_play": "Offense · EPA Per Play",
    "off_epa_per_play_delta3": "Offense · EPA Per Play delta3",
    "off_epa_per_play_roll3": "Offense · EPA Per Play roll3",
    "off_median_epa": "Offense · Median EPA",
    "opp": "Opp",
    "pass_epa_per_play": "Pass EPA Per Play",
    "pass_epa_per_play_allowed": "Pass EPA Per Play Allowed",
    "pass_rush_delta": "Pass Rush Delta",
    "pen_per_100_plays": "Pen Per 100 Plays",
    "pen_per_100_plays_w": "Pen Per 100 Plays W",
    "penalties": "Penalties",
    "penalties_pg": "Penalties (pg)",
    "penalty_yds": "Penalty Yards",
    "penalty_yds_pg": "Penalty Yards (pg)",
    "plays": "Plays",
    "plays_h1": "Plays h1",
    "plays_h2": "Plays h2",
    "plays_per_drive": "Plays Per Drive",
    "pot_per_game": "Points off Turnovers (pg)",
    "pot_per_takeaway": "Pot Per Takeaway",
    "pot_points": "Pot Points",
    "ppd_basic": "Ppd Basic",
    "ppd_basic_sum": "Ppd Basic Sum",
    "ppd_points": "Ppd Points",
    "ppd_true_sum": "Ppd True Sum",
    "presnap_pen_rate": "Presnap Pen Rate",
    "presnap_pen_rate_w": "Presnap Pen Rate W",
    "punt_drives": "Punt Drives",
    "punt_net": "Punt Net",
    "punt_opp_start_yd100": "Punt Opp Start yd100",
    "punt_plays": "Punt Plays",
    "punt_rate": "Punt Rate",
    "red_zone_epa": "Red Zone EPA",
    "red_zone_epa_allowed": "Red Zone EPA Allowed",
    "redzone_drive_rate": "Redzone Drive Rate",
    "ret_epa": "Ret EPA",
    "ret_epa_against": "Ret EPA Against",
    "ret_epa_diff": "Ret EPA Diff",
    "ret_plays": "Ret Plays",
    "rush_epa_per_play": "Rush EPA Per Play",
    "rush_epa_per_play_allowed": "Rush EPA Per Play Allowed",
    "rz_drives": "Red Zone Drives",
    "rz_efficiency": "Red Zone Efficiency",
    "rz_efficiency_allowed": "Red Zone Efficiency Allowed",
    "rz_empty": "Red Zone Empty",
    "rz_empty_allowed": "Red Zone Empty Allowed",
    "rz_fg": "Red Zone FG",
    "rz_fg_allowed": "Red Zone FG Allowed",
    "rz_pen_rate": "Red Zone Pen Rate",
    "rz_pen_rate_w": "Red Zone Pen Rate W",
    "rz_td": "Red Zone Td",
    "rz_td_allowed": "Red Zone Td Allowed",
    "rz_trips": "Red Zone Trips",
    "rz_trips_allowed": "Red Zone Trips Allowed",
    "score_drives": "Score Drives",
    "score_rate": "Score Rate",
    "sr": "Success Rate",
    "sr_h1": "Success Rate h1",
    "sr_h2": "Success Rate h2",
    "st_epa": "St EPA",
    "st_epa_per_play": "St EPA Per Play",
    "st_pen_rate": "St Pen Rate",
    "st_plays": "St Plays",
    "st_score": "Special Teams Score",
    "start_own_yardline_avg": "Start Own Yardline Avg",
    "start_yardline_100_avg": "Start Yardline 100 Avg",
    "success_rate": "Success Rate",
    "success_rate_allowed": "Success Rate Allowed",
    "success_rate_h1": "Success Rate h1",
    "success_rate_h2": "Success Rate h2",
    "successes": "Successes",
    "takeaways": "Takeaways",
    "td_drives": "Td Drives",
    "td_rate": "Td Rate",
    "third_down_sr": "Third Down Success Rate",
    "third_down_sr_allowed": "Third Down Success Rate Allowed",
    "third_fourth_pen": "Third Fourth Pen",
    "third_fourth_pen_pg": "3rd/4th Down Penalties (pg)",
    "to_drives": "Turnover Drives",
    "total_yards": "Total Yards",
    "turnover_epa": "Turnover EPA",
    "turnover_epa_forced": "Turnover EPA Forced",
    "turnover_epa_net": "Turnover EPA Net",
    "turnover_rate": "Turnover Rate",
    "xp_att": "XP Attempts",
    "xp_epa_per_att": "XP EPA Per Attempts",
    "xp_made": "XP Made",
    "xp_pct": "XP Pct",
    "yards_allowed": "Yards Allowed",
    "yds": "Yards",
    "yds_per_drive": "Yards Per Drive",
})

    # --- Final overrides: ręcznie ładniejsze nazwy niż z generatora ---
METRIC_LABELS.update({
    "ppd_basic_sum": "PPD (Points per Drive, sum)",   # zamiast "Ppd Basic Sum"
    "ppd_basic":     "PPD (Points per Drive)",
    "td_drives":     "TD Drives",                      # zamiast "Td Drives"
    "pen_per_100_plays_w": "Penalties / 100 plays",    # pełniejsza nazwa
})




def _normalize_team_features_per_game(team_features: pd.DataFrame, games_played: pd.DataFrame) -> pd.DataFrame:
    """
    Dzieli wartości sumowalnych kolumn przez liczbę meczów (per team).
    Sufiksy: _off/_def/_team zachowujemy — działamy po każdym z nich.
    """
    if games_played is None or games_played.empty:
        print("[WARN] No games_played table found — skipping per_game normalization.")
        return team_features

    tf = team_features.copy()
    gp = games_played.copy()
    gp["games_played"] = gp["games_played"].astype(float).clip(lower=1.0)  # zabezpieczenie

    tf = tf.merge(gp, on="team", how="left")
    tf["games_played"] = tf["games_played"].fillna(1.0)

    # dla każdej kolumny z sufiksami sprawdzamy stem
    for col in list(tf.columns):
        if col in {"team", "games_played"}:
            continue
        # tylko kolumny numeryczne
        if not pd.api.types.is_numeric_dtype(tf[col]):
            continue
        # sprawdź sufiks
        stem, suf = None, None
        for sfx in ("_off", "_def", "_team"):
            if col.endswith(sfx):
                stem = col[: -len(sfx)]
                suf = sfx
                break
        if stem is None:
            # kolumna bez sufiksu — pomijamy
            continue
        # jeśli stem wskazuje na sumowalną metrykę → dzielimy przez games_played
        if stem in SUM_NORMALIZE_STEMS:
            tf[col] = tf[col] / tf["games_played"]

    return tf.drop(columns=["games_played"], errors="ignore")


def main():
    ap = argparse.ArgumentParser(description="Build a matchup comparison using season-to-date metrics (through week-1).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True, help="Game week; we will use data from weeks < week")
    ap.add_argument("--home", type=str, required=True)
    ap.add_argument("--away", type=str, required=True)
    ap.add_argument("--in_dir", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--top", type=int, default=20, help="How many top edges to display")
    ap.add_argument("--min_abs", type=float, default=0.5, help="Min absolute edge to display")
    ap.add_argument("--include", type=str, default=None, help="Filter edge columns by substring (e.g., off_vs_def)")
    ap.add_argument("--score_n", type=int, default=17, help="How many key edges to show in scorecard")
    ap.add_argument("--normalize",type=str,choices=["none", "per_game"],default="none",help="Normalization mode for sum-like metrics (default: none).")
    ap.add_argument("--legend", action="store_true", help="Print a one-line legend above the scorecard")
    ap.add_argument("--export_md", type=str, default=None, help="Path to save the scorecard as a Markdown file")
    ap.add_argument("--export_html", type=str, default=None, help="Path to save the scorecard as an HTML file")
    ap.add_argument("--include_regex", type=str, default=None, help="Regex to filter edge names (applied to full 'edge.*' column)")
    ap.add_argument("--sort", type=str, choices=["abs", "zscore"], default="abs", help="Sort Top edges by absolute value or by league z-score")
    ap.add_argument("--show_raw", type=str, default=None, help="Comma-separated stems to display raw home/away values for (e.g., 'yds,ppd_basic_sum,rz_drives')")
    ap.add_argument("--export_table", type=str, default=None, help="Path to save the full comparison table (CSV)")
    ap.add_argument("--print_table", action="store_true", help="Print the full comparison table to console")
    ap.add_argument("--table_include", type=str, default=None, help="Substring filter for metrics included in the table")
    ap.add_argument("--table_regex", type=str, default=None, help="Regex filter for metrics included in the table")
    ap.add_argument("--table_top", type=int, default=9999, help="Limit rows in the comparison table (after sorting)")
    ap.add_argument("--table_sort", type=str, choices=["abs", "zscore"], default="abs", help="Sort table by |edge| or by z-score")




    args = ap.parse_args()

    if args.week <= 1:
        raise SystemExit("Week must be >= 2 (need at least one completed week to compare)")

    warnings.simplefilter("ignore", category=FutureWarning)

    in_dir = Path(args.in_dir)
    out_path = Path(args.out) if args.out else None

    print(f"📦 Loading features from: {in_dir}")
    feats = build_team_feature_table(in_dir, season=args.season, week=args.week)
    print(f"✅ Built team feature table with shape {feats.shape}")
    if args.normalize == "per_game":
        gp = _compute_games_played(in_dir, season=args.season, week=args.week)
        feats = _normalize_team_features_per_game(feats, gp)

    mu = build_matchup_row(feats, home=args.home, away=args.away)
    mu.insert(0, "week", args.week)
    mu.insert(0, "season", args.season)

    if out_path is None:
        out_dir = in_dir / "matchups"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.home}_{args.away}_w{args.week}_{args.season}.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    mu.to_csv(out_path, index=False)
    print(f"💾 Saved matchup: {out_path}")
    
    
    

    pretty_print_edges(mu, feats, top_n=args.top, min_abs=args.min_abs, include=args.include, include_regex=args.include_regex, sort_mode=args.sort)


        # Legend + Scorecard
    if args.legend:
        print("\n" + SCORECARD_LEGEND + "\n")

    score_df = build_scorecard_df(mu, max_rows=args.score_n)
    if not score_df.empty:
        print("\nScorecard (key edges):\n")
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
            print(score_df.to_string(index=False))
    else:
        print("\nScorecard: (no data)\n")

    # Eksporty
    if args.export_md:
        export_scorecard_md(score_df, args.export_md)
        print(f"📝 Saved scorecard (Markdown): {args.export_md}")

    if args.export_html:
        export_scorecard_html(score_df, args.export_html)
        print(f"🖼️ Saved scorecard (HTML): {args.export_html}")


    # Opcjonalny podgląd surowych wartości dla wybranych metryk
    print_raw_for_stems(mu, args.show_raw)
        # Pełna tabela porównawcza (home/away/edge/dir/z)
    comp_df = build_comparison_table(
        matchup=mu,
        team_features=feats,
        include_substr=args.table_include,
        include_regex=args.table_regex,
        sort_mode=args.table_sort,
        top_n=args.table_top
    )

    if args.print_table:
        print_comparison_table(comp_df, max_rows=min(args.table_top, 120))

    if args.export_table:
        export_table_csv(comp_df, args.export_table)

if __name__ == "__main__":
    main()