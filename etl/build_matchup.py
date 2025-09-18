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
    files: List[Path] = []
    for p in in_dir.glob("*.csv"):
        name = p.name.lower()
        if name.endswith("_weekly_2024.csv") or name.endswith("_weekly_2025.csv"):
            files.append(p)
        elif name.endswith("_team_2024.csv") or name.endswith("_team_2025.csv"):
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

        w = g["plays"] if "plays" in g.columns else None
        for c, how in agg_plan.items():
            if how == "sum":
                val = float(g[c].sum())
            elif how == "weighted_mean":
                val = _weighted_mean(g[c], w)
            else:
                val = float(g[c].mean())
            row[c] = val
        if "plays" in g.columns:
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

def pretty_print_edges(matchup: pd.DataFrame, top_n: int = 20, min_abs: float = 0.5, include: str | None = None) -> None:
    # Zbierz kolumny edge.*
    edge_cols = [c for c in matchup.columns if c.startswith("edge.")]
    if include:
        edge_cols = [c for c in edge_cols if include.lower() in c.lower()]
    if not edge_cols:
        print("No edge columns computed.")
        return

    s = matchup[edge_cols].iloc[0].dropna()
    if s.empty:
        print("No edges (all NaN).")
        return

    df = (
        s.rename("value").to_frame()
         .assign(abs=lambda d: d["value"].abs())
    )
    # próg istotności
    df = df[df["abs"] >= min_abs]
    if df.empty:
        print("No edges above threshold.")
        return

    df = df.sort_values("abs", ascending=False).drop(columns=["abs"]).head(top_n).copy()
    df["dir"] = df["value"].apply(_edge_direction_tag)

    print("\nTop edges (by |value|):\n")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
        print(df[["value", "dir"]].to_string())

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
    ap.add_argument("--score_n", type=int, default=12, help="How many key edges to show in scorecard")
    ap.add_argument("--normalize",type=str,choices=["none", "per_game"],default="none",help="Normalization mode for sum-like metrics (default: none).")
    ap.add_argument("--legend", action="store_true", help="Print a one-line legend above the scorecard")
    ap.add_argument("--export_md", type=str, default=None, help="Path to save the scorecard as a Markdown file")
    ap.add_argument("--export_html", type=str, default=None, help="Path to save the scorecard as an HTML file")


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

    pretty_print_edges(mu, top_n=args.top, min_abs=args.min_abs, include=args.include)

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


    print_scorecard(mu, max_rows=args.score_n)


if __name__ == "__main__":
    main()
