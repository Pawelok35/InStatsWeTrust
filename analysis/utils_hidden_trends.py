from __future__ import annotations

from pathlib import Path
import pandas as pd

HIDDEN_TREND_COLS = [
    "game_rhythm_q4",
    "play_call_entropy_neutral",
    "neutral_pass_rate",
    "neutral_plays",
    "drive_momentum_3plus",
    "drives_with_3plus",
    "drives_total",
    "field_flip_eff",
    "punts_tracked",
]


def load_hidden_trends(season: int, base_dir: Path) -> pd.DataFrame:
    """Wczytaj CSV z hidden trends i zwróć df z indexem team."""
    p = base_dir / f"data/processed/team_hidden_trends_{season}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Hidden Trends CSV not found: {p}")
    df = pd.read_csv(p)
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    return df.set_index("team")


def compute_hidden_trends_edges(df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """Zwraca dict z surowymi wartościami i deltami HOME–AWAY dla listy HIDDEN_TREND_COLS."""
    h = df.loc[home_team, HIDDEN_TREND_COLS].to_dict() if home_team in df.index else {k: None for k in HIDDEN_TREND_COLS}
    a = df.loc[away_team, HIDDEN_TREND_COLS].to_dict() if away_team in df.index else {k: None for k in HIDDEN_TREND_COLS}
    edges = {k: (h[k] - a[k]) if pd.notnull(h.get(k)) and pd.notnull(a.get(k)) else None for k in HIDDEN_TREND_COLS}
    return {"home": h, "away": a, "edges": edges}

