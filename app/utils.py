from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from dateutil import parser as dtparser
from pathlib import Path
import json
import re

# =========================
# MODELE DANYCH (minimal)
# =========================
class Edge(BaseModel):
    name: str
    team: Optional[str] = None
    value: Optional[float] = None

class Signals(BaseModel):
    side: Optional[str] = None
    total: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

class Game(BaseModel):
    home: str
    away: str
    kickoff: Optional[str] = None
    signals: Optional[Signals] = None
    edges: Optional[List[Edge]] = None
    risks: Optional[List[str]] = None
    why: Optional[List[str]] = None

class WeekAnalysis(BaseModel):
    season: int
    week: int
    generated_at: Optional[str] = None
    games: List[Game]

# =========================
# IO HELPERS
# =========================
def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def parse_week_analysis(path: str | Path) -> WeekAnalysis:
    raw = load_json(path)
    return WeekAnalysis(**raw)

def game_label(game: Game) -> str:
    # Format: "AWAY @ HOME — 2025-10-19 18:00"
    label = f"{game.away} @ {game.home}"
    if game.kickoff:
        try:
            ts = dtparser.isoparse(game.kickoff)
            label += f" — {ts.strftime('%Y-%m-%d %H:%M')}"
        except Exception:
            label += " — kickoff N/A"
    return label

# =========================
# PARSER PS1 ($games = @(...))
# =========================
NFL_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals",      "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",       "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",      "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",     "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",         "DEN": "Denver Broncos",
    "DET": "Detroit Lions",          "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",         "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",   "KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",   "LAR": "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",      "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",      "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",     "NYG": "New York Giants",
    "NYJ": "New York Jets",          "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",    "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",       "WAS": "Washington Commanders"
}

PS_GAME_LINE = re.compile(
    r'@\{\s*home\s*=\s*"(?P<home>[A-Z]{2,3})"\s*;\s*away\s*=\s*"(?P<away>[A-Z]{2,3})"\s*\}'
)

def load_games_from_ps1(path: str | Path) -> list[dict]:
    """
    Parsuje $games = @( @{ home = "CIN"; away = "PIT" } ... ) z pliku PS1.
    Zwraca listę słowników: {"home_abbr","away_abbr","home","away"}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"run_week_matchups.ps1 not found at: {p}")

    txt = p.read_text(encoding="utf-8", errors="ignore")
    games: list[dict] = []
    for m in PS_GAME_LINE.finditer(txt):
        home_abbr = m.group("home")
        away_abbr = m.group("away")
        home = NFL_ABBR_TO_NAME.get(home_abbr, home_abbr)
        away = NFL_ABBR_TO_NAME.get(away_abbr, away_abbr)
        games.append({
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home": home,
            "away": away,
        })
    return games

def game_key_from_abbr(home_abbr: str, away_abbr: str) -> str:
    """Zwraca klucz HOME_AWAY (np. CIN_PIT)."""
    return f"{home_abbr}_{away_abbr}"

def confidence_badge(conf: float | None) -> str:
    """Zwraca HTML badge dla confidence (kolor wg progu)."""
    if conf is None:
        return "<span style='padding:2px 8px;border-radius:12px;background:#444;color:#ddd;'>n/a</span>"
    if conf >= 0.66:
        bg = "#0a3"   # mocny zielony
    elif conf >= 0.55:
        bg = "#063"   # średni zielony
    else:
        bg = "#444"   # neutral
    return f"<span style='padding:2px 10px;border-radius:12px;background:{bg};color:white;font-weight:600'>{int(conf*100)}%</span>"

def detail_md_path(home_abbr: str, away_abbr: str, week: int, season: int) -> Path:
    """Buduje ścieżkę do pliku Markdown z długim opisem (HOME_AWAY_w{week}_{season}.md)."""
    fname = f"{home_abbr}_{away_abbr}_w{week}_{season}.md"
    return Path("data/processed/analyses/details") / fname

# aliasy nazw (gdy JSON ma inną frazę niż mapowanie z PS1) – dodawaj w razie potrzeby
NAME_ALIASES = {
    "Los Angeles Rams": {"LA Rams"},
    "Los Angeles Chargers": {"LA Chargers"},
}
def equal_names(a: str, b: str) -> bool:
    if a == b:
        return True
    # aliasy
    for canon, alset in NAME_ALIASES.items():
        if (a == canon and b in alset) or (b == canon and a in alset):
            return True
    return False

