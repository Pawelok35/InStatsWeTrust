from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

import json
import re
from functools import lru_cache

import pandas as pd
from dateutil import parser as dtparser
from pydantic import BaseModel, Field

# =========================================================
# 1) Parser "wklejonej analizy" -> 7 zakładek (bez kolizji regex)
# =========================================================

TAB_TITLES = [
    "1) Profil drużyn [2024 → 2025]",
    "2) Top przewagi (Top 12 Edges — Week 6, 2025)",
    "3) Diagnoza matchupu (Game Dynamics)",
    "4) Model punktacji i prognoza",
    "5) Scenariusze gry (Game Scripts)",
    "6) Ryzyka i punkty krytyczne (Swing Factors)",
    "7) Prognoza i typowanie (Final pick)",
]

# aliasy nagłówków (PL/EN) — możesz dopisywać własne frazy/regexy
HEADER_ALIASES: Dict[int, List[str]] = {
    1: [r"Profil drużyn\s*\[.*?\]", r"Team Profile", r"Profile 2024", r"Profil 2024"],
    2: [r"Top przewagi.*", r"Top 12 Edges.*", r"Edges.*Week.*", r"Top edges.*"],
    3: [r"Diagnoza matchupu.*", r"Game Dynamics.*", r"Matchup diagnosis.*"],
    4: [r"Model punktacji.*", r"Prognoza.*model.*", r"Scoring model.*", r"Forecast.*model.*"],
    5: [r"Scenariusze gry.*", r"Game Scripts.*", r"Scripts.*"],
    6: [r"Ryzyka.*", r"Swing Factors.*", r"Risks.*"],
    7: [r"Prognoza i typowanie.*", r"Final pick.*", r"Pick.*", r"Wniosek.*"],
}

def _build_header_regex() -> re.Pattern:
    """
    Wykrywa linie-nagłówki sekcji:
    - numerowane: '1) ...' do '7) ...'
    - lub aliasy z HEADER_ALIASES
    (używamy grup nienazwanych, żeby uniknąć kolizji nazw)
    """
    numbered = r"^[ \t]*([1-7])\)[ \t]+(.+?)\s*$"
    alias_parts: List[str] = []
    for pats in HEADER_ALIASES.values():
        for p in pats:
            alias_parts.append(rf"^[ \t]*(?:{p})\s*$")
    alias_block = r"(?:%s)" % "|".join(alias_parts) if alias_parts else r"(?:^\b$)"
    big = rf"(?:{numbered}|{alias_block})"
    return re.compile(big, flags=re.MULTILINE | re.IGNORECASE)

HEADER_RE = _build_header_regex()

def normalize_title(raw: str) -> str:
    """Mapuje znaleziony nagłówek do kanonicznego tytułu z TAB_TITLES."""
    raw_clean = (raw or "").strip()
    m = re.match(r"^[ \t]*([1-7])\)", raw_clean)
    if m:
        idx = int(m.group(1))
        return TAB_TITLES[idx - 1]
    for idx, pats in HEADER_ALIASES.items():
        for p in pats:
            if re.search(p, raw_clean, flags=re.IGNORECASE):
                return TAB_TITLES[idx - 1]
    return raw_clean  # fallback – nieznany nagłówek

def parse_analysis_sections(text: str) -> Dict[str, str]:
    """
    Wejście: cały wklejony tekst analizy.
    Wyjście: dict {kanoniczny_tytuł: markdown treści} dla 7 zakładek.
    """
    if not text or not text.strip():
        return {}

    matches: List[Tuple[int, int, str]] = []
    for m in HEADER_RE.finditer(text):
        start, end = m.span()
        line = m.group(0)
        matches.append((start, end, line))

    if not matches:
        return {TAB_TITLES[0]: text.strip()}

    matches.sort(key=lambda x: x[0])

    slices: Dict[str, str] = {}
    for i, (s, e, line) in enumerate(matches):
        title = normalize_title(line)
        nxt = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[e:nxt].strip("\n ")
        # kosmetyka głowy treści (po nagłówku bywa ":"/"-"/"–")
        body = re.sub(r"^\s*[:\-–]\s*", "", body)
        slices[title] = body

    out: Dict[str, str] = {}
    for t in TAB_TITLES:
        out[t] = slices.get(t, "—")
    return out

# =========================================================
# 2) MODELE DANYCH (Pydantic)
# =========================================================

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

# =========================================================
# 3) IO HELPERS (JSON WeekAnalysis)
# =========================================================

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

# =========================================================
# 4) Parser PS1 ($games = @(...))
# =========================================================

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

# =========================================================
# 5) UI helpers
# =========================================================

def confidence_badge(conf: float | None) -> str:
    """HTML badge dla confidence (kolor wg progu)."""
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
    """Ścieżka do pliku Markdown z długim opisem (HOME_AWAY_w{week}_{season}.md)."""
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
    for canon, alset in NAME_ALIASES.items():
        if (a == canon and b in alset) or (b == canon and a in alset):
            return True
    return False

# =========================================================
# 6) 2024 CSV – PROFILE LOADER
# =========================================================

CSV_2024_PATH = Path("data/processed/season_summary_2024_clean.csv")

FIELDS_MAP = {
    # left: klucz UI -> right: nazwa kolumny w CSV (dopasuj, jeśli masz inne)
    "PPD_off": "ppd_offense",
    "Yds/Drive": "yards_per_drive_offense",
    "EPA/play_off": "epa_per_play_offense",
    "3D SR off": "third_down_sr_offense",
    "RZ EPA off": "redzone_epa_offense",
    "Explosive off": "explosive_plays_offense",
    "Expl Allowed": "explosive_allowed_defense",
    "EPA/play allowed": "epa_per_play_defense",
    "3D SR allowed": "third_down_sr_allowed",
    "RZ EPA allowed": "redzone_epa_allowed",
    "Plays/Drive": "plays_per_drive",
    "Start FP": "start_fp",
    "HiddenY/Drive": "hidden_yards_per_drive",
}

@lru_cache(maxsize=1)
def load_2024_csv(path: str | Path = CSV_2024_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"2024 CSV not found at: {p}")
    df = pd.read_csv(p)
    # Normalizacja nazw kolumn na lower + underscores:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _fmt_pct(v: Optional[float]) -> str:
    if v is None: 
        return "N/A"
    try:
        if 0 <= float(v) <= 1:
            return f"{float(v)*100:.1f}%"
        return f"{float(v):.1f}%"
    except Exception:
        return str(v)

def _fmt_num(v: Optional[float], nd=2) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

def get_team_profile_2024(team_name: str, df: Optional[pd.DataFrame] = None) -> dict:
    """
    Zwraca słownik pól wg FIELDS_MAP dla podanej drużyny.
    Zakładamy kolumnę 'team' lub 'team_name' w CSV.
    """
    df = df if df is not None else load_2024_csv()
    team_col = "team" if "team" in df.columns else ("team_name" if "team_name" in df.columns else None)
    if not team_col:
        raise KeyError("CSV 2024 must contain 'team' or 'team_name' column.")

    row = df[df[team_col].str.lower() == team_name.lower()]
    if row.empty:
        # alias (np. LA Rams)
        for canon, alset in NAME_ALIASES.items():
            if team_name in alset:
                row = df[df[team_col].str.lower() == canon.lower()]
                if not row.empty:
                    break
    if row.empty:
        return {}

    row = row.iloc[0].to_dict()
    out = {}
    for ui_key, csv_col in FIELDS_MAP.items():
        col = csv_col.lower()
        out[ui_key] = row.get(col)
    return out

def build_profile_cards_2024(home_team: str, away_team: str) -> tuple[list[dict], list[dict]]:
    """Zwraca (cards_home, cards_away) – 2 listy kart do renderowania w UI."""
    df = load_2024_csv()
    h = get_team_profile_2024(home_team, df)
    a = get_team_profile_2024(away_team, df)

    def as_cards(d: dict) -> list[dict]:
        order = [
            "PPD_off", "Yds/Drive", "EPA/play_off", "3D SR off", "RZ EPA off", "Explosive off",
            "Expl Allowed", "EPA/play allowed", "3D SR allowed", "RZ EPA allowed",
            "Plays/Drive", "Start FP", "HiddenY/Drive",
        ]
        cards = []
        for k in order:
            val = d.get(k)
            if k in ("3D SR off", "3D SR allowed"):
                disp = _fmt_pct(val)
            elif k in ("PPD_off",):
                disp = _fmt_num(val, nd=2)
            else:
                disp = _fmt_num(val, nd=2)
            cards.append({"name": k, "value": disp})
        return cards

    return as_cards(h), as_cards(a)

# (opcjonalnie) alias zgodny z wcześniejszym app.py
def build_profile_cards_2024_inline(home_team: str, away_team: str) -> tuple[list[dict], list[dict]]:
    return build_profile_cards_2024(home_team, away_team)

def summarize_top_edges(game_obj: Optional[Game], n=3) -> list[str]:
    """Zwraca listę 3 krótkich punktów TL;DR na bazie największych |edges.value|."""
    if not game_obj or not game_obj.edges:
        return []
    edges_sorted = sorted(
        [e for e in game_obj.edges if e.value is not None],
        key=lambda e: abs(e.value),
        reverse=True
    )[:n]
    out = []
    for e in edges_sorted:
        dir_txt = f" → edge **{e.team}**" if e.team else ""
        try:
            val = f"{float(e.value):+,.2f}"
        except Exception:
            val = str(e.value)
        out.append(f"**{e.name}**: {val}{dir_txt}")
    return out
