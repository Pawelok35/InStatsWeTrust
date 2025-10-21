# etl/utils_team_history.py
# Python 3.11+
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import pandas as pd


NFL_TEAMS_3 = {
    # Zbiór skrótów 3-literowych (w razie potrzeby możesz rozszerzyć/zmienić)
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
    "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI",
    "PIT","SEA","SF","TB","TEN","WAS"
}

# Canonicalization map for legacy or ambiguous aliases
TEAM_ALIASES: Dict[str, Optional[str]] = {
    # Explicit mappings
    "LA": "LAR",              # ambiguous -> default to Rams per project convention
    "LA-CHARGERS": "LAC",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "STL": "LAR",            # legacy Rams
    "ST. LOUIS RAMS": "LAR",
    "SD": "LAC",             # legacy Chargers
    "SAN DIEGO CHARGERS": "LAC",
    "OAK": "LV",             # legacy Raiders
    "OAKLAND RAIDERS": "LV",
    "WSH": "WAS",            # legacy Washington
    "REDSKINS": "WAS",
    "JAC": "JAX",
    "NAN": None,              # invalid placeholder -> drop
}


def normalize_team_code(code: str | None) -> Optional[str]:
    """Return canonical 2-3 letter team code or None if invalid.

    - Uppercases and strips whitespace
    - Maps known aliases (e.g., LA -> LAR, SD -> LAC, OAK -> LV)
    - Returns None for invalid placeholders (e.g., NAN)
    """
    if code is None:
        return None
    c = str(code).strip().upper()
    if c in NFL_TEAMS_3:
        return c
    if c in TEAM_ALIASES:
        mapped = TEAM_ALIASES[c]
        return mapped if (mapped in NFL_TEAMS_3 or mapped is None) else mapped
    # Sometimes full names slip in; try a minimal full-name mapping
    full_to_abbr = {
        "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL",
        "BALTIMORE RAVENS": "BAL", "BUFFALO BILLS": "BUF",
        "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
        "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE",
        "DALLAS COWBOYS": "DAL", "DENVER BRONCOS": "DEN",
        "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
        "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND",
        "JACKSONVILLE JAGUARS": "JAX", "KANSAS CITY CHIEFS": "KC",
        "LOS ANGELES CHARGERS": "LAC", "LOS ANGELES RAMS": "LAR",
        "LAS VEGAS RAIDERS": "LV", "MIAMI DOLPHINS": "MIA",
        "MINNESOTA VIKINGS": "MIN", "NEW ENGLAND PATRIOTS": "NE",
        "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
        "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI",
        "PITTSBURGH STEELERS": "PIT", "SEATTLE SEAHAWKS": "SEA",
        "SAN FRANCISCO 49ERS": "SF", "TAMPA BAY BUCCANEERS": "TB",
        "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WAS",
    }
    if c in full_to_abbr:
        return full_to_abbr[c]
    return c if c in NFL_TEAMS_3 else None


def _ensure_store(store: str | Path) -> Path:
    p = Path(store)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _coerce_types(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    # Wymuszamy obecność i typy kluczowych kolumn
    df = df.copy()
    if "team" not in df.columns:
        raise ValueError("Brak kolumny 'team' w df_week_agg (oczekiwane 32×N z kolumną 'team').")

    df["team"] = df["team"].astype(str).str.upper()
    df["season"] = int(season)
    df["week"] = int(week)
    return df


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Kolejność: season, week, team, … (reszta w stabilnej kolejności alfabetycznej)
    base = ["season", "week", "team"]
    rest = [c for c in df.columns if c not in base]
    # Zachowaj stabilność: sort alfabetyczny, żeby między builderami nie „skakało”
    rest_sorted = sorted(rest)
    return df[base + rest_sorted]


def _union_columns(existing: pd.DataFrame, incoming: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Ujednolicony zestaw kolumn; brakujące wypełniamy NaN/None
    all_cols = sorted(set(existing.columns).union(incoming.columns))
    def _align(df: pd.DataFrame) -> pd.DataFrame:
        for c in all_cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[all_cols]
    return _align(existing), _align(incoming)


def _validate_teams(df: pd.DataFrame) -> None:
    unknown = sorted(set(df["team"]) - NFL_TEAMS_3)
    if unknown:
        # Nie blokujemy wykonania, ale podajemy ostrzeżenie (np. dla LAR/STL legacy itp.)
        print(f"[utils_team_history] Uwaga: niewspierane kody team: {unknown}")


def update_team_history(
    df_week_agg: pd.DataFrame,
    season: int,
    week: int,
    store: str | Path = "data/processed/teams",
) -> List[Path]:
    """
    Dopina tygodniowe metryki 32×N do plików {TEAM}.csv (append + dedupe po (season, week)).
    ...
    """

    # 🔒 Bezpiecznik: nie zapisuj placeholdera week<=0
    if int(week) <= 0:
        print("[utils_team_history] Pomijam zapis: week <= 0 (placeholder). Ustaw poprawny WEEK.")
        return []

    out_paths: List[Path] = []
    store_path = _ensure_store(store)

    df = _coerce_types(df_week_agg, season, week)
    _validate_teams(df)

    # Filtr: zostaw tylko prawidłowe kody NFL (usuń NaN i aliasy typu 'LA')
    df = df[df["team"].isin(NFL_TEAMS_3)].copy()

    df = _order_columns(df)

    # Dodatkowe sanity...
    df = df.sort_index()
    df = df.drop_duplicates(subset=["team"], keep="last")

    for team_code, chunk in df.groupby("team", as_index=False):
        team_file = store_path / f"{team_code}.csv"

        if team_file.exists():
            existing = pd.read_csv(team_file)
            existing, chunk_aligned = _union_columns(existing, chunk)
            combined = pd.concat(
                [existing, chunk_aligned.dropna(axis=1, how="all")],
                ignore_index=True,
                copy=False
            )

        else:
            combined = chunk.copy()

        combined["season"] = combined["season"].astype(int)
        combined["week"] = combined["week"].astype(int)
        combined["team"] = combined["team"].astype(str).str.upper()

        combined = combined.drop_duplicates(subset=["season", "week"], keep="last")
        combined = combined.sort_values(["season", "week"], kind="mergesort")

        combined = _order_columns(combined)
        combined.to_csv(team_file, index=False)
        out_paths.append(team_file)

    print(f"[utils_team_history] Zapisano/odświeżono {len(out_paths)} plików w {store_path}")
    return out_paths
