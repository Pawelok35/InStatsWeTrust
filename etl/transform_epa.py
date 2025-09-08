# etl/transform_epa.py
from pathlib import Path
import re
import pandas as pd
import numpy as np

def find_latest_pbp_csv(raw_dir: Path, season: int) -> Path:
    """Znajdź najnowszy plik pbp_<SEASON>_week_<N>.csv w data/raw."""
    files = list(raw_dir.glob(f"pbp_{season}_week_*.csv"))
    if not files:
        raise FileNotFoundError(f"Brak plików pbp_{season}_week_*.csv w {raw_dir}")
    def week_num(p: Path) -> int:
        m = re.search(r"week_(\d+)", p.name)
        return int(m.group(1)) if m else -1
    return max(files, key=week_num)

def safe_rate(s: pd.Series) -> float:
    denom = s.size
    return float(s.sum() / denom) if denom else 0.0

def main():
    # ===== Parametry =====
    SEASON = 2024  # zostajemy przy pełnym, zamkniętym sezonie 2024

    # ===== Ścieżki =====
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ===== Wczytanie najnowszego pliku PBP dla zadanego sezonu =====
    pbp_path = find_latest_pbp_csv(raw_dir, SEASON)
    week_match = re.search(r"week_(\d+)", pbp_path.name)
    latest_week = int(week_match.group(1)) if week_match else None

    print(f"📥 Wczytuję: {pbp_path.name}")
    pbp = pd.read_csv(pbp_path, low_memory=False)

    # Upewniamy się, że pola liczbowe są liczbowe
    for col in ["epa", "yards_gained", "down"]:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

    # Filtr: tylko akcje ofensywne z przypisaną drużyną ataku
    plays = pbp[pbp["posteam"].notna()].copy()

    # Dodatkowe cechy
    plays["success"] = plays["epa"].fillna(0) > 0
    plays["explosive"] = plays["yards_gained"].fillna(0) >= 20
    plays["early_down"] = plays["down"].isin([1, 2])
    plays["third_down_success"] = (plays["down"] == 3) & (plays["epa"].fillna(0) > 0)

    # ===== Podsumowanie ofensywne per team (posteam) =====
    off = (
        plays.groupby("posteam")
        .agg(
            plays=("epa", "size"),
            avg_epa=("epa", "mean"),
            median_epa=("epa", "median"),
            success_rate=("success", safe_rate),
            explosive_plays=("explosive", "sum"),
            explosive_rate=("explosive", safe_rate),
            total_yards=("yards_gained", "sum"),
            early_down_epa=("epa", lambda s: s[plays.loc[s.index, "early_down"]].mean()),
            third_down_sr=("third_down_success", safe_rate),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
        .sort_values(["avg_epa", "success_rate"], ascending=[False, False])
    )

    # Zaokrąglenia dla czytelności
    for c in ["avg_epa", "median_epa", "success_rate", "explosive_rate", "early_down_epa", "third_down_sr"]:
        if c in off.columns:
            off[c] = off[c].astype(float).round(4)

    off["week"] = latest_week
    off["season"] = SEASON

    out_off = processed_dir / f"epa_offense_summary_{SEASON}_week_{latest_week}.csv"
    off.to_csv(out_off, index=False)

    print(f"✅ Zapisano: {out_off}")
    print(off.head(10).to_string(index=False))

    # ===== (Opcjonalnie) proste podsumowanie defensywne per defteam =====
    if "defteam" in plays.columns:
        defn = (
            plays.groupby("defteam")
            .agg(
                plays=("epa", "size"),
                avg_epa_allowed=("epa", "mean"),  # im niżej, tym lepiej dla obrony
                success_rate_allowed=("success", safe_rate),
                explosive_allowed=("explosive", "sum"),
                explosive_rate_allowed=("explosive", safe_rate),
                yards_allowed=("yards_gained", "sum"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
            .sort_values(["avg_epa_allowed", "success_rate_allowed"], ascending=[True, True])
        )

        for c in ["avg_epa_allowed", "success_rate_allowed", "explosive_rate_allowed"]:
            defn[c] = defn[c].astype(float).round(4)

        defn["week"] = latest_week
        defn["season"] = SEASON

        out_def = processed_dir / f"epa_defense_summary_{SEASON}_week_{latest_week}.csv"
        defn.to_csv(out_def, index=False)
        print(f"✅ Zapisano: {out_def}")
        print(defn.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
