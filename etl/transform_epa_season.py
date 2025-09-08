from pathlib import Path
import pandas as pd
import numpy as np

SEASON = 2024

def safe_rate(s: pd.Series) -> float:
    denom = s.size
    return float(s.sum() / denom) if denom else 0.0

def load_pbp_all(raw_dir: Path, season: int) -> pd.DataFrame:
    all_path = raw_dir / f"pbp_{season}_all.csv"
    if all_path.exists():
        return pd.read_csv(all_path, low_memory=False)
    # fallback: łącz z tygodni
    parts = []
    for wk in range(1, 19):
        p = raw_dir / f"pbp_{season}_week_{wk}.csv"
        if p.exists():
            parts.append(pd.read_csv(p, low_memory=False))
    if not parts:
        raise FileNotFoundError(f"Brak danych PBP dla {season} w {raw_dir}")
    return pd.concat(parts, ignore_index=True)

def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Wczytuję PBP scalone dla {SEASON}…")
    pbp = load_pbp_all(raw_dir, SEASON)

    # Normalizacja typów
    for col in ["epa", "yards_gained", "down", "week"]:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

    # Tylko akcje z przypisaną drużyną ataku
    plays = pbp[pbp["posteam"].notna()].copy()
    plays["success"] = plays["epa"].fillna(0) > 0
    plays["explosive"] = plays["yards_gained"].fillna(0) >= 20
    plays["early_down"] = plays["down"].isin([1, 2])
    plays["third_down_success"] = (plays["down"] == 3) & (plays["epa"].fillna(0) > 0)

    # =======================
    # 1) OFFENSE – sezon per team
    # =======================
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
    for c in ["avg_epa","median_epa","success_rate","explosive_rate","early_down_epa","third_down_sr"]:
        off[c] = off[c].astype(float).round(4)
    off["season"] = SEASON
    out_off = processed_dir / f"epa_offense_summary_{SEASON}_season.csv"
    off.to_csv(out_off, index=False)
    print(f"✅ Zapisano: {out_off}")

    # =======================
    # 2) DEFENSE – sezon per team
    # =======================
    if "defteam" in plays.columns:
        defn = (
            plays.groupby("defteam")
            .agg(
                plays=("epa", "size"),
                avg_epa_allowed=("epa", "mean"),           # im niżej, tym lepiej
                median_epa_allowed=("epa", "median"),
                success_rate_allowed=("success", safe_rate),
                explosive_allowed=("explosive", "sum"),
                explosive_rate_allowed=("explosive", safe_rate),
                yards_allowed=("yards_gained", "sum"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
            .sort_values(["avg_epa_allowed", "success_rate_allowed"], ascending=[True, True])
        )
        for c in ["avg_epa_allowed","median_epa_allowed","success_rate_allowed","explosive_rate_allowed"]:
            defn[c] = defn[c].astype(float).round(4)
        defn["season"] = SEASON
        out_def = processed_dir / f"epa_defense_summary_{SEASON}_season.csv"
        defn.to_csv(out_def, index=False)
        print(f"✅ Zapisano: {out_def}")
    else:
        defn = pd.DataFrame(columns=["team"])  # pusty fallback

    # =======================
    # 3) PER-GAME BOX + POWER SIGNAL
    # =======================
    # Mapowanie game_id -> home/away
    game_teams = (
        pbp.loc[pbp["home_team"].notna() & pbp["away_team"].notna(), ["game_id","home_team","away_team"]]
        .drop_duplicates(subset=["game_id"])
    )

    # Offense per game (dla każdej drużyny w meczu)
    per_game_off = (
        plays.groupby(["game_id","posteam"])
        .agg(
            plays=("epa", "size"),
            offense_epa=("epa", "mean"),
            success_rate=("success", safe_rate),
            explosive_rate=("explosive", safe_rate),
            yards=("yards_gained", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # dołącz info home/away i pivot do jednego wiersza per mecz
    per_game = per_game_off.merge(game_teams, on="game_id", how="left")
    per_game["is_home"] = per_game.apply(lambda r: 1 if r["team"] == r["home_team"] else 0, axis=1)

    # rozdziel na home/away i sklej kolumny w jednym rekordzie per game_id
    home = per_game[per_game["is_home"] == 1].copy()
    away = per_game[per_game["is_home"] == 0].copy()

    # wybieramy istotne kolumny i dodajemy sufiksy
    home = home[["game_id","team","offense_epa","success_rate","explosive_rate","yards","plays"]]
    away = away[["game_id","team","offense_epa","success_rate","explosive_rate","yards","plays"]]
    home = home.add_prefix("home_"); home = home.rename(columns={"home_game_id":"game_id"})
    away = away.add_prefix("away_"); away = away.rename(columns={"away_game_id":"game_id"})

    matchup = home.merge(away, on="game_id", how="inner")

    # Power Signal – prosty sygnał: (offense_epa_home - offense_epa_allowed_home)
    # Tu zamiast allowed per-game (trudne z PBP) użyjemy sezonowego def summary (stabilniejsze):
    power = defn[["team","avg_epa_allowed"]].rename(columns={"team":"team_def","avg_epa_allowed":"def_allowed"})
    matchup = matchup.merge(power, left_on="home_team", right_on="team_def", how="left").drop(columns=["team_def"])
    matchup = matchup.rename(columns={"def_allowed":"home_def_allowed"})
    matchup = matchup.merge(power, left_on="away_team", right_on="team_def", how="left").drop(columns=["team_def"])
    matchup = matchup.rename(columns={"def_allowed":"away_def_allowed"})

    matchup["home_power_signal"] = matchup["home_offense_epa"] - matchup["home_def_allowed"]
    matchup["away_power_signal"] = matchup["away_offense_epa"] - matchup["away_def_allowed"]

    out_match = processed_dir / f"matchups_{SEASON}_season.csv"
    matchup.to_csv(out_match, index=False)
    print(f"✅ Zapisano: {out_match}")

    # =======================
    # 4) OFF + DEF JOIN – ranking sezonowy (wczesny Power Signal)
    # =======================
    if not defn.empty:
        joined = off.merge(defn, on="team", how="inner")
        joined["power_signal"] = joined["avg_epa"] - joined["avg_epa_allowed"]
        joined = joined.sort_values("power_signal", ascending=False)
        out_rank = processed_dir / f"power_signal_{SEASON}_season.csv"
        joined.to_csv(out_rank, index=False)
        print(f"🏆 Zapisano ranking: {out_rank}")
        print(joined.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
