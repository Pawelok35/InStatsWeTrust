import pandas as pd
import numpy as np
from pathlib import Path

# ======================
# Helpers
# ======================

def _safe_col(df: pd.DataFrame, a: str, b: str | None = None, default=None) -> pd.Series:
    """Zwróć kolumnę a (lub b), a gdy brak – serię z wartością domyślną."""
    if a in df.columns:
        return df[a]
    if b and b in df.columns:
        return df[b]
    if default is not None:
        return pd.Series(default, index=df.index)
    raise KeyError(f"Missing required column: {a} (or {b})")

def _booly(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Rzutowanie na bool:
    - brak kolumny -> False
    - numeric -> != 0
    - string -> {'1','true','t','yes','y'}
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0) != 0
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])

# ======================
# Trend #1: Game Rhythm (Tempo Q4)
# ======================

def game_rhythm_q4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plays per minute w 4Q przy |score_differential| <= 7
    (liczone z różnic zegara w ramach tej samej drużyny i meczu).
    Zwraca: ['team','game_rhythm_q4']
    """
    qtr = pd.to_numeric(_safe_col(df, "qtr", "quarter"), errors="coerce")
    sdiff = pd.to_numeric(_safe_col(df, "score_differential", "score_diff", default=np.nan), errors="coerce")
    posteam = _safe_col(df, "posteam")
    game_id = _safe_col(df, "game_id")
    gsr = pd.to_numeric(_safe_col(df, "game_seconds_remaining", "game_second_remaining", default=np.nan), errors="coerce")

    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    is_play = pt.isin(["run", "pass"])
    is_clean = (~_booly(df, "penalty")) & (~_booly(df, "no_play")) & (~_booly(df, "qb_kneel")) & (~_booly(df, "qb_spike"))

    close_q4 = (qtr == 4) & (sdiff.abs() <= 7)
    filt = close_q4 & is_play & is_clean
    if not filt.any():
        return pd.DataFrame(columns=["team", "game_rhythm_q4"])

    sub = pd.DataFrame({
        "game_id": game_id[filt].values,
        "posteam": posteam[filt].values,
        "play_type": pt[filt].values,
        "gsr": gsr[filt].values,
    }).dropna(subset=["gsr"])

    if sub.empty:
        return pd.DataFrame(columns=["team", "game_rhythm_q4"])

    sub = sub.sort_values(["game_id", "posteam", "gsr"], ascending=[True, True, False])

    same_game = sub["game_id"].eq(sub["game_id"].shift())
    same_team = sub["posteam"].eq(sub["posteam"].shift())
    valid_prev = same_game & same_team

    prev_gsr = sub["gsr"].shift()
    delta = (prev_gsr - sub["gsr"]).where(valid_prev, 0.0).clip(lower=0.0, upper=40.0)

    agg = sub.assign(delta_sec=delta).groupby("posteam").agg(
        plays=("play_type", "size"),
        poss_time_sec=("delta_sec", "sum"),
    ).reset_index()

    agg["poss_time_min"] = agg["poss_time_sec"] / 60.0
    agg["game_rhythm_q4"] = np.where(agg["poss_time_min"] > 0, agg["plays"] / agg["poss_time_min"], np.nan)

    return agg.rename(columns={"posteam": "team"})[["team", "game_rhythm_q4"]]

# ======================
# Trend #2: Play-Calling Variance (Entropy – neutral)
# ======================

def play_calling_entropy_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entropia run/pass w stanie neutralnym:
      - abs(score_differential) <= 7
      - down in {1, 2}
      - yardline_100 in [20, 80]
    Zwraca: ['team','play_call_entropy_neutral','neutral_pass_rate','neutral_plays']
    """
    score_diff = pd.to_numeric(_safe_col(df, "score_differential", "score_diff", default=np.nan), errors="coerce")
    posteam = _safe_col(df, "posteam")
    dn = pd.to_numeric(_safe_col(df, "down"), errors="coerce")
    y100 = pd.to_numeric(_safe_col(df, "yardline_100", default=np.nan), errors="coerce")

    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    is_play = pt.isin(["run", "pass"])
    is_clean = (~_booly(df, "penalty")) & (~_booly(df, "no_play")) & (~_booly(df, "qb_kneel")) & (~_booly(df, "qb_spike"))

    neutral = (score_diff.abs() <= 7) & (dn.isin([1, 2])) & (y100.between(20, 80, inclusive="both"))
    mask = neutral & is_play & is_clean
    if not mask.any():
        return pd.DataFrame(columns=["team", "play_call_entropy_neutral", "neutral_pass_rate", "neutral_plays"])

    base = pd.DataFrame({
        "posteam": posteam[mask].values,
        "play_type": pt[mask].values,
    })

    counts = base.pivot_table(index="posteam", columns="play_type", aggfunc="size", fill_value=0).astype(float)
    for col in ["run", "pass"]:
        if col not in counts.columns:
            counts[col] = 0.0
    counts = counts[["run", "pass"]]
    counts["total"] = counts.sum(axis=1).replace(0, np.nan)

    p_run = counts["run"] / counts["total"]
    p_pass = counts["pass"] / counts["total"]

    def H(p: pd.Series) -> pd.Series:
        p = p.clip(1e-12, 1.0)
        return -(p * np.log2(p))

    entropy = H(p_run).fillna(0) + H(p_pass).fillna(0)

    out = pd.DataFrame({
        "team": counts.index,
        "play_call_entropy_neutral": entropy.astype(float),
        "neutral_pass_rate": p_pass.astype(float),
        "neutral_plays": counts["total"].astype("Int64"),
    }).reset_index(drop=True)

    return out

# ======================
# Trend #3: Drive Momentum (3+ success)
# ======================

def _max_consecutive_true(mask: np.ndarray) -> int:
    """Długość najdłuższej serii True w tablicy bool."""
    if mask.size == 0:
        return 0
    best = cur = 0
    for v in mask.astype(np.int8):
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def drive_momentum_3plus(df: pd.DataFrame) -> pd.DataFrame:
    """
    % drive’ów z serią ≥3 udanych zagrań (success==1 lub epa>0).
    Zwraca: ['team','drive_momentum_3plus','drives_with_3plus','drives_total']
    """
    posteam = _safe_col(df, "posteam")
    game_id = _safe_col(df, "game_id")

    drive_col = "drive" if "drive" in df.columns else ("drive_id" if "drive_id" in df.columns else None)
    if drive_col is None:
        return pd.DataFrame(columns=["team", "drive_momentum_3plus", "drives_with_3plus", "drives_total"])

    if "success" in df.columns:
        succ = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    else:
        epa = pd.to_numeric(_safe_col(df, "epa", default=np.nan), errors="coerce")
        succ = epa > 0

    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    is_play = pt.isin(["run", "pass"])
    is_clean = (~_booly(df, "penalty")) & (~_booly(df, "no_play")) & (~_booly(df, "qb_kneel")) & (~_booly(df, "qb_spike"))
    filt = is_play & is_clean

    qtr = pd.to_numeric(_safe_col(df, "qtr", "quarter"), errors="coerce")
    gsr = pd.to_numeric(_safe_col(df, "game_seconds_remaining", "game_second_remaining", default=np.nan), errors="coerce")

    base = pd.DataFrame({
        "game_id": game_id,
        "team": posteam,
        "drive": df[drive_col],
        "succ": succ,
        "qtr": qtr,
        "gsr": gsr,
    })
    base = base.loc[filt].dropna(subset=["team", "drive", "qtr", "gsr"])
    if base.empty:
        return pd.DataFrame(columns=["team", "drive_momentum_3plus", "drives_with_3plus", "drives_total"])

    base = base.sort_values(["game_id", "team", "drive", "qtr", "gsr"], ascending=[True, True, True, True, False])

    def per_drive(group: pd.DataFrame) -> pd.Series:
        max_streak = _max_consecutive_true(group["succ"].to_numpy())
        return pd.Series({
            "has_3plus": int(max_streak >= 3),
            "plays": int(group.shape[0]),
        })

    drive_stats = base.groupby(["game_id", "team", "drive"], as_index=False).apply(per_drive, include_groups=False)

    team_agg = drive_stats.groupby("team").agg(
        drives_with_3plus=("has_3plus", "sum"),
        drives_total=("has_3plus", "size"),
    ).reset_index()

    team_agg["drive_momentum_3plus"] = np.where(
        team_agg["drives_total"] > 0,
        team_agg["drives_with_3plus"] / team_agg["drives_total"],
        np.nan
    )

    return team_agg[["team", "drive_momentum_3plus", "drives_with_3plus", "drives_total"]]

def field_flip_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Field Flip Efficiency:
    Średnia zmiana pozycji boiska po puncie:
      flip = opponent_start_y100 (następna akcja rywala) - pre_punt_y100 (pozycja puntera)
    Dodatnia wartość = lepszy flip (rywal zaczyna dalej od naszej endzone).
    Zwraca: ['team','field_flip_eff','punts_tracked']
    """
    # Bezpieczne kolumny
    game_id = _safe_col(df, "game_id")
    posteam = _safe_col(df, "posteam")
    pt_raw = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    y100 = pd.to_numeric(_safe_col(df, "yardline_100", default=np.nan), errors="coerce")

    # Filtr: puncie i czyste akcje
    is_punt = pt_raw.eq("punt")
    is_clean = (~_booly(df, "penalty")) & (~_booly(df, "no_play"))  # zostawiamy prosto: bez kar i no_play

    punts = df.loc[is_punt & is_clean, ["game_id", "posteam"]].copy()
    if punts.empty:
        return pd.DataFrame(columns=["team", "field_flip_eff", "punts_tracked"])

    punts["idx"] = punts.index
    punts["pre_punt_y100"] = y100.loc[punts["idx"]].values

    # Przygotuj kolumny potrzebne do szukania następnej akcji
    next_posteam = posteam.shift(-1)
    next_game = game_id.shift(-1)
    next_y100 = pd.to_numeric(y100.shift(-1), errors="coerce")

    # Metoda: dla KAŻDEGO punta idziemy w dół indeksu aż znajdziemy:
    # - ten sam mecz
    # - posteam != punting_team
    # - akcję "czystą" (bez penalty/no_play), najlepiej normalną ofensywną
    # Dla bezpieczeństwa robimy pętlę — liczba puntów na sezon jest umiarkowana, więc OK.
    opponent_starts = []
    for _, row in punts.iterrows():
        i = int(row["idx"])
        g = row["game_id"]
        team = row["posteam"]
        opp_start = np.nan

        # Szukaj w następnych wierszach w tym samym meczu
        j = i + 1
        while j < len(df) and game_id.iat[j] == g:
            # zmiana posiadania?
            if posteam.iat[j] != team:
                # sprawdź czystość
                if (not _booly(df, "penalty").iat[j]) and (not _booly(df, "no_play").iat[j]):
                    opp_start = pd.to_numeric(y100.iat[j], errors="coerce")
                    break
            j += 1

        opponent_starts.append(opp_start)

    punts["opp_start_y100"] = opponent_starts
    punts = punts.dropna(subset=["pre_punt_y100", "opp_start_y100"])

    if punts.empty:
        return pd.DataFrame(columns=["team", "field_flip_eff", "punts_tracked"])

    punts["flip"] = punts["opp_start_y100"] - punts["pre_punt_y100"]

    out = punts.groupby("posteam").agg(
        field_flip_eff=("flip", "mean"),
        punts_tracked=("flip", "size"),
    ).reset_index().rename(columns={"posteam": "team"})

    return out[["team", "field_flip_eff", "punts_tracked"]]

def two_minute_drill_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skuteczność w ostatnich 2 minutach każdej połowy (Q2 i Q4):
      - filtr: qtr ∈ {2, 4} i quarter_seconds_remaining <= 120
      - tylko czyste akcje run/pass
    Zwraca: ['team','twomin_success_rate','twomin_epa_play','twomin_plays']
    """
    # Kolumny bazowe
    posteam = _safe_col(df, "posteam")
    qtr = pd.to_numeric(_safe_col(df, "qtr", "quarter"), errors="coerce")
    gsr = pd.to_numeric(_safe_col(df, "game_seconds_remaining", "game_second_remaining", default=np.nan), errors="coerce")
    epa = pd.to_numeric(_safe_col(df, "epa", default=np.nan), errors="coerce")

    # seconds remaining w danej kwarcie: qsr = gsr - (4 - qtr)*900
    qsr = gsr - (4 - qtr) * 900

    # Normalne, czyste zagrania
    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    is_play = pt.isin(["run", "pass"])
    is_clean = (~_booly(df, "penalty")) & (~_booly(df, "no_play")) & (~_booly(df, "qb_kneel")) & (~_booly(df, "qb_spike"))

    # Filtr 2-min: Q2 i Q4 oraz ≤120 s do końca kwarty
    last2 = qtr.isin([2, 4]) & (qsr <= 120)

    # Sukces: preferuj 'success', inaczej epa>0
    if "success" in df.columns:
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    else:
        success = epa > 0

    mask = last2 & is_play & is_clean

    if not mask.any():
        return pd.DataFrame(columns=["team", "twomin_success_rate", "twomin_epa_play", "twomin_plays"])

    base = pd.DataFrame({
        "team": posteam[mask].values,
        "success": success[mask].values,
        "epa": epa[mask].values,
    }).dropna(subset=["team"])

    if base.empty:
        return pd.DataFrame(columns=["team", "twomin_success_rate", "twomin_epa_play", "twomin_plays"])

    agg = base.groupby("team").agg(
        twomin_success_rate=("success", "mean"),
        twomin_epa_play=("epa", "mean"),
        twomin_plays=("success", "size"),
    ).reset_index()

    return agg[["team", "twomin_success_rate", "twomin_epa_play", "twomin_plays"]]


# ======================
# Builder
# ======================

def build_hidden_trends(pbp_path: str, out_path: str, season: int):
    """
    Build Hidden Trends metrics per team.
    Input: play-by-play parquet file for a given season.
    Output: team_hidden_trends_{season}.csv
    """
    print(f"📊 Building Hidden Trends for season {season}...")

    df = pd.read_parquet(pbp_path)
    print(f"Loaded {len(df):,} plays from {pbp_path}")

    # Lista zespołów (gwarantuje wiersze nawet przy pustych metrykach)
    teams = df["posteam"].dropna().unique().tolist() if "posteam" in df.columns else []
    base = pd.DataFrame({"team": sorted(teams)}) if teams else pd.DataFrame({"team": pd.Series(dtype=object)})

    # Trendy
    try:
        t1 = game_rhythm_q4(df)
    except Exception as e:
        print(f"[WARN] game_rhythm_q4 failed: {e}")
        t1 = pd.DataFrame(columns=["team", "game_rhythm_q4"])

    try:
        t2 = play_calling_entropy_neutral(df)
    except Exception as e:
        print(f"[WARN] play_calling_entropy_neutral failed: {e}")
        t2 = pd.DataFrame(columns=["team", "play_call_entropy_neutral", "neutral_pass_rate", "neutral_plays"])

    try:
        t3 = drive_momentum_3plus(df)
    except Exception as e:
        print(f"[WARN] drive_momentum_3plus failed: {e}")
        t3 = pd.DataFrame(columns=["team", "drive_momentum_3plus", "drives_with_3plus", "drives_total"])
  
    # === Trend 4: Field Flip Efficiency ===
    try:
        t4 = field_flip_efficiency(df)
    except Exception as e:
        print(f"[WARN] field_flip_efficiency failed: {e}")
        t4 = pd.DataFrame(columns=["team", "field_flip_eff", "punts_tracked"])

    # === Trend 6: Two-Minute Drill Efficiency ===
    try:
        t5 = two_minute_drill_efficiency(df)
    except Exception as e:
        print(f"[WARN] two_minute_drill_efficiency failed: {e}")
        t5 = pd.DataFrame(columns=["team", "twomin_success_rate", "twomin_epa_play", "twomin_plays"])

    out_df = base.merge(t1, on="team", how="left") \
                .merge(t2, on="team", how="left") \
                .merge(t3, on="team", how="left") \
                .merge(t4, on="team", how="left") \
                .merge(t5, on="team", how="left")

    # Kolejność kolumn (uwzględnia Trend #3)
    expected_cols = [
        "team",
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
    for col in expected_cols:
        if col not in out_df.columns:
            out_df[col] = pd.Series(dtype=float)
    out_df = out_df[expected_cols]

    print("→ out_df columns:", list(out_df.columns))
    print("→ out_df shape:", out_df.shape)
    try:
        print("→ sample:\n", out_df.head(8).to_string(index=False))
    except Exception:
        pass

    out_path = Path(out_path)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved hidden trends to {out_path}")
    
        # 7) Dopisanie tygodnia do historii drużyn (Hidden Trends)
    from etl.utils_team_history import update_team_history
    WEEK = 6  # Ustaw np. 6, jeśli uruchamiasz konkretny tydzień
    update_team_history(out_df, season=season, week=WEEK or 0, store="data/processed/teams")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--in_pbp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    build_hidden_trends(args.in_pbp, args.out, args.season)


