"""
Clutch Index – metrics + composite score

Inputs: play-by-play DataFrame with at least columns (nflverse-style):
  - season, game_id
  - posteam (possession team), defteam (defense team)
  - wpa (win probability added per play, float in [-1,1])
  - wp  (pre-play win probability for posteam, 0..1)
  - game_seconds_remaining (0..3600)
  - score_differential (posteam score – defteam score) BEFORE the play
  - down, ydstogo, first_down (bool) – optional, improves 3rd/4th conv%
  - yardline_100 (yards to end zone for offense) – for Red Zone (<=20)
  - touchdown (bool), field_goal_result ("made"/"missed"/None)
  - drive (numeric id within game) or drive_id – to group drives

Output (per team-season):
  season, team,
  clutch_off_wpa_play,
  clutch_def_wpa_play_allowed,
  clutch_off_success_rate,
  clutch_off_3rd4th_conv,
  clutch_off_rz_td_pct,
  clutch_ppd,
  clutch_index_raw,
  clutch_index_0_100

Design notes
------------
"Clutch" window is defined with two complementary gates:
  G1 (time/score gate): last 15 min of game (<= 900s) AND one-score game (|diff| <= 8)
  G2 (leverage gate): medium-volatility state 20% <= wp <= 80% within last 30 min (<= 1800s)
A play is clutch if G1 OR G2 is true.

Composite (0–100):
  z-standardize components across teams, then weighted sum (higher is better):
    +40%: Off. Clutch WPA/play
    +20%: Off. Clutch 3rd/4th Conv%
    +20%: Off. Clutch Red Zone TD%
    +20%: DEF clutch prevention = (− Def. Clutch WPA/play)
  Finally, min–max scale to 0–100 across teams in the season.

Robustness: if a component has < N_min attempts (default 8 plays / 5 attempts),
we back off to league average for that season to reduce noise.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ClutchConfig:
    time_gate_seconds: int = 900         # last 15 minutes
    leverage_gate_seconds: int = 1800    # last 30 minutes
    one_score_margin: int = 8
    wp_low: float = 0.20
    wp_high: float = 0.80
    min_plays_for_rates: int = 8
    min_attempts_for_rates: int = 5
    weights: dict = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "off_wpa_play": 0.40,
                "third_fourth_conv": 0.20,
                "rz_td_pct": 0.20,
                "def_wpa_play_neg": 0.20,
            }


def _derive_missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # game_seconds_remaining
    if "game_seconds_remaining" not in df.columns:
        # próbujemy z qtr + sec pozostale w kwarcie
        qtr_col = None
        for cand in ["qtr", "quarter", "quarter_number"]:
            if cand in df.columns:
                qtr_col = cand
                break
        qsec_col = None
        for cand in ["quarter_seconds_remaining", "qtr_seconds_remaining", "sec_in_quarter", "secs_remaining_in_quarter"]:
            if cand in df.columns:
                qsec_col = cand
                break
        if qtr_col is not None and qsec_col is not None:
            # Q1..Q4 po 900s (OT pominiemy lub traktujemy jak Q5=900)
            q = pd.to_numeric(df[qtr_col], errors="coerce").fillna(0).astype(int)
            q = q.clip(lower=1, upper=5)
            secs_in_q = pd.to_numeric(df[qsec_col], errors="coerce").fillna(0)
            # czas pozostały w meczu: (pozostałe pełne kwarty * 900) + sekundy w bieżącej kwarcie
            remaining_full_quarters = (4 - q).clip(lower=0)
            df["game_seconds_remaining"] = remaining_full_quarters * 900 + secs_in_q

    # score_differential (posteam - defteam)
    if "score_differential" not in df.columns:
        # spróbujemy z 'posteam_score' i 'defteam_score'
        if "posteam_score" in df.columns and "defteam_score" in df.columns:
            df["score_differential"] = pd.to_numeric(df["posteam_score"], errors="coerce").fillna(0) - \
                                       pd.to_numeric(df["defteam_score"], errors="coerce").fillna(0)
        # alternatywa: home/away + kto ma piłkę
        elif {"home_team","away_team","home_score","away_score","posteam"}.issubset(df.columns):
            # ustal, czy posteam == home_team
            is_home = df["posteam"] == df["home_team"]
            home_score = pd.to_numeric(df["home_score"], errors="coerce").fillna(0)
            away_score = pd.to_numeric(df["away_score"], errors="coerce").fillna(0)
            df["score_differential"] = np.where(is_home, home_score - away_score, away_score - home_score)

    # yardline_100 — przyda się do RZ; jeśli brak, zrobimy fallback
    if "yardline_100" not in df.columns:
        for cand in ["yards_to_goalline", "yardline_100_off", "distance_to_endzone"]:
            if cand in df.columns:
                df["yardline_100"] = pd.to_numeric(df[cand], errors="coerce")
                break

    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal: season, game_id, posteam, defteam.
    Spróbujemy doderive'ować brakujące pola, ale już NIE będziemy rzucać błędu,
    jeśli nie uda się uzyskać 'game_seconds_remaining' lub 'score_differential'.
    add_clutch_flag() poradzi sobie z fallbackami (Q4-only, bez progu 1-score).
    """
    required_min = ["season", "game_id", "posteam", "defteam"]
    missing_min = [c for c in required_min if c not in df.columns]
    if missing_min:
        raise ValueError(f"Missing required PBP columns: {missing_min} (minimal set)")

    # spróbujmy wzbogacić dane o przydatne kolumny
    df = _derive_missing_fields(df).copy()

    # dodatkowe próby parsowania zegara, jeśli brak 'game_seconds_remaining'
    if "game_seconds_remaining" not in df.columns:
        # poszukaj typowych nazw zegara
        clock_col = None
        for cand in ["clock", "time", "game_clock", "clock_display"]:
            if cand in df.columns:
                clock_col = cand
                break
        qtr_col = None
        for cand in ["qtr", "quarter", "quarter_number"]:
            if cand in df.columns:
                qtr_col = cand
                break
        if clock_col is not None and qtr_col is not None:
            # clock w formacie "MM:SS"
            def _parse_clock(s):
                try:
                    if isinstance(s, str) and ":" in s:
                        m, s = s.split(":")
                        return int(m) * 60 + int(s)
                except Exception:
                    return np.nan
                return np.nan
            secs_in_q = pd.to_numeric(df[clock_col].map(_parse_clock), errors="coerce")
            q = pd.to_numeric(df[qtr_col], errors="coerce").fillna(0).astype(int).clip(lower=1, upper=5)
            remaining_full_quarters = (4 - q).clip(lower=0)
            df["game_seconds_remaining"] = remaining_full_quarters * 900 + secs_in_q.fillna(0)

    return df


def add_clutch_flag(df: pd.DataFrame, cfg: ClutchConfig) -> pd.DataFrame:
    df = _ensure_columns(df)

    # Czy mamy dokładny czas?
    has_time = "game_seconds_remaining" in df.columns
    # Czy mamy kwartę? (fallback Q4-only)
    has_qtr = any(c in df.columns for c in ["qtr", "quarter", "quarter_number"])
    if has_qtr:
        qtr_col = next(c for c in ["qtr", "quarter", "quarter_number"] if c in df.columns)
        q = pd.to_numeric(df[qtr_col], errors="coerce").fillna(0).astype(int)
        in_last_period = (q >= 4)  # Q4 (OT też traktujemy jako clutch)
    else:
        in_last_period = pd.Series(False, index=df.index)

    # Czy mamy różnicę punktów?
    has_score = "score_differential" in df.columns
    if has_score:
        abs_diff = df["score_differential"].abs()
        score_ok = abs_diff <= cfg.one_score_margin
    else:
        score_ok = pd.Series(True, index=df.index)  # jeśli nie wiemy, nie ograniczamy do 1-score

    # Budujemy bramkę czasu:
    if has_time:
        in_last_15 = df["game_seconds_remaining"] <= cfg.time_gate_seconds
    elif has_qtr:
        # fallback: cała Q4 jako "ostatnie 15 min"
        in_last_15 = in_last_period
    else:
        # brak jakiejkolwiek informacji o czasie → nie włączamy time gate
        in_last_15 = pd.Series(False, index=df.index)

    time_gate = in_last_15 & score_ok

    # Leverage gate tylko jeśli mamy WP
    if "wp" in df.columns:
        leverage_gate = ( (has_time and (df["game_seconds_remaining"] <= cfg.leverage_gate_seconds)) | (~has_time & in_last_period) ) & \
                        (df["wp"].between(cfg.wp_low, cfg.wp_high))
    else:
        leverage_gate = pd.Series(False, index=df.index)

    df["is_clutch"] = (time_gate | leverage_gate).astype(bool)
    return df
    df = _ensure_columns(df)

    abs_diff = df["score_differential"].abs()
    time_gate = (df["game_seconds_remaining"] <= cfg.time_gate_seconds) & (abs_diff <= cfg.one_score_margin)

    # leverage_gate używamy tylko jeśli 'wp' istnieje
    if "wp" in df.columns:
        leverage_gate = (df["game_seconds_remaining"] <= cfg.leverage_gate_seconds) & (df["wp"].between(cfg.wp_low, cfg.wp_high))
    else:
        leverage_gate = pd.Series(False, index=df.index)

    df["is_clutch"] = (time_gate | leverage_gate).astype(bool)
    return df



def _play_success(df: pd.DataFrame) -> pd.Series:
    """Binary success: EPA>0 if available, else WPA>0 fallback, else touchdown/first_down heuristics."""
    if "epa" in df.columns:
        return (df["epa"] > 0).astype(int)
    # Fallback to WPA sign if no EPA
    if "wpa" in df.columns:
        return (df["wpa"] > 0).astype(int)
    # Last resort: touchdown or first down gained flags if present
    cols = [c for c in ["touchdown","first_down"] if c in df.columns]
    if cols:
        return df[cols].any(axis=1).astype(int)
    return pd.Series(0, index=df.index)


def _third_fourth_attempt(df: pd.DataFrame) -> pd.Series:
    if "down" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["down"].isin([3,4])


def _third_fourth_converted(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit first_down flag; fallback to new series by yards gained >= ydstogo
    converted = pd.Series(False, index=df.index)
    if "first_down" in df.columns:
        converted = df["first_down"].fillna(False).astype(bool)
    elif "ydstogo" in df.columns and "yards_gained" in df.columns:
        converted = (df["yards_gained"] >= df["ydstogo"]).fillna(False)
    return converted


def _red_zone(df: pd.DataFrame) -> pd.Series:
    return (df["yardline_100"] <= 20)


def _made_fg(df: pd.DataFrame) -> pd.Series:
    if "field_goal_result" in df.columns:
        return (df["field_goal_result"] == "made")
    return pd.Series(False, index=df.index)


def _drive_id(df: pd.DataFrame) -> pd.Series:
    for cand in ["drive_id","drive"]:
        if cand in df.columns:
            return df[cand]
    # fallback: create synthetic within game by cumulative count of posteam changes
    return (
        df.groupby(["game_id"])  # per game
          .apply(lambda g: (g["posteam"] != g["posteam"].shift()).cumsum())
          .reset_index(level=0, drop=True)
    )


def _points_this_play(df: pd.DataFrame) -> pd.Series:
    # Approximate scoring: touchdown==6, made FG==3, add 1/2 for XP/2pt is noisy; 
    # we use 6/3 only to avoid PAT noise. That’s fine for relative PPD.
    pts = pd.Series(0, index=df.index, dtype=float)
    if "touchdown" in df.columns:
        pts = pts + df["touchdown"].fillna(False).astype(int) * 6
    pts = pts + _made_fg(df).astype(int) * 3
    return pts


def compute_team_clutch_metrics(
    pbp: pd.DataFrame,
    season: int | None = None,
    cfg: ClutchConfig | None = None
) -> pd.DataFrame:
    cfg = cfg or ClutchConfig()

    # 1) Flaga clutch (działa też bez wp/wpa – leverage_gate off, zostaje time/score gate)
    df = add_clutch_flag(pbp, cfg)
    if season is not None and "season" in df.columns:
        df = df[df["season"] == season]

    clutch = df[df["is_clutch"]].copy()
    if clutch.empty:
        raise ValueError("No clutch plays found with current gates. Consider relaxing config or verifying inputs.")

    # 2) Pochodne per play
    clutch = clutch.assign(
        success=_play_success(clutch),                 # EPA>0 fallback
        t4_attempt=_third_fourth_attempt(clutch),      # 3rd/4th down
        t4_conv=_third_fourth_converted(clutch),       # conversion
        in_rz=_red_zone(clutch),                       # red zone flag
        drive_id=_drive_id(clutch),                    # drive grouping
        play_points=_points_this_play(clutch),         # 6/3 points only
    )

    # 3) OFF perspective (posteam)
    off = clutch.groupby("posteam", dropna=False)

    # WPA/play (tylko jeśli wpa istnieje)
    has_wpa = "wpa" in clutch.columns
    if has_wpa:
        off_wpa_sum = off["wpa"].sum(min_count=1)
        off_plays_for_wpa = off["wpa"].count()
        off_wpa_play = (off_wpa_sum / off_plays_for_wpa.replace(0, np.nan)).rename("clutch_off_wpa_play")
    else:
        off_wpa_play = None

    # Success rate (zawsze dostępne – z _play_success)
    off_sr = off["success"].mean().rename("clutch_off_success_rate")

    # 3rd/4th conversion
    t4_att = off["t4_attempt"].sum()
    t4_conv = off["t4_conv"].sum()
    off_t4 = (t4_conv / t4_att.replace(0, np.nan)).fillna(0.0).rename("clutch_off_3rd4th_conv")

    # Red Zone TD%
    # jeśli brak kolumny 'touchdown', to _points_this_play już zlicza TD=6, ale tu RZ-TD% potrzebuje booleana
    td_bool = clutch["touchdown"].fillna(False).astype(bool) if "touchdown" in clutch.columns else pd.Series(False, index=clutch.index)
    rz_att = off["in_rz"].sum()
    rz_td = off.apply(lambda g: (g["in_rz"] & td_bool.loc[g.index]).sum())
    off_rz = (rz_td / rz_att.replace(0, np.nan)).fillna(0.0).rename("clutch_off_rz_td_pct")

    # Points per drive (PPD) w oknie clutch
    drive_points = (
        clutch
        .groupby(["posteam", "game_id", "drive_id"], dropna=False)["play_points"]
        .sum()
        .reset_index()
    )
    ppd = drive_points.groupby("posteam", dropna=False)["play_points"].mean().rename("clutch_ppd")

    # 4) DEF perspective (allowed) – tylko jeśli wpa mamy
    if has_wpa:
        def_ = clutch.groupby("defteam", dropna=False)
        def_wpa_sum = def_["wpa"].sum(min_count=1)
        def_plays_for_wpa = def_["wpa"].count()
        def_wpa_play_allowed = (def_wpa_sum / def_plays_for_wpa.replace(0, np.nan)).rename("clutch_def_wpa_play_allowed")
    else:
        def_wpa_play_allowed = None

    # 5) Złożenie ramki wynikowej
    parts = []
    if off_wpa_play is not None:
        parts.append(off_wpa_play)
    if def_wpa_play_allowed is not None:
        parts.append(def_wpa_play_allowed)

    parts.extend([off_sr, off_t4, off_rz, ppd])

    out = (
        pd.concat(parts, axis=1)
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "team"})
        .drop_duplicates(subset=["team"])  # w razie gdyby concat z def dał duplikaty
    )

    # 6) Backoff na małej próbie (bazujemy na licznikach z OFF)
    #    - success rate: liczba clutch plays ofensywnych
    off_plays_any = off["success"].count()
    #    - 3rd/4th conversion: liczba prób t4
    #    - RZ TD%: liczba plays in_rz (przybliżenie; ewentualnie można liczyć po drive-ach)
    def backoff(col, attempts, min_needed):
        if col not in out.columns:
            return
        mu = out[col].mean()
        use_mu = attempts.reindex(out["team"]).fillna(0).values < min_needed
        out.loc[use_mu, col] = mu

    backoff("clutch_off_success_rate", off_plays_any, cfg.min_plays_for_rates)
    backoff("clutch_off_3rd4th_conv", t4_att, cfg.min_attempts_for_rates)
    backoff("clutch_off_rz_td_pct", rz_att, cfg.min_attempts_for_rates)

    # 7) Kompozyt (z-score + wagi, automatyczne przeważenie jeśli brak WPA)
    z = out.copy()

    use_off_wpa = "clutch_off_wpa_play" in z.columns and not z["clutch_off_wpa_play"].isna().all()
    use_def_wpa = "clutch_def_wpa_play_allowed" in z.columns and not z["clutch_def_wpa_play_allowed"].isna().all()

    if "clutch_off_3rd4th_conv" not in z.columns:
        z["clutch_off_3rd4th_conv"] = 0.0
    if "clutch_off_rz_td_pct" not in z.columns:
        z["clutch_off_rz_td_pct"] = 0.0

    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    components = []
    weights = []
    base_weights = ClutchConfig().weights  # {'off_wpa_play':0.40, 'third_fourth_conv':0.20, 'rz_td_pct':0.20, 'def_wpa_play_neg':0.20}

    if use_off_wpa:
        z["clutch_off_wpa_play"] = zscore(z["clutch_off_wpa_play"])
        components.append(z["clutch_off_wpa_play"])
        weights.append(base_weights["off_wpa_play"])

    z["clutch_off_3rd4th_conv"] = zscore(z["clutch_off_3rd4th_conv"])
    components.append(z["clutch_off_3rd4th_conv"])
    weights.append(base_weights["third_fourth_conv"])

    z["clutch_off_rz_td_pct"] = zscore(z["clutch_off_rz_td_pct"])
    components.append(z["clutch_off_rz_td_pct"])
    weights.append(base_weights["rz_td_pct"])

    if use_def_wpa:
        z["def_wpa_play_neg"] = -zscore(z["clutch_def_wpa_play_allowed"])
        components.append(z["def_wpa_play_neg"])
        weights.append(base_weights["def_wpa_play_neg"])

    # znormalizuj wagi do sumy 1 dla dostępnych komponentów
    weights = np.array(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    out["clutch_index_raw"] = 0.0
    for comp, w in zip(components, weights):
        out["clutch_index_raw"] += w * comp

    # Skala 0–100
    mn, mx = out["clutch_index_raw"].min(), out["clutch_index_raw"].max()
    out["clutch_index_0_100"] = 50.0 if mx - mn < 1e-9 else 100 * (out["clutch_index_raw"] - mn) / (mx - mn)

    # 8) Dodaj sezon, kolejność kolumn
    if season is not None:
        out.insert(0, "season", season)

    cols = [
        *(["season"] if "season" in out.columns else []),
        "team",
        # składowe
        *(["clutch_off_wpa_play"] if "clutch_off_wpa_play" in out.columns else []),
        *(["clutch_def_wpa_play_allowed"] if "clutch_def_wpa_play_allowed" in out.columns else []),
        "clutch_off_success_rate",
        "clutch_off_3rd4th_conv",
        "clutch_off_rz_td_pct",
        "clutch_ppd",
        # wynik
        "clutch_index_raw",
        "clutch_index_0_100",
    ]
    # zabezpieczenie: niektóre kolumny mogą nie istnieć — przefiltruj
    cols = [c for c in cols if c in out.columns]
    return out[cols]

    # Sample-size backoffs to season means
    def backoff(col, attempts, min_needed):
        mu = out[col].mean()
        use_mu = attempts < min_needed
        out.loc[use_mu, col] = mu

    # Backoff for rate-like stats based on attempts
    backoff("clutch_off_success_rate", off_plays.reindex(out["team"]).fillna(0).values, cfg.min_plays_for_rates)
    backoff("clutch_off_3rd4th_conv", t4_att.reindex(out["team"]).fillna(0).values, cfg.min_attempts_for_rates)
    backoff("clutch_off_rz_td_pct", rz_att.reindex(out["team"]).fillna(0).values, cfg.min_attempts_for_rates)

    # Build composite (z-score within season)
    z = out.copy()
    for c in ["clutch_off_wpa_play","clutch_off_3rd4th_conv","clutch_off_rz_td_pct"]:
        z[c] = (z[c] - z[c].mean()) / (z[c].std(ddof=0) + 1e-9)
    # Defense: lower allowed is better; invert sign
    z["def_wpa_play_neg"] = - (z["clutch_def_wpa_play_allowed"] - z["clutch_def_wpa_play_allowed"].mean()) / (z["clutch_def_wpa_play_allowed"].std(ddof=0) + 1e-9)

    w = ClutchConfig().weights
    out["clutch_index_raw"] = (
        w["off_wpa_play"] * z["clutch_off_wpa_play"]
        + w["third_fourth_conv"] * z["clutch_off_3rd4th_conv"]
        + w["rz_td_pct"] * z["clutch_off_rz_td_pct"]
        + w["def_wpa_play_neg"] * z["def_wpa_play_neg"]
    )

    # Min–max to 0–100 for readability
    mn, mx = out["clutch_index_raw"].min(), out["clutch_index_raw"].max()
    if mx - mn < 1e-9:
        out["clutch_index_0_100"] = 50.0
    else:
        out["clutch_index_0_100"] = 100 * (out["clutch_index_raw"] - mn) / (mx - mn)

    if season is not None:
        out.insert(0, "season", season)

    # Final ordering
    cols = [
        *("season" in out and ["season"] or []),
        "team",
        "clutch_off_wpa_play",
        "clutch_def_wpa_play_allowed",
        "clutch_off_success_rate",
        "clutch_off_3rd4th_conv",
        "clutch_off_rz_td_pct",
        "clutch_ppd",
        "clutch_index_raw",
        "clutch_index_0_100",
    ]
    return out[cols]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Clutch Index per team-season")
    parser.add_argument("--pbp_csv", required=True, help="Path to season play-by-play CSV (nflverse style)")
    parser.add_argument("--season", type=int, required=False, help="Season filter (optional if CSV already single-season)")
    parser.add_argument("--out_csv", required=True, help="Output CSV path for team clutch metrics")
    args = parser.parse_args()

    pbp = pd.read_csv(args.pbp_csv)
    df = compute_team_clutch_metrics(pbp, season=args.season)
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved Clutch Index: {args.out_csv}")
