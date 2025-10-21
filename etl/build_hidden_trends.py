import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

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


def penalty_timing_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #5 – Penalty Timing Index
    Wpływ kar w kluczowych kontekstach:
      A) 3rd down – ofensywne kary na 3rd down (zwiększające dystans) -> 'kill'
      B) Red Zone (yardline_100 <= 20) – ofensywne kary w RZ
    Zwraca: ['team','pen_timing_index','pen_3rd_kill_rate','pen_rz_off_pen_rate','off_pens_tracked']
    Definicje:
      - 'offensive penalty': _booly('penalty')==True i posteam istnieje
      - 3rd-kill: down==3 i kara, a 'ydstogo'/'yards_to_go' po karze > przed karą (przybliżenie: patrzymy na 'ydstogo' na playu karnym)
      - RZ penalty: yardline_100 <= 20 oraz kara
    """
    posteam = _safe_col(df, "posteam")
    dn = pd.to_numeric(_safe_col(df, "down", default=np.nan), errors="coerce")
    ytg = pd.to_numeric(_safe_col(df, "ydstogo", "yards_to_go", default=np.nan), errors="coerce")
    y100 = pd.to_numeric(_safe_col(df, "yardline_100", default=np.nan), errors="coerce")
    is_pen = _booly(df, "penalty")

    mask_off_pen = is_pen & posteam.notna()
    if not mask_off_pen.any():
        return pd.DataFrame(columns=["team","pen_timing_index","pen_3rd_kill_rate","pen_rz_off_pen_rate","off_pens_tracked"])

    base = pd.DataFrame({
        "team": posteam[mask_off_pen].values,
        "down": dn[mask_off_pen].values,
        "ytg": ytg[mask_off_pen].values,
        "y100": y100[mask_off_pen].values,
    }).dropna(subset=["team"])

    if base.empty:
        return pd.DataFrame(columns=["team","pen_timing_index","pen_3rd_kill_rate","pen_rz_off_pen_rate","off_pens_tracked"])

    pen_3rd_kill = (base["down"] == 3) & (base["ytg"] >= 6)
    pen_rz = base["y100"] <= 20

    grp = base.groupby("team").agg(
        off_pens_tracked=("team", "size"),
        pen_3rd_kills=("down", lambda s: int(((base.loc[s.index, "down"] == 3) & (base.loc[s.index, "ytg"] >= 6)).sum())),
        pen_rz_cnt=("y100", lambda s: int((base.loc[s.index, "y100"] <= 20).sum())),
    )

    grp = grp.reset_index()
    grp["pen_3rd_kill_rate"] = np.where(grp["off_pens_tracked"] > 0, grp["pen_3rd_kills"] / grp["off_pens_tracked"], np.nan)
    grp["pen_rz_off_pen_rate"] = np.where(grp["off_pens_tracked"] > 0, grp["pen_rz_cnt"] / grp["off_pens_tracked"], np.nan)

    grp["pen_timing_index"] = grp[["pen_3rd_kill_rate","pen_rz_off_pen_rate"]].mean(axis=1)

    return grp[["team","pen_timing_index","pen_3rd_kill_rate","pen_rz_off_pen_rate","off_pens_tracked"]]


def qb_volatility_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #7 – QB Volatility Index
    Zmienność EPA/dropback: odchylenie standardowe EPA na akcjach podaniowych (pass + sacks jeśli obecne).
    Zwraca: ['team','qb_volatility_std','qb_dropbacks']
    """
    posteam = _safe_col(df, "posteam")
    epa = pd.to_numeric(_safe_col(df, "epa", default=np.nan), errors="coerce")
    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)

    is_dropback = pt.eq("pass")
    mask = is_dropback & posteam.notna() & epa.notna()
    if not mask.any():
        return pd.DataFrame(columns=["team","qb_volatility_std","qb_dropbacks"])

    base = pd.DataFrame({"team": posteam[mask].values, "epa": epa[mask].values})
    agg = base.groupby("team").agg(qb_volatility_std=("epa","std"), qb_dropbacks=("epa","size")).reset_index()
    return agg[["team","qb_volatility_std","qb_dropbacks"]]


def red_zone_denial_rate_def(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #8 – Red Zone Denial Rate (Defense)
    % wizyt RZ rywala kończących się FG (made) zamiast TD.
    Implementacja drive-based:
      - RZ drive: dowolny play z yardline_100 <= 20 dla danej ofensywy
      - Wynik drive'u: TD jeśli jakikolwiek play w drive ma touchdown==1; w przeciwnym razie FG jeśli field_goal_result=='made'
    Zwraca metryki w ujęciu 'defteam'.
    Kolumny zwracane: ['team','rz_denial_rate_def','rz_trips_def','rz_fg_def','rz_td_def']
    """
    game_id = _safe_col(df, "game_id")
    off = _safe_col(df, "posteam")
    deff = _safe_col(df, "defteam", "def_team")
    drive_col = "drive" if "drive" in df.columns else ("drive_id" if "drive_id" in df.columns else None)
    if drive_col is None:
        return pd.DataFrame(columns=["team","rz_denial_rate_def","rz_trips_def","rz_fg_def","rz_td_def"])

    y100 = pd.to_numeric(_safe_col(df, "yardline_100", default=np.nan), errors="coerce")
    td = _booly(df, "touchdown")
    fgr = df["field_goal_result"].astype(str).str.lower() if "field_goal_result" in df.columns else pd.Series(index=df.index, dtype=object)

    base = pd.DataFrame({
        "game_id": game_id, "off": off, "def": deff,
        "drive": df[drive_col], "y100": y100, "td": td, "fgr": fgr
    }).dropna(subset=["off","def","drive"])

    if base.empty:
        return pd.DataFrame(columns=["team","rz_denial_rate_def","rz_trips_def","rz_fg_def","rz_td_def"])

    d = base.groupby(["game_id","off","def","drive"]).agg(
        entered_rz=("y100", lambda s: int((s <= 20).any())),
        any_td=("td","any"),
        any_fg=("fgr", lambda s: any(s == "made")),
    ).reset_index()

    d = d[d["entered_rz"] == 1]
    if d.empty:
        return pd.DataFrame(columns=["team","rz_denial_rate_def","rz_trips_def","rz_fg_def","rz_td_def"])

    agg = d.groupby("def").agg(
        rz_trips_def=("entered_rz","size"),
        rz_fg_def=("any_fg","sum"),
        rz_td_def=("any_td","sum"),
    ).reset_index()
    agg["rz_denial_rate_def"] = np.where((agg["rz_fg_def"] + agg["rz_td_def"]) > 0,
                                         agg["rz_fg_def"] / (agg["rz_fg_def"] + agg["rz_td_def"]), np.nan)
    return agg.rename(columns={"def":"team"})[["team","rz_denial_rate_def","rz_trips_def","rz_fg_def","rz_td_def"]]


def negative_epa_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #9 – Negative EPA Recovery
    Jak często zespół naprawia negatywną akcję (epa<0) kolejną udaną (success==1 lub epa>0) w tej samej serii (drive).
    Zwraca: ['team','neg_epa_recovery_rate','neg_events_tracked']
    """
    posteam = _safe_col(df, "posteam")
    game_id = _safe_col(df, "game_id")
    drive_col = "drive" if "drive" in df.columns else ("drive_id" if "drive_id" in df.columns else None)
    if drive_col is None:
        return pd.DataFrame(columns=["team","neg_epa_recovery_rate","neg_events_tracked"])

    epa = pd.to_numeric(_safe_col(df, "epa", default=np.nan), errors="coerce")
    if "success" in df.columns:
        succ = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    else:
        succ = epa > 0

    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    is_play = pt.isin(["run","pass"])
    is_clean = (~_booly(df,"penalty")) & (~_booly(df,"no_play")) & (~_booly(df,"qb_kneel")) & (~_booly(df,"qb_spike"))

    base = pd.DataFrame({
        "game_id": game_id, "team": posteam, "drive": df[drive_col],
        "epa": epa, "succ": succ, "idx": np.arange(len(df))
    })
    base = base.loc[is_play & is_clean].dropna(subset=["team","drive","epa"])
    if base.empty:
        return pd.DataFrame(columns=["team","neg_epa_recovery_rate","neg_events_tracked"])

    base = base.sort_values(["game_id","team","drive","idx"])
    same_pair = (base["game_id"].shift(-1).eq(base["game_id"])) & \
                (base["team"].shift(-1).eq(base["team"])) & \
                (base["drive"].shift(-1).eq(base["drive"]))
    neg_event = base["epa"] < 0
    next_succ = base["succ"].shift(-1).fillna(False).astype(bool)
    recovered = same_pair & neg_event & next_succ


    team_agg = base.groupby("team").agg(
        neg_events_tracked=("epa", lambda s: int((s < 0).sum())),
        recovered_cnt=("epa", lambda s: int(recovered.loc[s.index].sum()))
    ).reset_index()
    team_agg["neg_epa_recovery_rate"] = np.where(team_agg["neg_events_tracked"] > 0,
                                                 team_agg["recovered_cnt"] / team_agg["neg_events_tracked"], np.nan)
    return team_agg[["team","neg_epa_recovery_rate","neg_events_tracked"]]


def third_down_depth_to_go(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #10 – 3rd Down Depth to Go
    Średni dystans do przejścia na 3rd down.
    Zwraca: ['team','third_down_avg_ytg','third_down_plays']
    """
    posteam = _safe_col(df, "posteam")
    dn = pd.to_numeric(_safe_col(df, "down", default=np.nan), errors="coerce")
    ytg = pd.to_numeric(_safe_col(df, "ydstogo", "yards_to_go", default=np.nan), errors="coerce")

    mask = posteam.notna() & (dn == 3) & ytg.notna()
    if not mask.any():
        return pd.DataFrame(columns=["team","third_down_avg_ytg","third_down_plays"])

    base = pd.DataFrame({"team": posteam[mask].values, "ytg": ytg[mask].values})
    agg = base.groupby("team").agg(third_down_avg_ytg=("ytg","mean"), third_down_plays=("ytg","size")).reset_index()
    return agg[["team","third_down_avg_ytg","third_down_plays"]]


def run_game_sustainability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #11 – Run Game Sustainability
    % biegów z >2 yds before contact (YBC). Jeśli brak kolumny YBC – zwróć pustkę (NaN + 0).
    Próbujemy kolumn: 'yards_before_contact', 'rush_yards_before_contact', 'rushing_yards_before_contact'.
    Zwraca: ['team','run_sustainability_rate','rushes_tracked_ybc']
    """
    posteam = _safe_col(df, "posteam")
    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)
    is_run = pt.eq("run")

    ybc = None
    for cand in ["yards_before_contact", "rush_yards_before_contact", "rushing_yards_before_contact"]:
        if cand in df.columns:
            ybc = pd.to_numeric(df[cand], errors="coerce")
            break
    if ybc is None:
        return pd.DataFrame(columns=["team","run_sustainability_rate","rushes_tracked_ybc"])

    mask = is_run & posteam.notna() & ybc.notna()
    if not mask.any():
        return pd.DataFrame(columns=["team","run_sustainability_rate","rushes_tracked_ybc"])

    base = pd.DataFrame({"team": posteam[mask].values, "ybc": ybc[mask].values})
    agg = base.groupby("team").agg(
        rushes_tracked_ybc=("ybc","size"),
        runs_gt2_ybc=("ybc", lambda s: int((s > 2.0).sum()))
    ).reset_index()
    agg["run_sustainability_rate"] = np.where(agg["rushes_tracked_ybc"] > 0, agg["runs_gt2_ybc"] / agg["rushes_tracked_ybc"], np.nan)
    return agg[["team","run_sustainability_rate","rushes_tracked_ybc"]]


def defensive_adjustment_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend #12 – Defensive Adjustment Speed
    ΔEPA/drive 1H→2H (szybkość adaptacji). Mierzymy defensywnie:
      - liczymy przeciwnikowi EPA/drive w 1. połowie (qtr 1–2) i w 2. połowie (3–4)
      - agregujemy po 'defteam'
      - speed = -(EPA/drive 2H - EPA/drive 1H) (dodatni = poprawa w 2H)
    Zwraca: ['team','def_adj_speed','def_epa_drive_h1','def_epa_drive_h2','def_drives_tracked']
    """
    deff = _safe_col(df, "defteam", "def_team")
    game_id = _safe_col(df, "game_id")
    drive_col = "drive" if "drive" in df.columns else ("drive_id" if "drive_id" in df.columns else None)
    if drive_col is None:
        return pd.DataFrame(columns=["team","def_adj_speed","def_epa_drive_h1","def_epa_drive_h2","def_drives_tracked"])

    qtr = pd.to_numeric(_safe_col(df, "qtr", "quarter"), errors="coerce")
    epa = pd.to_numeric(_safe_col(df, "epa", default=np.nan), errors="coerce")

    base = pd.DataFrame({
        "game_id": game_id, "def": deff, "drive": df[drive_col], "qtr": qtr, "epa": epa
    }).dropna(subset=["def","drive","qtr"])

    if base.empty:
        return pd.DataFrame(columns=["team","def_adj_speed","def_epa_drive_h1","def_epa_drive_h2","def_drives_tracked"])

    def half_tag(q):
        return np.where(q.isin([1,2]), "H1", np.where(q.isin([3,4]), "H2", "OTH"))

    d = base.groupby(["game_id","def","drive"]).agg(
        qtr_min=("qtr","min"),
        qtr_max=("qtr","max"),
        epa_sum=("epa","sum"),
        plays=("epa","size"),
    ).reset_index()
    d["half"] = half_tag(d["qtr_max"])

    d = d[d["half"].isin(["H1","H2"])]
    if d.empty:
        return pd.DataFrame(columns=["team","def_adj_speed","def_epa_drive_h1","def_epa_drive_h2","def_drives_tracked"])

    agg = d.groupby(["def","half"]).agg(
        epa_per_drive=("epa_sum","mean"),
        drives=("epa_sum","size"),
    ).reset_index()

    wide = agg.pivot_table(index="def", columns="half", values="epa_per_drive", aggfunc="first")
    counts = agg.pivot_table(index="def", columns="half", values="drives", aggfunc="first").fillna(0)

    out = pd.DataFrame({
        "team": wide.index,
        "def_epa_drive_h1": wide.get("H1"),
        "def_epa_drive_h2": wide.get("H2"),
        "def_drives_tracked": (counts.get("H1", 0) + counts.get("H2", 0)).astype("Int64"),
    }).reset_index(drop=True)

    out["def_adj_speed"] = -(out["def_epa_drive_h2"] - out["def_epa_drive_h1"])
    return out[["team","def_adj_speed","def_epa_drive_h1","def_epa_drive_h2","def_drives_tracked"]]


def explosiveness_to_success_balance(df: pd.DataFrame, pass_thr: int = 15, rush_thr: int = 10) -> pd.DataFrame:
    """
    Trend #13 – Explosiveness-to-Success Balance
    Balans ryzyko–stabilność: explosive_rate / success_rate dla ofensywy.
    - Explosive: pass >=15y, run >=10y
    - Success: jeśli 'success' kolumna; w przeciwnym razie epa>0
    Zwraca: ['team','exp_to_succ_balance','explosive_rate','success_rate','plays_tracked']
    """
    posteam = _safe_col(df, "posteam")
    epa = pd.to_numeric(_safe_col(df, "epa", default=np.nan), errors="coerce")
    yards_gained = pd.to_numeric(_safe_col(df, "yards_gained", default=np.nan), errors="coerce")
    pt = df["play_type"].astype(str).str.lower() if "play_type" in df.columns else pd.Series(index=df.index, dtype=object)

    is_play = pt.isin(["run","pass"])
    if "success" in df.columns:
        succ = pd.to_numeric(df["success"], errors="coerce").fillna(0) == 1
    else:
        succ = epa > 0

    mask = posteam.notna() & is_play & yards_gained.notna()
    if not mask.any():
        return pd.DataFrame(columns=["team","exp_to_succ_balance","explosive_rate","success_rate","plays_tracked"])

    base = pd.DataFrame({
        "team": posteam[mask].values,
        "pt": pt[mask].values,
        "yg": yards_gained[mask].values,
        "succ": succ[mask].values,
    })

    explosive = ((base["pt"] == "pass") & (base["yg"] >= pass_thr)) | ((base["pt"] == "run") & (base["yg"] >= rush_thr))
    grp = base.groupby("team").agg(
        plays_tracked=("succ","size"),
        explosive_cnt=("succ", lambda s: int(explosive.loc[s.index].sum())),
        success_cnt=("succ", lambda s: int(base.loc[s.index, "succ"].sum())),
    ).reset_index()

    grp["explosive_rate"] = np.where(grp["plays_tracked"] > 0, grp["explosive_cnt"] / grp["plays_tracked"], np.nan)
    grp["success_rate"] = np.where(grp["plays_tracked"] > 0, grp["success_cnt"] / grp["plays_tracked"], np.nan)
    grp["exp_to_succ_balance"] = grp["explosive_rate"] / grp["success_rate"].replace(0, np.nan)

    return grp[["team","exp_to_succ_balance","explosive_rate","success_rate","plays_tracked"]]

# ======================
# Builder
# ======================

def build_hidden_trends(pbp_path: str, out_path: str, season: int, week: int | None = None):
    """
    Build Hidden Trends metrics per team.
    Input:  play-by-play parquet file for a given season (pbp_path)
    Output: team_hidden_trends_{season}.csv (out_path)
    """
    print(f"[INFO] Building Hidden Trends for season {season}...")
    df = pd.read_parquet(pbp_path)
    print(f"[INFO] Loaded {len(df):,} plays from {pbp_path}")

    # Lista zespołów (gwarancja wierszy nawet przy pustych metrykach)
    teams = df["posteam"].dropna().unique().tolist() if "posteam" in df.columns else []
    base = pd.DataFrame({"team": sorted(teams)}) if teams else pd.DataFrame({"team": pd.Series(dtype=object)})

    # === Trend 1: Game Rhythm Q4 ===
    try:
        t1 = game_rhythm_q4(df)
    except Exception as e:
        print(f"[WARN] game_rhythm_q4 failed: {e}")
        t1 = pd.DataFrame(columns=["team", "game_rhythm_q4"])

    # === Trend 2: Play-Calling Entropy (neutral) ===
    try:
        t2 = play_calling_entropy_neutral(df)
    except Exception as e:
        print(f"[WARN] play_calling_entropy_neutral failed: {e}")
        t2 = pd.DataFrame(columns=["team", "play_call_entropy_neutral", "neutral_pass_rate", "neutral_plays"])

    # === Trend 3: Drive Momentum (3+ successes) ===
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

    # === Trend 5: Penalty Timing Index ===
    try:
        t6_pen = penalty_timing_index(df)
    except Exception as e:
        print(f"[WARN] penalty_timing_index failed: {e}")
        t6_pen = pd.DataFrame(columns=["team", "pen_timing_index", "pen_3rd_kill_rate", "pen_rz_off_pen_rate", "off_pens_tracked"])

    # === Trend 7: QB Volatility Index ===
    try:
        t7 = qb_volatility_index(df)
    except Exception as e:
        print(f"[WARN] qb_volatility_index failed: {e}")
        t7 = pd.DataFrame(columns=["team", "qb_volatility_std", "qb_dropbacks"])

    # === Trend 8: Red Zone Denial Rate (Defense) ===
    try:
        t8 = red_zone_denial_rate_def(df)
    except Exception as e:
        print(f"[WARN] red_zone_denial_rate_def failed: {e}")
        t8 = pd.DataFrame(columns=["team", "rz_denial_rate_def", "rz_trips_def", "rz_fg_def", "rz_td_def"])

    # === Trend 9: Negative EPA Recovery ===
    try:
        t9 = negative_epa_recovery(df)
    except Exception as e:
        print(f"[WARN] negative_epa_recovery failed: {e}")
        t9 = pd.DataFrame(columns=["team", "neg_epa_recovery_rate", "neg_events_tracked"])

    # === Trend 10: 3rd Down Depth to Go ===
    try:
        t10 = third_down_depth_to_go(df)
    except Exception as e:
        print(f"[WARN] third_down_depth_to_go failed: {e}")
        t10 = pd.DataFrame(columns=["team", "third_down_avg_ytg", "third_down_plays"])

    # === Trend 11: Run Game Sustainability (YBC) ===
    try:
        t11 = run_game_sustainability(df)
    except Exception as e:
        print(f"[WARN] run_game_sustainability failed: {e}")
        t11 = pd.DataFrame(columns=["team", "run_sustainability_rate", "rushes_tracked_ybc"])

    # === Trend 12: Defensive Adjustment Speed ===
    try:
        t12 = defensive_adjustment_speed(df)
    except Exception as e:
        print(f"[WARN] defensive_adjustment_speed failed: {e}")
        t12 = pd.DataFrame(columns=["team", "def_adj_speed", "def_epa_drive_h1", "def_epa_drive_h2", "def_drives_tracked"])

    # === Trend 13: Explosiveness-to-Success Balance ===
    try:
        t13 = explosiveness_to_success_balance(df)
    except Exception as e:
        print(f"[WARN] explosiveness_to_success_balance failed: {e}")
        t13 = pd.DataFrame(columns=["team", "exp_to_succ_balance", "explosive_rate", "success_rate", "plays_tracked"])

    # MERGE wszystkich trendów
    out_df = (
        base.merge(t1, on="team", how="left")
            .merge(t2, on="team", how="left")
            .merge(t3, on="team", how="left")
            .merge(t4, on="team", how="left")
            .merge(t5, on="team", how="left")
            .merge(t6_pen, on="team", how="left")
            .merge(t7, on="team", how="left")
            .merge(t8, on="team", how="left")
            .merge(t9, on="team", how="left")
            .merge(t10, on="team", how="left")
            .merge(t11, on="team", how="left")
            .merge(t12, on="team", how="left")
            .merge(t13, on="team", how="left")
    )

    # Kolejność kolumn (pełna lista oczekiwanych)
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
        "twomin_success_rate",
        "twomin_epa_play",
        "twomin_plays",
        "pen_timing_index",
        "pen_3rd_kill_rate",
        "pen_rz_off_pen_rate",
        "off_pens_tracked",
        "qb_volatility_std",
        "qb_dropbacks",
        "rz_denial_rate_def",
        "rz_trips_def",
        "rz_fg_def",
        "rz_td_def",
        "neg_epa_recovery_rate",
        "neg_events_tracked",
        "third_down_avg_ytg",
        "third_down_plays",
        "run_sustainability_rate",
        "rushes_tracked_ybc",
        "def_adj_speed",
        "def_epa_drive_h1",
        "def_epa_drive_h2",
        "def_drives_tracked",
        "exp_to_succ_balance",
        "explosive_rate",
        "success_rate",
        "plays_tracked",
    ]

    # Dołóż brakujące kolumny i ustaw finalną kolejność
    for col in expected_cols:
        if col not in out_df.columns:
            out_df[col] = pd.Series(dtype=float)
    out_df = out_df[expected_cols]

    # === Canonicalize team codes and drop invalid ===
    try:
        from etl.utils_team_history import normalize_team_code, NFL_TEAMS_3
    except Exception:
        try:
            from .utils_team_history import normalize_team_code, NFL_TEAMS_3  # type: ignore
        except Exception:
            normalize_team_code = None  # type: ignore
            NFL_TEAMS_3 = set()  # type: ignore

    out_df["team"] = out_df["team"].astype(str).str.upper().str.strip()
    if normalize_team_code is not None:
        before_unique = sorted(set(out_df["team"].tolist()))
        mapped: List[str] = []
        dropped: List[str] = []
        out_df["team"] = out_df["team"].map(lambda x: normalize_team_code(x))
        # Track dropped rows (None)
        na_mask = out_df["team"].isna()
        if na_mask.any():
            dropped = sorted(set(before_unique) - set(out_df.loc[~na_mask, "team"].dropna().unique().tolist()))
            out_df = out_df.loc[~na_mask].copy()
        # Filter to known NFL teams when list available
        if NFL_TEAMS_3:
            bad = out_df.loc[~out_df["team"].isin(NFL_TEAMS_3), "team"].unique().tolist()
            if bad:
                print(f"[WARN] Dropping unknown teams after normalization: {sorted(bad)}")
            out_df = out_df.loc[out_df["team"].isin(NFL_TEAMS_3)].copy()
        print(f"[INFO] Team codes normalized. Unique teams: {out_df['team'].nunique()} (dropped aliases: {dropped or []})")

    # Final column order and logging
    print("[INFO] out_df columns:", list(out_df.columns))
    print("[INFO] out_df shape:", out_df.shape)
    try:
        print("[INFO] sample:\n", out_df.head(8).to_string(index=False))
    except Exception:
        pass

    out_path = Path(out_path)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved hidden trends to {out_path}")

    # Opcjonalny update historii drużyn – bezpieczny import
    update_team_history = None
    try:
        # jeśli skrypt uruchamiany jako część pakietu etl
        from etl.utils_team_history import update_team_history  # type: ignore
    except Exception:
        try:
            # jeśli masz strukturę modułów i chcesz import relatywny
            from .utils_team_history import update_team_history  # type: ignore
        except Exception:
            update_team_history = None

    if update_team_history is not None and week is not None:
        try:
            update_team_history(out_df, season=season, week=week, store="data/processed/teams")
            print(f"[OK] Team history updated for season={season}, week={week}.")
        except Exception as e:
            print(f"[WARN] update_team_history failed: {e}")
    else:
        print("[INFO] Skipping team history update (no function available or no week provided).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--in_pbp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--week", type=int, default=None, help="Optional week number for team history update")
    args = parser.parse_args()

    build_hidden_trends(
        pbp_path=args.in_pbp,
        out_path=args.out,
        season=args.season,
        week=args.week,
    )


