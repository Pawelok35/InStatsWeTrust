#!/usr/bin/env python3
"""
Build Drive Efficiency metrics (Offense/Defense) from PBP.

Outputs:
- weekly CSV (per season, week, team): <out_weekly>
- team CSV (season aggregates per team): <out_team>

Metrics (per side O/D):
- drives: number of offensive drives (for D: opponent drives faced)
- score_rate: (TD + made FG) / drives
- td_rate: TD drives / drives
- fg_rate: made FG drives / drives
- turnover_rate: (INT/FUM/ToD) / drives
- punt_rate: punts / drives
- ppd_basic: (7*TD + 3*FG_made) / drives  (fallback)
- ppd_points: true points per drive if `drive_points` is available (preferred)
- yds_per_drive: total yards gained / drives
- plays_per_drive: offensive plays / drives
- start_yardline_100_avg: average starting field position (0 = own GL, 100 = opp GL)
- redzone_drive_rate: drives with any snap in y<=20 / drives

Schema-flexible: works with nflfastR-like columns if present; otherwise falls back to heuristics.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
# --- stdlib ---
import re  # 👈 potrzebne do re.search / re.escape


def col01(df: pd.DataFrame, name: str, default=0) -> pd.Series:
    """
    Zwraca serię 0/1 z dowolnej kolumny (bool/int/float/str) z bezpiecznym rzutowaniem:
    True/False -> 1/0, liczby !=0 -> 1, NaN/None/"" -> 0. Gdy kolumna nie istnieje → default.
    """
    if name in df.columns:
        s = df[name]
    else:
        s = pd.Series(default, index=df.index)
    s = s.replace({True: 1, False: 0})
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s != 0).astype(int)

def sstr(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    """
    Bezpieczny lowercase string: brak kolumny/NaN → "", zawsze .str.lower().
    """
    if name in df.columns:
        s = df[name].astype("string").fillna(default)
    else:
        s = pd.Series(default, index=df.index, dtype="string")
    return s.str.lower()

# ---------------------------- helpers ---------------------------- #

def _clock_to_seconds(s: pd.Series) -> pd.Series:
    """
    Convert 'MM:SS' game clock to seconds remaining in quarter.
    NFL clock counts down; we sort by qtr ASC and time_sec DESC.
    """
    if s is None:
        return pd.Series(dtype="int64")
    ss = s.fillna("00:00").astype(str)
    parts = ss.str.split(":", n=1, expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    secs = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    return (mins * 60 + secs).astype(int)

SCORING_FG_VALUES = {"good": 3, "made": 3, "successful": 3}

def col(df: pd.DataFrame, *names, default=None):
    """Return first existing column among names, else a Series filled with default."""
    for n in names:
        if n in df.columns:
            return df[n]
    if default is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)

def detect_week(df: pd.DataFrame) -> pd.Series:
    return col(df, "week", "game_week", "WEEK", default=np.nan).astype("Int64")

def detect_season(df: pd.DataFrame) -> pd.Series:
    return col(df, "season", "Season", default=np.nan).astype("Int64")

def detect_team_off(df: pd.DataFrame) -> pd.Series:
    return col(df, "posteam", "offense", "offense_team")

def detect_team_def(df: pd.DataFrame) -> pd.Series:
    return col(df, "defteam", "defense", "defense_team")

def detect_game_id(df: pd.DataFrame) -> pd.Series:
    return col(df, "game_id", "gameId", "GameId")

def fg_made_value(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.map(SCORING_FG_VALUES).fillna(0).astype(int)

def detect_punt(df: pd.DataFrame) -> pd.Series:
    # nflfastR: play_type == 'punt' or punt == 1
    punt_flag = (col(df, "punt", default=0).fillna(0).astype(int) == 1)
    if "play_type" in df.columns:
        punt_flag |= df["play_type"].astype(str).str.lower().eq("punt")
    return punt_flag.astype(int)

# --- PODMIANA detect_turnover ---
def detect_turnover(df: pd.DataFrame) -> pd.Series:
    """
    Heurystyka: INT / FUMBLE LOST / TURNOVER ON DOWNS na podstawie play_type/desc.
    Zwraca serię 0/1.
    """
    pt   = sstr(df, "play_type")
    desc = sstr(df, "desc")

    # 1) Interception
    is_int = pt.eq("interception") | desc.str.contains("intercept", na=False)

    # 2) Fumble lost: 'fumble' + odzyskane przez DEF (nie out of bounds, nie przez offense)
    has_fum       = desc.str.contains("fumble", na=False)
    out_of_bounds = desc.str.contains("out of bounds", na=False)

    off_low = sstr(df, "posteam")
    def_low = sstr(df, "defteam")

    rec_by_off = pd.Series(
    [bool(re.search(rf"recovered by (the )?{re.escape(o)}\b", t))
     if o else False
     for o, t in zip(off_low, desc)],
    index=df.index,
    dtype=bool,
    )

    rec_by_def = pd.Series(
        [bool(re.search(rf"recovered by (the )?{re.escape(d)}\b", t))
        if d else False
        for d, t in zip(def_low, desc)],
        index=df.index,
        dtype=bool,
    )


    is_fum_lost = has_fum & ~out_of_bounds & rec_by_def & ~rec_by_off

    # 3) Turnover on downs: 4th down + brak first down + fraza "on downs"
    is_4th        = pd.to_numeric(df.get("down", np.nan), errors="coerce").fillna(0).astype(float).eq(4)
    first_down_01 = col01(df, "first_down", default=0)
    no_first_down = ~first_down_01.eq(1)
    downs_text    = desc.str.contains("turnover on downs", na=False) | desc.str.contains(r"\bon downs\b", na=False)
    is_tod        = is_4th & no_first_down & downs_text

    return (is_int | is_fum_lost | is_tod).astype(int)




def detect_redzone(df: pd.DataFrame) -> pd.Series:
    # yardline_100: distance to opponent goal line. Red zone if <= 20.
    yl = col(df, "yardline_100", default=np.nan).astype(float)
    return (yl <= 20).fillna(False)

def is_offensive_td(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit offensive TD flags if present; else fallback to generic `touchdown`
    pt = col(df, "pass_touchdown", default=0).fillna(0).astype(int)
    rt = col(df, "rush_touchdown", default=0).fillna(0).astype(int)
    rec = col(df, "receive_touchdown", default=0).fillna(0).astype(int)
    any_td = col(df, "touchdown", default=0).fillna(0).astype(int)
    td_off_like = ((pt + rt + rec) > 0) | (any_td > 0)
    return td_off_like.astype(int)

# -------- drive detection (robust) -------- #

def detect_drive_key(df: pd.DataFrame) -> pd.Series:
    """
    Return a robust per-drive key.
    Priority:
      1) existing `drive_id`
      2) derived from `new_drive` (cumsum per game)
      3) derived from possession changes (when `posteam` changes within a game)
      4) fallback: game_id + posteam + numeric `drive` (rarely used)
    Assumes df is sorted by (game_id, qtr ASC, time DESC, play_id ASC).
    """
    game_id = detect_game_id(df).fillna("UNK").astype(str)
    off_team = detect_team_off(df).fillna("UNK").astype(str)

    # 1) Explicit drive id
    if "drive_id" in df.columns and df["drive_id"].notna().any():
        return df["drive_id"].astype(str)

    # 2) `new_drive` available (nflfastR style)
    if "new_drive" in df.columns and df["new_drive"].notna().any():
        nd = pd.to_numeric(df["new_drive"], errors="coerce").fillna(0).astype(int)
        grp = game_id
        drive_idx = nd.groupby(grp).cumsum()
        first_mask = drive_idx.groupby(grp).transform("min").eq(0)
        drive_idx = drive_idx + first_mask.astype(int)
        return game_id + "_" + drive_idx.astype("Int64").astype(str)

    # 3) Possession-change heuristic (works for your current schema)
    grp = game_id
    posteam_prev = off_team.shift(1)
    same_game_prev = grp.eq(grp.shift(1))
    new_possession = (~same_game_prev) | off_team.ne(posteam_prev)
    drive_idx = new_possession.groupby(grp).cumsum()
    return game_id + "_" + drive_idx.astype("Int64").astype(str)

    # 4) Fallback (kept for completeness)
    # drive_no = col(df, "drive", "Drive", "drive_number", "DriveNumber", default=np.nan)
    # drive_no = pd.to_numeric(drive_no, errors="coerce").astype("Int64")
    # return game_id + "_" + off_team + "_" + drive_no.astype(str)

# ---------------------------- core logic ---------------------------- #

def build_drive_df(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()

    # Drop rows without a game id
    if "game_id" in pbp.columns:
        pbp = pbp[pbp["game_id"].notna()].copy()

    # Ensure proper chronological order inside each game:
    # game_id ASC, qtr ASC, time (seconds) DESC, play_id ASC
    time_sec = _clock_to_seconds(pbp.get("time", pd.Series(index=pbp.index, dtype="float64")))
    pbp = pbp.assign(_time_sec=time_sec)

    sort_cols = ["game_id"]
    sort_asc = [True]
    if "qtr" in pbp.columns:
        sort_cols.append("qtr")
        sort_asc.append(True)
    sort_cols.append("_time_sec")
    sort_asc.append(False)
    if "play_id" in pbp.columns:
        sort_cols.append("play_id")
        sort_asc.append(True)

    pbp = pbp.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    # basic columns
    pbp["season"] = detect_season(pbp)
    pbp["week"] = detect_week(pbp)
    pbp["game_id"] = detect_game_id(pbp)
    pbp["drive_key"] = detect_drive_key(pbp)
    pbp["off"] = detect_team_off(pbp)
    pbp["def"] = detect_team_def(pbp)

    # per-play derived flags
    pbp["off_td_play"] = is_offensive_td(pbp)

    # FG made: prefer structured result; fallback do heurystyki z opisu
    fg_res = col(pbp, "field_goal_result", "kick_result", default="")
    desc = col(pbp, "desc", default="").astype(str).str.lower()
    play_type = col(pbp, "play_type", default="").astype(str).str.lower()

    fg_made_text = (
        play_type.eq("field_goal")
        & desc.str.contains(r"\bgood\b")
        & ~desc.str.contains("no good")
    )

    pbp["fg_made_play"] = (
        ((fg_made_value(fg_res) // 3) > 0) | fg_made_text
    ).astype(int)

    pbp["punt_play"] = detect_punt(pbp)
    pbp["turnover_play"] = detect_turnover(pbp)
    pbp["yards_gained"] = col(pbp, "yards_gained", default=0).fillna(0).astype(float)
    pbp["is_redzone_play"] = detect_redzone(pbp)

    # True drive points (if present)
    drive_points_true_series = col(pbp, "drive_points", default=np.nan).astype(float)

    # Aggregate per drive
    agg = {
        "off": "first",
        "def": "first",
        "season": "first",
        "week": "first",
        "game_id": "first",
        "yards_gained": "sum",
        "off_td_play": "max",
        "fg_made_play": "max",
        "punt_play": "max",
        "turnover_play": "max",
        "is_redzone_play": "max",
    }
    g = pbp.groupby("drive_key", dropna=False).agg(agg).reset_index()

    # Points per drive (basic vs true)
    g["drive_points_basic"] = g["off_td_play"].astype(int) * 7 + g["fg_made_play"].astype(int) * 3
    dp_true_by_drive = drive_points_true_series.groupby(pbp["drive_key"]).max()
    g["drive_points_true"] = dp_true_by_drive.reindex(g["drive_key"]).values

    # plays per drive (offense snaps)
    plays = (
        pbp.assign(_one=1)
        .groupby("drive_key")["_one"]
        .sum()
        .reindex(g["drive_key"])
        .fillna(0)
        .astype(int)
        .values
    )
    g["plays"] = plays



         # ---- starting field position per drive (robust) ----
    if "yardline_100" in pbp.columns:
        # porządek wewnątrz drive’u (już powinno być posortowane wcześniej)
        pbp["_ord_in_drive"] = pbp.groupby("drive_key").cumcount()

        # pomocnicze kolumny
        pbp["_yl"]   = pd.to_numeric(pbp["yardline_100"], errors="coerce")
        pbp["_pt"]   = col(pbp, "play_type", default="").astype(str).str.lower()
        pbp["_desc"] = col(pbp, "desc", default="").astype(str).str.lower()
        pbp["_np"]   = pd.to_numeric(col(pbp, "no_play", default=0), errors="coerce").fillna(0).astype(int)

        def _drive_start_yl(sub: pd.DataFrame) -> float:
            sub = sub.sort_values("_ord_in_drive")
            # 1) pierwszy snap ze scrimmage: nie kickoff, nie no_play, znany yardline
            is_kick = sub["_pt"].eq("kickoff") | sub["_desc"].str.contains("kicks off", na=False)
            m1 = (~is_kick) & (sub["_np"] == 0) & sub["_yl"].notna()
            if m1.any():
                return float(sub.loc[m1, "_yl"].iloc[0])
            # 2) fallback: pierwszy jakikolwiek play z nie-NaN yardline
            m2 = sub["_yl"].notna()
            if m2.any():
                return float(sub.loc[m2, "_yl"].iloc[0])
            return float("nan")

        starts = pbp.groupby("drive_key", sort=False).apply(_drive_start_yl)
        # merge po drive_key (bez pułapek alignu po indeksach)
        g = g.merge(starts.rename("start_yardline_100"), left_on="drive_key", right_index=True, how="left")

        # sprzątanie
        pbp.drop(columns=["_ord_in_drive","_yl","_pt","_desc","_np"], errors="ignore", inplace=True)
    else:
        g["start_yardline_100"] = np.nan
    # ---- END starting field position ----







    # Red zone drive if any play reached RZ
    g["rz_drive"] = g["is_redzone_play"].astype(bool).astype(int)

    # Side labels
    g = g.rename(columns={"off": "team_off", "def": "team_def"})

    return g

def rate(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return (numer / denom).fillna(0.0)

def summarize_side(drive_df: pd.DataFrame, side: str):
    assert side in ("off", "def")
    key_team = "team_off" if side == "off" else "team_def"

    # Weekly
    by = ["season", "week", key_team]
    wk = (
        drive_df.groupby(by)
        .agg(
            drives=("drive_key", "nunique"),  # UNIKALNE drive'y!
            td_drives=("off_td_play", "sum"),
            fg_drives=("fg_made_play", "sum"),
            punt_drives=("punt_play", "sum"),
            to_drives=("turnover_play", "sum"),
            yds=("yards_gained", "sum"),
            plays=("plays", "sum"),
            rz_drives=("rz_drive", "sum"),
            start_yardline_100_avg=("start_yardline_100", "mean"),
            ppd_basic_sum=("drive_points_basic", "sum"),
            ppd_true_sum=("drive_points_true", lambda s: s.sum(min_count=1)),

        )
        .reset_index()
    )

    wk["score_drives"] = wk["td_drives"] + wk["fg_drives"]
    wk["score_rate"] = rate(wk["score_drives"], wk["drives"])
    wk["td_rate"] = rate(wk["td_drives"], wk["drives"])
    wk["fg_rate"] = rate(wk["fg_drives"], wk["drives"])
    wk["turnover_rate"] = rate(wk["to_drives"], wk["drives"])
    wk["punt_rate"] = rate(wk["punt_drives"], wk["drives"])
    wk["yds_per_drive"] = rate(wk["yds"], wk["drives"])
    wk["plays_per_drive"] = rate(wk["plays"], wk["drives"])
    wk["redzone_drive_rate"] = rate(wk["rz_drives"], wk["drives"])

    # Points per drive
    wk["ppd_basic"] = rate(wk["ppd_basic_sum"], wk["drives"])
    wk["ppd_points"] = wk["ppd_basic"].copy()
    mask_true = wk["ppd_true_sum"].notna()
    wk.loc[mask_true, "ppd_points"] = rate(
        wk.loc[mask_true, "ppd_true_sum"], wk.loc[mask_true, "drives"]
    )
    
      
    
    

    
  
        
    

    # TWARDY fallback: jeśli i tak wyszły NaN lub 0 przez brak drive_points, ustaw jak ppd_basic
    wk["ppd_points"] = wk["ppd_points"].where(
        wk["ppd_points"].notna() & wk["ppd_points"].ne(0),
        wk["ppd_basic"]
    )



    # Jeśli nigdzie nie ma 'drive_points', trzymaj ppd_points == ppd_basic (bez ryzyka zerowania)
    wk["ppd_points"] = wk["ppd_points"].fillna(wk["ppd_basic"])

    wk = wk.rename(columns={key_team: "team"})
    wk["side"] = side

    # Season aggregates per team
    team = (
        wk.groupby(["season", "team", "side"]).agg(
            drives=("drives", "sum"),
            score_rate=("score_rate", "mean"),
            td_rate=("td_rate", "mean"),
            fg_rate=("fg_rate", "mean"),
            turnover_rate=("turnover_rate", "mean"),
            punt_rate=("punt_rate", "mean"),
            ppd_basic=("ppd_basic", "mean"),
            ppd_points=("ppd_points", "mean"),
            yds_per_drive=("yds_per_drive", "mean"),
            plays_per_drive=("plays_per_drive", "mean"),
            redzone_drive_rate=("redzone_drive_rate", "mean"),
            start_yardline_100_avg=("start_yardline_100_avg", "mean"),
        ).reset_index()
    )

    return wk, team

def main(args=None):
    p = argparse.ArgumentParser(description="Build Drive Efficiency (O/D) from PBP")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--in_pbp", type=Path, required=True, help="Path to pbp_clean_<season>.parquet")
    p.add_argument("--out_weekly", type=Path, required=True)
    p.add_argument("--out_team", type=Path, required=True)
    ns = p.parse_args(args)

    df = pd.read_parquet(ns.in_pbp)
    if "season" in df.columns:
        df = df[df["season"] == ns.season].copy()

    drives = build_drive_df(df)

    wk_off, team_off = summarize_side(drives, "off")
    wk_def, team_def = summarize_side(drives, "def")

    weekly = pd.concat([wk_off, wk_def], ignore_index=True)

        # >>> add: SFP + Hidden Yards (weekly) - opcjonalnie
    weekly["start_own_yardline_avg"] = 100 - weekly["start_yardline_100_avg"]

    _hidden_w = weekly.pivot(index=["season","week","team"], columns="side",
                             values="start_own_yardline_avg")
    _hidden_w["hidden_yards_per_drive"] = _hidden_w["off"] - _hidden_w["def"]

    weekly = weekly.merge(_hidden_w[["hidden_yards_per_drive"]],
                          left_on=["season","week","team"], right_index=True, how="left")
    # <<< end add

    team = pd.concat([team_off, team_def], ignore_index=True)
    # >>> FIX: SFP + Hidden Yards bez duplikatów
    # upewnij się, że mamy własną linię startu
    if "start_own_yardline_avg" not in team.columns:
        team["start_own_yardline_avg"] = 100 - team["start_yardline_100_avg"]

    # policz hidden na poziomie teamu (OFF - DEF)
    _hidden = team.pivot(index="team", columns="side", values="start_own_yardline_avg")
    _hidden["hidden_yards_per_drive"] = _hidden["off"] - _hidden["def"]

    # wyczyść ewentualne stare kolumny (_x/_y) i przypisz JEDNĄ kolumnę przez map()
    for c in ["hidden_yards_per_drive_x", "hidden_yards_per_drive_y", "hidden_yards_per_drive"]:
        if c in team.columns:
            del team[c]
    team["hidden_yards_per_drive"] = team["team"].map(_hidden["hidden_yards_per_drive"])
    # <<< END FIX


    weekly = weekly.sort_values(["season", "week", "team", "side"]).reset_index(drop=True)
    team = team.sort_values(["season", "team", "side"]).reset_index(drop=True)
    

        # >>> add: rounding (czytelność) - opcjonalnie
    cols_round_team = [
        "start_yardline_100_avg","start_own_yardline_avg","hidden_yards_per_drive",
        "ppd_points","ppd_basic","yds_per_drive","plays_per_drive",
        "score_rate","td_rate","fg_rate","turnover_rate","punt_rate","redzone_drive_rate",
    ]
    cols_round_team = [c for c in cols_round_team if c in team.columns]
    team[cols_round_team] = team[cols_round_team].astype(float).round(2)

    cols_round_wk = [
        "start_yardline_100_avg","start_own_yardline_avg","hidden_yards_per_drive",
        "ppd_points","yds_per_drive","plays_per_drive","score_rate",
    ]
    cols_round_wk = [c for c in cols_round_wk if c in weekly.columns]
    weekly[cols_round_wk] = weekly[cols_round_wk].astype(float).round(2)
    # <<< end add


    ns.out_weekly.parent.mkdir(parents=True, exist_ok=True)
    ns.out_team.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(ns.out_weekly, index=False)
    team.to_csv(ns.out_team, index=False)

    n_w = len(weekly)
    n_t = len(team)
    teams = sorted(team["team"].unique())
    print(f"✅ Zapisano weekly: {ns.out_weekly} (rows={n_w})")
    print(f"🏁 Zapisano team:   {ns.out_team} (rows={n_t}, teams={len(teams)})")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        preview = (
            team.sort_values(["ppd_points"], ascending=False)
                .groupby("side")
                .head(5)
        )
        print("\n📈 Top-5 PPD (points per drive) per side:\n", preview)


           # 7) Dopisanie tygodnia do historii drużyn (Drive Efficiency, szeroki format 32×N)
    from etl.utils_team_history import update_team_history

    # team: po sezonie i drużynie mamy wiersze 'off' i 'def' -> pivot do szerokiego 32×N
    wide = team.pivot(index=["season","team"], columns="side")
    # Spłaszczenie MultiIndex kolumn: ('ppd_points','off') -> 'drives__ppd_points_off'
    wide.columns = [f"drives__{m}_{side}" for (m, side) in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    # dołóż week (WEEK ustawiony wcześniej; jeśli nie – zapisz 0)
    try:
        WEEK  # noqa: F821
    except NameError:
        WEEK = 6
    wide["week"] = WEEK or 0

    # ułóż kolumny i zapisz
    non_feat = ["season","week","team"]
    feat = [c for c in wide.columns if c not in non_feat]
    out_df = wide[non_feat + feat]

    update_team_history(out_df, season=ns.season, week=(WEEK or 0), store="data/processed/teams")


if __name__ == "__main__":
    sys.exit(main())
