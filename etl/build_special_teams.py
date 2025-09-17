# etl/build_special_teams.py
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

ST_TYPES = {
    "kickoff", "kickoff_return", "kickoff_downed", "kickoff_recovered",
    "punt", "punt_return", "punt_blocked",
    "field_goal", "extra_point",
}

def _safe_div(a, b):
    return float(a) / float(b) if b not in (None, 0) else 0.0

def _zser(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    m, sd = s.mean(), s.std(ddof=0)
    return (s - m) / sd if sd and sd > 0 else s * 0.0

def build(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- Filtry sezonu / REG ---
    if "season" in df.columns:
        df = df.loc[df["season"] == season].copy()
    if "season_type" in df.columns:
        df = df.loc[df["season_type"] == "REG"].copy()
    elif "week" in df.columns:
        df = df.loc[df["week"].between(1, 18)].copy()

    # --- Normalizacje bazowe ---
    def _series(df, name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    for col in ["posteam", "defteam", "play_type"]:
        df[col] = _series(df, col, "").fillna("").astype(str)

    df["play_type"] = df["play_type"].str.lower()

    # opisy i pomocnicze tekstowe
    df["kickoff_result"] = _series(df, "kickoff_result", "").fillna("").astype(str).str.lower()
    df["desc"] = _series(df, "desc", "").fillna("").astype(str)
    df["desc_lower"] = df["desc"].str.lower()

    df["week"] = pd.to_numeric(_series(df, "week"), errors="coerce").fillna(0).astype(int)
    df["epa"] = pd.to_numeric(_series(df, "epa"), errors="coerce").fillna(0.0)

    df["yardline_100"]  = pd.to_numeric(_series(df, "yardline_100"), errors="coerce")
    df["return_yards"]  = pd.to_numeric(_series(df, "return_yards"), errors="coerce").fillna(0)
    df["punt_distance"] = pd.to_numeric(_series(df, "punt_distance"), errors="coerce").fillna(0)
    df["touchback"]     = pd.to_numeric(_series(df, "touchback"), errors="coerce").fillna(0).astype(int)
    df["penalty"]       = pd.to_numeric(_series(df, "penalty"), errors="coerce").fillna(0).astype(int)

    # flaga ST też jako seria
    is_st_flag = _series(df, "special_teams", 0).astype(bool)

    # wybór zagrań ST
    is_st = is_st_flag | df["play_type"].isin(ST_TYPES)
    st = df.loc[is_st].copy()

    # --- Maski typów ---
    st["is_ko"] = st["play_type"].str.startswith("kickoff")
    st["is_punt"] = st["play_type"].str.startswith("punt")
    st["is_fg"] = st["play_type"].eq("field_goal")
    st["is_xp"] = st["play_type"].eq("extra_point")
    st["is_return"] = st["play_type"].isin(["kickoff_return", "punt_return"])

    # --- Heurystyka touchbacków i zakończenia kickoffa z 'desc' ---
    # TB: 'touchback' lub 'kneels ... end zone' lub 'kick through end zone'
    st["ko_tb_flag"] = np.where(
        st["is_ko"] & (
            st["desc_lower"].str.contains("touchback")
            | (st["desc_lower"].str.contains("kneels") & st["desc_lower"].str.contains("end zone"))
            | st["desc_lower"].str.contains("kick through end zone")
        ),
        1, 0
    )

    # parser końcowej pozycji po kickoffie: bierzemy OSTATNIE 'to <TEAM> <NUM>'
    def _extract_ko_end_yd100(desc: str) -> float:
        if not isinstance(desc, str) or not desc:
            return np.nan
        matches = list(re.finditer(r"\bto\s+([A-Z]{2,3})\s+(\d{1,2})\b", desc))
        if matches:
            yd = int(matches[-1].group(2))
            return 100 - yd  # 'to ARI 30' → start na 30, więc yardline_100 = 70
        if "touchback" in desc.lower():
            return 70.0  # 25-yard line → 100-25
        return np.nan

    st["ko_end_yd100"] = np.where(
        st["is_ko"],
        st["desc"].apply(_extract_ko_end_yd100),
        np.nan
    )

    # start rywala po puntach (fallback na yardline_100 końca akcji)
    st["punt_opp_start_yd100"] = np.where(st["is_punt"], st["yardline_100"], np.nan)

    # Drużyna ST (dla returnów to team returnujący = defteam; w innych — posteam)
    st["st_team"] = np.where(st["is_return"], st["defteam"], st["posteam"])
    st["opp_team"] = np.where(st["is_return"], st["posteam"], st["defteam"])

    # Wartości pomocnicze
    st["ret_epa_val"] = np.where(st["is_return"], st["epa"], 0.0)
    st["punt_net_part"] = np.where(st["is_punt"], st["punt_distance"], 0.0)
    st["punt_ret_part"] = np.where(st["is_punt"], st["return_yards"], 0.0)
    st["punt_tb_part"] = np.where(st["is_punt"], st["touchback"], 0)

    # === WEEKLY ===
    grp = ["week", "st_team"]
    weekly_base = st.groupby(grp, as_index=False).agg(
        st_plays=("epa", "count"),
        st_epa=("epa", "sum"),
        st_pen=("penalty", "sum"),
        ko_tb=("ko_tb_flag", "sum"),
        ko_plays=("is_ko", "sum"),
        ko_opp_start_yd100=("ko_end_yd100", "mean"),
        punt_opp_start_yd100=("punt_opp_start_yd100", "mean"),
        ret_epa=("ret_epa_val", "sum"),
        ret_plays=("is_return", "sum"),
        punt_net_raw=("punt_net_part", "sum"),
        punt_ret_yards=("punt_ret_part", "sum"),
        punt_tb=("punt_tb_part", "sum"),
        punt_plays=("is_punt", "sum"),
        fg_att=("is_fg", "sum"),
        xp_att=("is_xp", "sum"),
    )

    # FG/XPA wyniki jeśli mamy kolumny z wynikami
    if "fg_result" in st.columns:
        fg_good = st.loc[st["is_fg"]].groupby(grp)["fg_result"].apply(lambda s: (s == "good").sum())
        weekly_base = weekly_base.merge(fg_good.rename("fg_made"), on=grp, how="left")
    else:
        weekly_base["fg_made"] = 0

    if "extra_point_result" in st.columns:
        xp_good = st.loc[st["is_xp"]].groupby(grp)["extra_point_result"].apply(lambda s: (s == "good").sum())
        weekly_base = weekly_base.merge(xp_good.rename("xp_made"), on=grp, how="left")
    else:
        weekly_base["xp_made"] = 0

    # ret_epa_against: EPA z returnów rywali przeciwko Tobie
    ret_against = (
        st.loc[st["is_return"]]
        .groupby(["week", "opp_team"], as_index=False)["ret_epa_val"].sum()
        .rename(columns={"opp_team": "st_team", "ret_epa_val": "ret_epa_against"})
    )
    weekly = weekly_base.merge(ret_against, on=grp, how="left")
    weekly["ret_epa_against"] = weekly["ret_epa_against"].fillna(0.0)

    # Pochodne weekly
    weekly["st_epa_per_play"] = weekly.apply(lambda r: _safe_div(r.st_epa, r.st_plays), axis=1)
    weekly["ko_tb_rate"] = weekly.apply(lambda r: _safe_div(r.ko_tb, r.ko_plays), axis=1)
    weekly["punt_net"] = weekly.apply(
        lambda r: _safe_div(r.punt_net_raw - r.punt_ret_yards - 20 * r.punt_tb, r.punt_plays),
        axis=1,
    )
    weekly["fg_pct"] = weekly.apply(lambda r: _safe_div(r.fg_made, r.fg_att), axis=1)
    weekly["xp_pct"] = weekly.apply(lambda r: _safe_div(r.xp_made, r.xp_att), axis=1)
    weekly["fg_epa_per_att"] = weekly.apply(lambda r: _safe_div(r.st_epa if r.fg_att else 0.0, r.fg_att), axis=1)
    weekly["xp_epa_per_att"] = weekly.apply(lambda r: _safe_div(r.st_epa if r.xp_att else 0.0, r.xp_att), axis=1)
    weekly["st_pen_rate"] = weekly.apply(lambda r: _safe_div(r.st_pen, r.st_plays), axis=1)

    weekly = weekly.rename(columns={"st_team": "team"})

    # === TEAM (sumy + średnie ważone gdzie trzeba) ===
    sum_cols = [
        "st_plays", "st_epa", "st_pen",
        "ko_tb", "ko_plays",
        "punt_net_raw", "punt_ret_yards", "punt_tb", "punt_plays",
        "fg_att", "fg_made",
        "xp_att", "xp_made",
        "ret_epa", "ret_epa_against",
    ]
    team = weekly.groupby("team", as_index=False)[sum_cols].sum()

    # Średnie / wskaźniki
    team["st_epa_per_play"] = team.apply(lambda r: _safe_div(r.st_epa, r.st_plays), axis=1)
    team["ko_tb_rate"] = team.apply(lambda r: _safe_div(r.ko_tb, r.ko_plays), axis=1)
    team["punt_net"] = team.apply(
        lambda r: _safe_div(r.punt_net_raw - r.punt_ret_yards - 20 * r.punt_tb, r.punt_plays),
        axis=1,
    )
    team["fg_pct"] = team.apply(lambda r: _safe_div(r.fg_made, r.fg_att), axis=1)
    team["xp_pct"] = team.apply(lambda r: _safe_div(r.xp_made, r.xp_att), axis=1)

    # EPA/próba FG/XP: średnia z weekly jeśli istnieje
    if "fg_epa_per_att" in weekly.columns:
        fg_epa_avg = weekly.groupby("team")["fg_epa_per_att"].mean()
        team = team.merge(fg_epa_avg.rename("fg_epa_per_att"), on="team", how="left")
    else:
        team["fg_epa_per_att"] = 0.0

    if "xp_epa_per_att" in weekly.columns:
        xp_epa_avg = weekly.groupby("team")["xp_epa_per_att"].mean()
        team = team.merge(xp_epa_avg.rename("xp_epa_per_att"), on="team", how="left")
    else:
        team["xp_epa_per_att"] = 0.0

    # Średnie ważone startów rywala po KO i puntach
    ko_starts = weekly.groupby("team").apply(
        lambda g: _safe_div((g["ko_opp_start_yd100"].fillna(0) * g["ko_plays"]).sum(), g["ko_plays"].sum())
        if g["ko_plays"].sum() else np.nan
    )
    team = team.merge(ko_starts.rename("ko_opp_start_yd100"), on="team", how="left")

    punt_starts = weekly.groupby("team").apply(
        lambda g: _safe_div((g["punt_opp_start_yd100"].fillna(0) * g["punt_plays"]).sum(), g["punt_plays"].sum())
        if g["punt_plays"].sum() else np.nan
    )
    team = team.merge(punt_starts.rename("punt_opp_start_yd100"), on="team", how="left")

    # --- Special Teams Impact Score (0–100) ---
    team["st_pen_rate"] = team.apply(lambda r: _safe_div(r.st_pen, r.st_plays), axis=1)
    team["ret_epa_diff"] = team["ret_epa"] - team["ret_epa_against"]

    z = (
        0.35 * _zser(team["st_epa_per_play"]) +
        0.20 * _zser(-team["punt_opp_start_yd100"].fillna(team["punt_opp_start_yd100"].mean())) +  # niżej = lepiej
        0.15 * _zser(-team["ko_opp_start_yd100"].fillna(team["ko_opp_start_yd100"].mean())) +      # niżej = lepiej
        0.15 * _zser(team["ret_epa_diff"]) +
        0.10 * _zser(team["fg_epa_per_att"]) +
        0.05 * _zser(-team["st_pen_rate"])
    )
    team["st_score"] = 50 + 10 * z

    # Kolumny wyjściowe
    keep_weekly = [
        "week", "team",
        "st_plays", "st_epa", "st_epa_per_play",
        "ko_plays", "ko_tb_rate", "ko_opp_start_yd100",
        "punt_plays", "punt_opp_start_yd100",
        "ret_plays", "ret_epa", "ret_epa_against",
        "fg_att", "fg_made", "fg_pct", "fg_epa_per_att",
        "xp_att", "xp_made", "xp_pct", "xp_epa_per_att",
        "st_pen_rate",
    ]
    weekly = weekly[[c for c in keep_weekly if c in weekly.columns]].copy()

    keep_team = [
        "team",
        "st_plays", "st_epa", "st_epa_per_play",
        "ko_tb_rate", "ko_opp_start_yd100",
        "punt_opp_start_yd100",
        "ret_epa", "ret_epa_against", "ret_epa_diff",
        "fg_pct", "fg_epa_per_att", "xp_pct", "xp_epa_per_att",
        "st_pen_rate", "st_score",
        "punt_net",  # zostawione pomocniczo – może być 0 jeśli brak dystansu w PBP
    ]
    team = team[[c for c in keep_team if c in team.columns]].copy()

    return weekly, team

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=str, required=True)
    ap.add_argument("--out_weekly", type=str, required=True)
    ap.add_argument("--out_team", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_pbp)
    weekly, team = build(df, args.season)

    Path(args.out_weekly).parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out_weekly, index=False)
    team.to_csv(args.out_team, index=False)

    print(f"✅ Zapisano weekly: {args.out_weekly} (rows={len(weekly)})")
    print(f"🏁 Zapisano team:   {args.out_team} (rows={len(team)})")
