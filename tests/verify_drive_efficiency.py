#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

# ---------- pomocnicze (te same heurystyki co w ETL) ----------

def col(df, *names, default=None):
    for n in names:
        if n in df.columns:
            return df[n]
    if default is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)

def _clock_to_seconds(s: pd.Series) -> pd.Series:
    ss = s.fillna("00:00").astype(str)
    parts = ss.str.split(":", n=1, expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    secs = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    return (mins * 60 + secs).astype(int)

def detect_game_id(df): return col(df, "game_id", "gameId", "GameId")
def detect_team_off(df): return col(df, "posteam", "offense", "offense_team")
def detect_team_def(df): return col(df, "defteam", "defense", "defense_team")

def detect_drive_key(df: pd.DataFrame) -> pd.Series:
    game_id = detect_game_id(df).fillna("UNK").astype(str)
    off_team = detect_team_off(df).fillna("UNK").astype(str)

    if "drive_id" in df.columns and df["drive_id"].notna().any():
        return df["drive_id"].astype(str)

    if "new_drive" in df.columns and df["new_drive"].notna().any():
        nd = pd.to_numeric(df["new_drive"], errors="coerce").fillna(0).astype(int)
        grp = game_id
        drive_idx = nd.groupby(grp).cumsum()
        first_mask = drive_idx.groupby(grp).transform("min").eq(0)
        drive_idx = drive_idx + first_mask.astype(int)
        return game_id + "_" + drive_idx.astype("Int64").astype(str)

    grp = game_id
    posteam_prev = off_team.shift(1)
    same_game_prev = grp.eq(grp.shift(1))
    new_possession = (~same_game_prev) | off_team.ne(posteam_prev)
    drive_idx = new_possession.groupby(grp).cumsum()
    return game_id + "_" + drive_idx.astype("Int64").astype(str)

SCORING_FG_VALUES = {"good": 3, "made": 3, "successful": 3}
def fg_made_value(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.map(SCORING_FG_VALUES).fillna(0).astype(int)

def detect_punt(df: pd.DataFrame) -> pd.Series:
    punt_flag = (col(df, "punt", default=0).fillna(0).astype(int) == 1)
    if "play_type" in df.columns:
        punt_flag |= df["play_type"].astype(str).str.lower().eq("punt")
    return punt_flag.astype(int)

def detect_turnover(df: pd.DataFrame) -> pd.Series:
    pt = col(df, "play_type", default="").astype(str).str.lower()
    desc = col(df, "desc", default="").astype(str).str.lower()

    is_int = pt.eq("interception") | desc.str.contains("intercept")

    has_fum = desc.str.contains("fumble")
    out_of_bounds = desc.str.contains("out of bounds")

    off_low = col(df, "posteam", default="").astype(str).str.lower()
    def_low = col(df, "defteam", default="").astype(str).str.lower()

    rec_by_off = pd.Series(
        [("recovered by " + o) in t if o else False for t, o in zip(desc.tolist(), off_low.tolist())],
        index=df.index, dtype=bool
    )
    rec_by_def = pd.Series(
        [("recovered by " + d) in t if d else False for t, d in zip(desc.tolist(), def_low.tolist())],
        index=df.index, dtype=bool
    )
    is_fum_lost = has_fum & ~out_of_bounds & rec_by_def & ~rec_by_off

    is_4th = col(df, "down", default=np.nan).fillna(0).astype(float).eq(4)
    no_first_down = ~col(df, "first_down", default=0).astype(int).eq(1)
    downs_text = desc.str.contains("turnover on downs") | desc.str.contains("on downs")
    is_tod = is_4th & no_first_down & downs_text

    return (is_int | is_fum_lost | is_tod).astype(int)

def is_offensive_td(df: pd.DataFrame) -> pd.Series:
    pt = col(df, "pass_touchdown", default=0).fillna(0).astype(int)
    rt = col(df, "rush_touchdown", default=0).fillna(0).astype(int)
    rec = col(df, "receive_touchdown", default=0).fillna(0).astype(int)
    any_td = col(df, "touchdown", default=0).fillna(0).astype(int)
    td_off_like = ((pt + rt + rec) > 0) | (any_td > 0)
    return td_off_like.astype(int)

def detect_redzone(df: pd.DataFrame) -> pd.Series:
    yl = col(df, "yardline_100", default=np.nan).astype(float)
    return (yl <= 20).fillna(False)

# ---------- właściwy test ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--team", type=str, required=True)  # np. DET
    ap.add_argument("--side", type=str, choices=["off","def"], required=True)
    ap.add_argument("--in_pbp", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_pbp)

    # filtr sezon/tydzień
    if "season" in df.columns:
        df = df[df["season"] == args.season].copy()
    if "week" in df.columns:
        df = df[df["week"] == args.week].copy()

    # sort jak w ETL
    time_sec = _clock_to_seconds(df.get("time", pd.Series(index=df.index)))
    df = df.assign(_time_sec=time_sec)
    sort_cols, sort_asc = ["game_id"], [True]
    if "qtr" in df.columns:
        sort_cols.append("qtr"); sort_asc.append(True)
    sort_cols.append("_time_sec"); sort_asc.append(False)
    if "play_id" in df.columns:
        sort_cols.append("play_id"); sort_asc.append(True)
    df = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    # drive key, strony
    df["drive_key"] = detect_drive_key(df)
    df["off"] = detect_team_off(df)
    df["def"] = detect_team_def(df)

    # wybór danych dla side
    if args.side == "off":
        df = df[df["off"] == args.team].copy()
    else:
        df = df[df["def"] == args.team].copy()

    if df.empty:
        print("Brak playów po filtrze (season/week/team/side).")
        return 0

    # flagi per play
    df["off_td_play"] = is_offensive_td(df)
    fg_res = col(df, "field_goal_result", "kick_result", default="")
    desc = col(df, "desc", default="").astype(str).str.lower()
    play_type = col(df, "play_type", default="").astype(str).str.lower()
    fg_made_text = (play_type.eq("field_goal")) & (desc.str.contains(" is good") | desc.str.contains(" good"))
    df["fg_made_play"] = (((fg_made_value(fg_res) // 3) > 0) | fg_made_text).astype(int)

    df["punt_play"] = detect_punt(df)
    df["turnover_play"] = detect_turnover(df)
    df["yards_gained"] = col(df, "yards_gained", default=0).fillna(0).astype(float)
    df["is_redzone_play"] = detect_redzone(df)

    # start FP (pierwszy scrimmage w drive, fallback na pierwszy znany yardline_100)
    df["_ord_in_drive"] = df.groupby("drive_key").cumcount()
    yl = pd.to_numeric(col(df, "yardline_100", default=np.nan), errors="coerce")
    desc_low = col(df, "desc", default="").astype(str).str.lower()
    pt_low = col(df, "play_type", default="").astype(str).str.lower()
    no_play = col(df, "no_play", default=0).astype(int)
    is_kickoff = pt_low.eq("kickoff") | desc_low.str.contains("kicks off")
    is_scrimmage = (no_play.eq(0)) & (~is_kickoff) & yl.notna()

    first_scrim = (
        df.loc[is_scrimmage, ["drive_key", "_ord_in_drive"]]
          .assign(yardline_100=yl[is_scrimmage].values)
          .sort_values(["drive_key","_ord_in_drive"])
          .drop_duplicates(subset=["drive_key"], keep="first")
          .set_index("drive_key")["yardline_100"]
    )
    first_any = (
        df.loc[yl.notna(), ["drive_key", "_ord_in_drive"]]
          .assign(yardline_100=yl[yl.notna()].values)
          .sort_values(["drive_key","_ord_in_drive"])
          .drop_duplicates(subset=["drive_key"], keep="first")
          .set_index("drive_key")["yardline_100"]
    )

    # agregacja per drive
    g = (
        df.groupby("drive_key")
          .agg(
              team_off=("off","first"),
              team_def=("def","first"),
              plays=("play_id","count"),
              yards=("yards_gained","sum"),
              td=("off_td_play","max"),
              fg=("fg_made_play","max"),
              punt=("punt_play","max"),
              to=("turnover_play","max"),
              rz=("is_redzone_play","max"),
          )
          .reset_index()
    )
    g["drive_points_basic"] = g["td"]*7 + g["fg"]*3

    # dołącz start FP z wyliczeń
    start_fp = first_scrim.reindex(g["drive_key"])
    missing = start_fp.isna()
    if missing.any():
        start_fp.loc[missing] = first_any.reindex(g.loc[missing, "drive_key"]).values
    g["start_yardline_100"] = pd.to_numeric(start_fp, errors="coerce")

    # etykieta wyniku drive’u
    def outcome_row(r):
        if r["td"] == 1: return "TD"
        if r["fg"] == 1: return "FG"
        if r["punt"] == 1: return "Punt"
        if r["to"] == 1: return "Turnover"
        return "Other"
    g["outcome"] = g.apply(outcome_row, axis=1)

    # posortuj wg kolejności w meczu
    order = df.groupby("drive_key")["_ord_in_drive"].min().reindex(g["drive_key"]).values
    g["_order"] = order
    g = g.sort_values(["team_off","_order"]).reset_index(drop=True)

    # --------- WYNIKI SZCZEGÓŁOWE ---------
    print(f"\n=== Drives for {args.team} ({'OFF' if args.side=='off' else 'DEF'}) — Season {args.season}, Week {args.week} ===")
    print(g[["drive_key","team_off","team_def","plays","yards","start_yardline_100","outcome","drive_points_basic"]])

    # --------- PODSUMOWANIE / METRYKI ---------
    drives = len(g)
    td_drives = int(g["td"].sum())
    fg_drives = int(g["fg"].sum())
    to_drives = int(g["to"].sum())
    punt_drives = int(g["punt"].sum())

    score_drives = td_drives + fg_drives
    def rate(x): return (x / drives) if drives else 0.0
    ppd_basic = g["drive_points_basic"].sum() / drives if drives else 0.0

    print("\n--- SUMMARY ---")
    print(f"drives={drives}")
    print(f"score_rate={rate(score_drives):.6f}  (td_rate={rate(td_drives):.6f}, fg_rate={rate(fg_drives):.6f})")
    print(f"turnover_rate={rate(to_drives):.6f}  punt_rate={rate(punt_drives):.6f}")
    print(f"ppd_basic={ppd_basic:.6f}")
    print(f"yds_per_drive={g['yards'].sum()/drives if drives else 0.0:.6f}")
    print(f"plays_per_drive={g['plays'].sum()/drives if drives else 0.0:.6f}")
    print(f"redzone_drive_rate={g['rz'].sum()/drives if drives else 0.0:.6f}")
    print(f"start_yardline_100_avg={g['start_yardline_100'].mean() if g['start_yardline_100'].notna().any() else np.nan}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
