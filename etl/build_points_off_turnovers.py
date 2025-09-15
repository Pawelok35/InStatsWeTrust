#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import re
from pathlib import Path

def _has(col, df): return col in df.columns

# ---------- SCORING (fallback bez total_*_score) ----------
def compute_play_points(row):
    import math

    def flag(col):
        v = row.get(col, 0)
        if v is None: return 0
        if isinstance(v, float) and math.isnan(v): return 0
        if isinstance(v, bool): return 1 if v else 0
        if isinstance(v, int): return 1 if v != 0 else 0
        if isinstance(v, float): return 1 if v != 0.0 else 0
        s = str(v).strip().lower()
        if s in {"1","true","yes","y"}: return 1
        if s in {"0","false","no","n",""}: return 0
        return 0

    def text(col):
        v = row.get(col)
        return str(v).strip().lower() if v is not None else ""

    pts, scorer = 0, None
    home, away = row.get("home_team"), row.get("away_team")
    posteam, defteam = row.get("posteam"), row.get("defteam")

    def opp(team):
        if team is None: return None
        if team == home: return away
        if team == away: return home
        return None

    # --- PRIORYTET: delta punktów jeśli mamy total_* + prev_* ---
    th = row.get("total_home_score"); ta = row.get("total_away_score")
    ph = row.get("prev_total_home_score"); pa = row.get("prev_total_away_score")
    have_deltas = (th is not None and ta is not None and ph is not None and pa is not None)
    if have_deltas and not (isinstance(ph, float) and math.isnan(ph)) and not (isinstance(pa, float) and math.isnan(pa)):
        try:
            dh = int(th - ph)
            da = int(ta - pa)
            if dh > 0 and da == 0:
                return dh, home
            if da > 0 and dh == 0:
                return da, away
            if dh == 0 and da == 0:
                return 0, None
            return dh + da, None  # edge case
        except Exception:
            pass  # fallback poniżej

    # --- FALLBACK: z flag akcji (TD/FG/XP/2PT/Safety) ---
    td_flag = flag("touchdown") or flag("rush_touchdown") or flag("pass_touchdown")
    td_team = row.get("td_team")
    if td_flag or (isinstance(td_team, str) and td_team):
        pts += 6
        scorer = td_team if (isinstance(td_team, str) and td_team) else posteam

    if text("field_goal_result") in {"made", "good", "success"}:
        pts += 3; scorer = posteam

    if text("extra_point_result") in {"good", "made", "success"}:
        pts += 1; scorer = posteam

    if text("two_point_conv_result") in {"success", "good", "converted"}:
        pts += 2
        scorer = defteam if flag("defensive_two_point_attempt") == 1 else posteam

    if flag("safety") == 1:
        pts += 2; scorer = defteam if defteam else opp(posteam)

    return (int(pts), scorer) if pts else (0, None)


# ---------- SERIE POSIADANIA ----------
def build_series(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["game_id","home_team","away_team","posteam"]:
        if c not in df.columns:
            raise ValueError(f"PBP missing required column: {c}")

    df = df.copy()
    order_cols = ["game_id"]
    if "play_id" in df.columns: order_cols.append("play_id")
    elif "index" in df.columns: order_cols.append("index")
    df = df.sort_values(order_cols).reset_index(drop=True)

    first_in_game = df.groupby("game_id", group_keys=False).cumcount() == 0
    poss_change = (df.groupby("game_id", group_keys=False)["posteam"]
                     .apply(lambda s: s != s.shift(1))
                     .reset_index(level=0, drop=True).fillna(True))

    new_series = (first_in_game | poss_change)

    # honoruj dodatkowe sygnały, jeśli są
    if "new_drive" in df.columns:
        new_series |= (df["new_drive"] == 1)
    if "drive" in df.columns:
        new_series |= (df.groupby("game_id", group_keys=False)["drive"]
                         .apply(lambda s: s != s.shift(1))
                         .reset_index(level=0, drop=True))
    if "drive_id" in df.columns:
        new_series |= (df.groupby("game_id", group_keys=False)["drive_id"]
                         .apply(lambda s: s != s.shift(1))
                         .reset_index(level=0, drop=True))

    df["series_id"] = new_series.groupby(df["game_id"]).cumsum()
    df["series_owner"] = df["posteam"]
    df["row_in_game"] = df.groupby("game_id", group_keys=False).cumcount()
    return df

# ---------- TAKEAWAYS (wiele heurystyk + regex) ----------
INTERCEPT_RE = re.compile(r"\bintercept(?:ed|ion)\b", re.I)
FUMBLE_RE    = re.compile(r"\bfumble(?:s|d)?\b", re.I)
RECOVER_RE   = re.compile(r"\brecover(?:y|ed) by\b", re.I)

SPECIAL_PT = {"punt","field_goal","kickoff","free_kick","extra_point","no_play"}

def detect_takeaways(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Heurystyka A: pola INT
    int_flags = []
    if _has("interception", df): int_flags.append(df["interception"] == 1)
    if _has("interception_player_id", df): int_flags.append(df["interception_player_id"].notna())
    if _has("pass_interception", df): int_flags.append(df["pass_interception"] == 1)
    if _has("interception_return_yards", df): int_flags.append(df["interception_return_yards"].fillna(0) > 0)
    is_int = False
    for f in int_flags:
        is_int = (is_int | f) if isinstance(is_int, pd.Series) else f

    # Heurystyka B: fumble odzyskane przez obronę
    fum_flags = []
    if _has("fumble_lost", df): fum_flags.append(df["fumble_lost"] == 1)
    if _has("fumble", df):
        rec_cols = [c for c in ["fumble_recovery_1_team","fumble_recovery_2_team","recover_team","return_team"]
                    if _has(c, df)]
        if rec_cols and _has("posteam", df):
            any_def_rec = False
            for rc in rec_cols:
                any_def_rec = any_def_rec | ((df[rc] != df["posteam"]) & df[rc].notna())
            fum_flags.append((df["fumble"] == 1) & any_def_rec)
        elif _has("change_of_possession", df):
            play_type = df.get("play_type", pd.Series("", index=df.index)).astype(str).str.lower()
            fum_flags.append((df["fumble"] == 1) & (df["change_of_possession"] == 1) & (~play_type.isin(SPECIAL_PT)))
    is_fumlost = False
    for f in fum_flags:
        is_fumlost = (is_fumlost | f) if isinstance(is_fumlost, pd.Series) else f

    # Heurystyka C: czysta zmiana posiadania (nie special teams)
    change_flags = pd.Series(False, index=df.index)
    if _has("change_of_possession", df):
        play_type = df.get("play_type", pd.Series("", index=df.index)).astype(str).str.lower()
        change_flags = (df["change_of_possession"] == 1) & (~play_type.isin(SPECIAL_PT))

    # Heurystyka D: regex w opisie akcji
    desc_col = "desc" if _has("desc", df) else ("play_description" if _has("play_description", df) else None)
    regex_flags = pd.Series(False, index=df.index)
    if desc_col:
        desc = df[desc_col].astype(str)
        regex_flags = desc.str.contains(INTERCEPT_RE) | desc.str.contains(FUMBLE_RE) | desc.str.contains(RECOVER_RE)

    # Zbiorczo
    if (not isinstance(is_int, pd.Series)) and (not isinstance(is_fumlost, pd.Series)):
        base = pd.Series(False, index=df.index)
    else:
        base = (is_int.fillna(False) if isinstance(is_int, pd.Series) else pd.Series(False, index=df.index)) | \
               (is_fumlost.fillna(False) if isinstance(is_fumlost, pd.Series) else pd.Series(False, index=df.index))

    is_takeaway = (base | change_flags | regex_flags).fillna(False)

    # Kto przejął?
    takeaway_team = pd.Series([None]*len(df), index=df.index, dtype=object)
    if _has("defteam", df):
        takeaway_team = df["defteam"].where(is_takeaway, None)

    # Jeżeli brak defteam albo None – weź właściciela następnej serii po tej akcji
    # (uzupełnimy później po zbudowaniu serii)
    df["is_takeaway"] = is_takeaway
    df["is_int"] = (is_int.fillna(False) if isinstance(is_int, pd.Series) else False)
    df["is_fumlost"] = (is_fumlost.fillna(False) if isinstance(is_fumlost, pd.Series) else False)
    df["takeaway_team"] = takeaway_team
    return df

# ---------- GŁÓWNA LOGIKA PoT ----------
def compute_pot(df: pd.DataFrame, season: int):
    df = df.copy()
    for c in ["season","week","game_id","home_team","away_team","posteam"]:
        if c not in df.columns:
            raise ValueError(f"PBP missing required column: {c}")

    df = df[df["season"] == season].copy()
    if df.empty:
        raise ValueError(f"Brak danych dla sezonu {season} w PBP.")

    df = detect_takeaways(df)
    df = build_series(df)

        # jeśli mamy total_*_score — przygotuj poprzednie wartości do delty (per game)
    if "total_home_score" in df.columns and "total_away_score" in df.columns:
        df["prev_total_home_score"] = df.groupby("game_id")["total_home_score"].shift(1)
        df["prev_total_away_score"] = df.groupby("game_id")["total_away_score"].shift(1)
    else:
        df["prev_total_home_score"] = None
        df["prev_total_away_score"] = None

    pts_scorer = df.apply(compute_play_points, axis=1, result_type=None)
    df["play_points"] = [t[0] for t in pts_scorer]
    df["play_scorer"] = [t[1] for t in pts_scorer]


    # Uzupełnij brakujące takeaway_team: następna seria należy do zespołu, który przejął
    mask_missing = df["is_takeaway"] & (~df["takeaway_team"].astype(bool))
    if mask_missing.any():
        # dla każdego playu z missing, znajdź w tym meczu pierwszą serię > series_id
        def fill_team(row, g):
            after = g[g["series_id"] > row["series_id"]]
            cand = after["series_owner"].iloc[0] if not after.empty else None
            if cand and cand != row.get("posteam"):
                return cand
            # fallback: przeciwnik posteam
            p, h, a = row.get("posteam"), row.get("home_team"), row.get("away_team")
            if p == h: return a
            if p == a: return h
            return None

        filled = []
        for gid, g in df.groupby("game_id", sort=False):
            idxs = g.index[g["is_takeaway"] & (~g["takeaway_team"].astype(bool))]
            for i in idxs:
                filled.append((i, fill_team(df.loc[i], g)))
        for i, val in filled:
            df.at[i, "takeaway_team"] = val

    # policz punkty i strzelca na KAŻDYM playu
    pts_team = df.apply(compute_play_points, axis=1, result_type=None)
    df["play_points"] = [t[0] for t in pts_team]
    df["play_scorer"] = [t[1] for t in pts_team]

    # --- DIAGNOSTYKA ---
    a = int(df["is_int"].sum())
    b = int(df["is_fumlost"].sum())
    c = int(((df.get("change_of_possession", 0) == 1) &
             (~df.get("play_type", pd.Series("", index=df.index)).astype(str).str.lower().isin({"punt","field_goal","kickoff","free_kick","extra_point","no_play"}))).sum()) if _has("change_of_possession", df) else 0
    d = int(df["is_takeaway"].sum())
    print(f"🔍 Heurystyki: INT={a}, FUM_Lost={b}, COP_nonST={c}, ALL_TAKEAWAYS={d}")

    # policz PoT per takeaway
    records = []
    for gid, g in df.groupby("game_id", sort=False):
        takeaways = g[g["is_takeaway"]].copy()
        for _, row in takeaways.iterrows():
            t_team = row["takeaway_team"]
            if not t_team:
                continue

            # 1) same-play defensive score?
            same_pts = int(row.get("play_points", 0) or 0)
            same_team = row.get("play_scorer")
            pot_points = 0
            pot_type = "NONE"

            if same_pts > 0 and same_team == t_team:
                pot_points = same_pts
                pot_type = "DEF_TD_SAME_PLAY"
            else:
                # 2) pierwsza ofensywna seria t_team po przechwycie
                curr_series = row["series_id"]
                later = g[g["series_id"] > curr_series]
                next_series_ids = later.loc[later["series_owner"] == t_team, "series_id"].unique()
                if len(next_series_ids) == 0:
                    pot_points = 0
                    pot_type = "NO_NEXT_SERIES"
                else:
                    ns = next_series_ids[0]
                    splays = g[g["series_id"] == ns]
                    drive_pts = int(splays.loc[splays["play_scorer"] == t_team, "play_points"].sum())
                    pot_points = drive_pts
                    pot_type = "NEXT_OFF_SERIES"

            records.append({
                "season": row["season"],
                "week": row["week"],
                "game_id": gid,
                "takeaway_team": t_team,
                "opponent": row.get("posteam"),
                "pot_points": int(pot_points),
                "pot_type": pot_type
            })

    pot_df = pd.DataFrame.from_records(records)
    if pot_df.empty:
        weekly = pd.DataFrame(columns=["season","week","team","pot_points","takeaways","pot_per_takeaway","games"])
        team = pd.DataFrame(columns=["season","team","pot_points","takeaways","pot_per_takeaway","games","pot_per_game"])
        return weekly, team

    weekly = (
        pot_df.groupby(["season","week","takeaway_team"], as_index=False)
              .agg(pot_points=("pot_points","sum"),
                   takeaways=("pot_points","count"))
              .rename(columns={"takeaway_team":"team"})
    )

    # games per team-week
    tg = []
    for gid, g in df.groupby("game_id", sort=False):
        season_g = int(g["season"].iloc[0]); week_g = int(g["week"].iloc[0])
        home = g["home_team"].iloc[0]; away = g["away_team"].iloc[0]
        tg.append((season_g, week_g, gid, home))
        tg.append((season_g, week_g, gid, away))
    team_games = pd.DataFrame(tg, columns=["season","week","game_id","team"]).drop_duplicates()

    weekly = weekly.merge(
        team_games.groupby(["season","week","team"], as_index=False).agg(games=("game_id","nunique")),
        on=["season","week","team"], how="left"
    )
    weekly["pot_per_takeaway"] = weekly["pot_points"] / weekly["takeaways"].where(weekly["takeaways"] > 0, pd.NA)

    team = (
        weekly.groupby(["season","team"], as_index=False)
              .agg(pot_points=("pot_points","sum"),
                   takeaways=("takeaways","sum"),
                   games=("games","sum"))
    )
    team["pot_per_takeaway"] = team["pot_points"] / team["takeaways"].where(team["takeaways"] > 0, pd.NA)
    team["pot_per_game"] = team["pot_points"] / team["games"].where(team["games"] > 0, pd.NA)

    return weekly, team

def main():
    ap = argparse.ArgumentParser(description="Build Points off Turnovers (PoT)")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=str, required=True)
    ap.add_argument("--out_weekly", type=str, required=True)
    ap.add_argument("--out_team", type=str, required=True)
    args = ap.parse_args()

    in_path = Path(args.in_pbp)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    print(f"📥 Wczytuję PBP: {in_path}")
    df = pd.read_parquet(in_path)

    weekly, team = compute_pot(df, args.season)

    out_w = Path(args.out_weekly); out_t = Path(args.out_team)
    out_w.parent.mkdir(parents=True, exist_ok=True)
    out_t.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_w, index=False)
    team.to_csv(out_t, index=False)

    print(f"✅ Zapisano weekly: {out_w} (rows={len(weekly)})")
    print(f"🏁 Zapisano team:   {out_t} (rows={len(team)})")
    print("🎯 Points off Turnovers — GOTOWE.")

if __name__ == "__main__":
    main()
