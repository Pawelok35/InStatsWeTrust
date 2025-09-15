# etl/build_redzone.py  (FINAL with synthetic drive id)

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict

TD_PRIOR = 2
FG_PRIOR = 1
EMPTY_PRIOR = 0

def _coalesce_columns(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pretty_sample_columns(df: pd.DataFrame, n: int = 30) -> str:
    cols = list(df.columns)[:n]
    more = "" if len(df.columns) <= n else f" … (+{len(df.columns)-n} więcej)"
    return ", ".join(map(str, cols)) + more

def _prepare_columns(df: pd.DataFrame) -> Dict[str, str | None]:
    # Szerokie aliasy pod typowe warianty schema
    col_season  = _coalesce_columns(df, ["season", "Season", "year"])
    col_week    = _coalesce_columns(df, ["week", "Week"])
    col_game    = _coalesce_columns(df, ["game_id", "gameId", "GameID", "gameid"])
    col_posteam = _coalesce_columns(df, ["posteam", "possession_team", "offense_team", "Poss_Team"])
    col_defteam = _coalesce_columns(df, ["defteam", "defense_team", "Def_Team"])
    col_drive   = _coalesce_columns(df, ["drive", "drive_id", "drive_number", "Drive"])  # może być None
    col_yard100 = _coalesce_columns(df, ["yardline_100", "yardline_100_clean", "YardsToEZ", "yardline100"])
    col_td      = _coalesce_columns(df, ["touchdown", "td", "is_touchdown", "pass_touchdown", "rush_touchdown"])
    col_fg      = _coalesce_columns(df, ["field_goal_result", "fg_result"])
    col_qtr     = _coalesce_columns(df, ["qtr", "quarter"])
    col_time    = _coalesce_columns(df, ["time", "game_clock", "clock"])

    needed_pairs = {
        "season": col_season,
        "week": col_week,
        "game_id": col_game,
        "posteam": col_posteam,
        "defteam": col_defteam,
        "yardline_100": col_yard100,
        "qtr": col_qtr,
        "time": col_time,
    }
    missing = [k for k, v in needed_pairs.items() if v is None]
    if missing:
        sample = _pretty_sample_columns(df)
        raise ValueError(
            "Brak wymaganych kolumn w PBP: "
            + ", ".join(missing)
            + f".\nPodgląd dostępnych kolumn (do 30): {sample}"
        )

    # touchdown: OR(pass_td, rush_td) jeśli brak ogólnej
    if col_td is None:
        pass_td = _coalesce_columns(df, ["pass_touchdown"])
        rush_td = _coalesce_columns(df, ["rush_touchdown"])
        if pass_td or rush_td:
            df["__touchdown__"] = (df.get(pass_td, 0)).fillna(0).astype(int) + (df.get(rush_td, 0)).fillna(0).astype(int)
            df["__touchdown__"] = df["__touchdown__"].clip(upper=1)
            col_td = "__touchdown__"
        else:
            df["__touchdown__"] = 0
            col_td = "__touchdown__"

    if col_fg is None:
        df["__field_goal_result__"] = None
        col_fg = "__field_goal_result__"

    return {
        "season": col_season, "week": col_week, "game_id": col_game,
        "posteam": col_posteam, "defteam": col_defteam, "drive": col_drive,
        "yard100": col_yard100, "td": col_td, "fg": col_fg,
        "qtr": col_qtr, "time": col_time,
    }

def _parse_mmss_to_secs(s: str) -> int:
    # 'MM:SS' -> total seconds (int); gdy brak/niepoprawne -> NaN (później fill)
    try:
        parts = str(s).split(":")
        if len(parts) != 2:
            return None
        m = int(parts[0]); sec = int(parts[1])
        return m * 60 + sec
    except Exception:
        return None

def _ensure_drive(df: pd.DataFrame, cols: Dict[str, str | None]) -> pd.Series:
    """
    Zwraca serię z identyfikatorem drive'u.
    Jeśli oryginalny 'drive' jest dostępny -> użyjemy go.
    W przeciwnym razie tworzymy 'drive_synth' jako kumulacyjny licznik
    zmian posiadania w obrębie game_id w kolejności chronologicznej gry:
      sort: (game_id asc, qtr asc, time secs desc)
      boundary: zmiana posteam względem poprzedniego (w tym samym game_id)
    """
    if cols["drive"] is not None and cols["drive"] in df.columns:
        return df[cols["drive"]]

    # budujemy porządek
    tmp = df[[cols["game_id"], cols["posteam"], cols["qtr"], cols["time"]]].copy()
    tmp["_secs"] = tmp[cols["time"]].map(_parse_mmss_to_secs)
    # brakujące zegary ustaw na -1, aby nie psuły sortowania
    tmp["_secs"] = tmp["_secs"].fillna(-1)

    order = (
        df[cols["game_id"]].astype(str)
        + "||" + df[cols["qtr"]].astype(int).astype(str).str.zfill(2)
        + "||" + tmp["_secs"].rsub(9999).astype(int).astype(str).str.zfill(4)  # malejąco po czasie => rosnąco po (9999 - secs)
    )
    # sort_index wg posortowanych wartości
    ord_idx = order.sort_values().index

    # w tej kolejności oznaczamy boundaries
    game = df.loc[ord_idx, cols["game_id"]].values
    post = df.loc[ord_idx, cols["posteam"]].values

    boundary = [True]  # pierwsza akcja meczu to nowy drive
    for i in range(1, len(ord_idx)):
        is_new_drive = (game[i] != game[i-1]) or (post[i] != post[i-1])
        boundary.append(is_new_drive)

    # kumulacyjnie numerujemy drive'y per game
    import numpy as np
    boundary = np.array(boundary, dtype=bool)
    # licznik wspólny, ale potem znormalizujemy na per-game
    drive_seq = boundary.cumsum()

    # przemapuj do oryginalnego indeksu dataframe
    synth = pd.Series(index=df.index, dtype="int64")
    synth.loc[ord_idx] = drive_seq

    # normalizacja do per-game (żeby drive_id zaczynało się od 1 dla każdego game_id)
    # odejmij minimalny licznik w obrębie game_id i dodaj 1
    per_game_min = synth.groupby(df[cols["game_id"]]).transform("min")
    synth = (synth - per_game_min + 1).astype("int64")

    return synth

def compute_redzone_metrics(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = _prepare_columns(pbp)
    df = pbp.copy()

    # Typy
    df[cols["td"]] = pd.to_numeric(df[cols["td"]], errors="coerce").fillna(0).astype(int)
    df[cols["yard100"]] = pd.to_numeric(df[cols["yard100"]], errors="coerce")

    # Zapewnij identyfikator drive’u
    drive_id = _ensure_drive(df, cols)

    grp_keys = [cols["season"], cols["week"], cols["game_id"], cols["posteam"], cols["defteam"], drive_id]

    # Agregacja per drive
    agg = df.groupby(grp_keys).agg(
        any_rz=(cols["yard100"], lambda s: pd.to_numeric(s, errors="coerce").le(20).any()),
        td_in_drive=(cols["td"], "max"),
        fg_made_in_drive=(cols["fg"], lambda s: s.fillna("").astype(str).str.lower().eq("made").any()),
    ).reset_index(names=["season","week","game_id","posteam","defteam","drive_id"])

    def res_score(row):
        if row["td_in_drive"] >= 1:
            return TD_PRIOR
        if row["fg_made_in_drive"]:
            return FG_PRIOR
        return EMPTY_PRIOR

    agg["drive_result_score"] = agg.apply(res_score, axis=1)

    rz = agg[agg["any_rz"]].copy()

    # Offense weekly
    off_weekly = rz.groupby(["season","week","posteam","defteam"]).agg(
        rz_trips=("any_rz", "size"),
        rz_td=("drive_result_score", lambda s: (s == TD_PRIOR).sum()),
        rz_fg=("drive_result_score", lambda s: (s == FG_PRIOR).sum()),
        rz_empty=("drive_result_score", lambda s: (s == EMPTY_PRIOR).sum()),
    ).reset_index()
    off_weekly["rz_efficiency"] = (off_weekly["rz_td"] / off_weekly["rz_trips"]).round(4)
    off_weekly = off_weekly.rename(columns={"posteam":"team","defteam":"opp"})

    # Defense weekly (allowed)
    def_weekly = rz.groupby(["season","week","defteam","posteam"]).agg(
        rz_trips_allowed=("any_rz", "size"),
        rz_td_allowed=("drive_result_score", lambda s: (s == TD_PRIOR).sum()),
        rz_fg_allowed=("drive_result_score", lambda s: (s == FG_PRIOR).sum()),
        rz_empty_allowed=("drive_result_score", lambda s: (s == EMPTY_PRIOR).sum()),
    ).reset_index()
    def_weekly["rz_efficiency_allowed"] = (def_weekly["rz_td_allowed"] / def_weekly["rz_trips_allowed"]).round(4)
    def_weekly = def_weekly.rename(columns={"defteam":"team","posteam":"opp"})

    weekly = pd.merge(off_weekly, def_weekly, on=["season","week","team","opp"], how="outer").fillna(0)
    weekly = weekly.sort_values(["season", "week", "team"]).reset_index(drop=True)

    # Team summary
    off_team = off_weekly.groupby(["season", "team"]).agg(
        rz_trips=("rz_trips", "sum"),
        rz_td=("rz_td", "sum"),
        rz_fg=("rz_fg", "sum"),
        rz_empty=("rz_empty", "sum"),
    ).reset_index()
    off_team["rz_efficiency"] = (off_team["rz_td"] / off_team["rz_trips"].where(off_team["rz_trips"] > 0, pd.NA)).astype(float).round(4)

    def_team = def_weekly.groupby(["season", "team"]).agg(
        rz_trips_allowed=("rz_trips_allowed", "sum"),
        rz_td_allowed=("rz_td_allowed", "sum"),
        rz_fg_allowed=("rz_fg_allowed", "sum"),
        rz_empty_allowed=("rz_empty_allowed", "sum"),
    ).reset_index()
    def_team["rz_efficiency_allowed"] = (def_team["rz_td_allowed"] / def_team["rz_trips_allowed"].where(def_team["rz_trips_allowed"] > 0, pd.NA)).astype(float).round(4)

    team = pd.merge(off_team, def_team, on=["season","team"], how="outer").fillna(0)
    team = team.sort_values(["season", "team"]).reset_index(drop=True)

    return weekly, team

def main():
    ap = argparse.ArgumentParser(description="Build Red Zone Efficiency (Offense/Defense)")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=str, required=True)
    ap.add_argument("--out_weekly", type=str, required=True)
    ap.add_argument("--out_team", type=str, required=True)
    ap.add_argument("--debug", action="store_true", help="Wypisz listę kolumn i przerwij")
    args = ap.parse_args()

    in_path = Path(args.in_pbp)
    if not in_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku PBP: {in_path}")

    print(f"📥 Wczytuję PBP: {in_path}")
    pbp = pd.read_parquet(in_path)

    if args.debug:
        print("🔎 Kolumny w PBP:", _pretty_sample_columns(pbp, 200))
        return

    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == args.season].copy()

    weekly, team = compute_redzone_metrics(pbp)

    outw = Path(args.out_weekly); outw.parent.mkdir(parents=True, exist_ok=True)
    outt = Path(args.out_team);  outt.parent.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(outw, index=False)
    team.to_csv(outt, index=False)

    print(f"✅ Zapisano weekly: {outw} (rows={len(weekly)})")
    print(f"🏁 Zapisano team:   {outt} (rows={len(team)})")
    print("🎯 Red Zone Efficiency — GOTOWE.")

if __name__ == "__main__":
    main()
