# etl/build_weekly_summaries.py
import pandas as pd
from pathlib import Path
import argparse

try:
    import nfl_data_py as nfl
except Exception as e:
    raise SystemExit("❌ Brak pakietu nfl_data_py. Zainstaluj: pip install nfl-data-py") from e


def import_pbp(season: int) -> pd.DataFrame:
    # PBP: jedna ramka dla całego sezonu
    df = nfl.import_pbp_data([season])
    # minimalny zestaw kolumn
    need = {
        "season","week","posteam","defteam","epa","down","yards_gained",
        "yardline_100","pass_attempt","rush_attempt","success",
        "interception","fumble_lost",
    }
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ Brak kolumn w PBP: {missing}")
    # standaryzacja typów
    for c in ["pass_attempt","rush_attempt","success","interception","fumble_lost"]:
        df[c] = df[c].fillna(0).astype(int)
    df["down"] = df["down"].fillna(0).astype(int)
    return df


def summarize_offense_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp["explosive_off"] = ((pbp["rush_attempt"]==1) & (pbp["yards_gained"]>=10)) | \
                           ((pbp["pass_attempt"]==1) & (pbp["yards_gained"]>=15))
    pbp["early"] = pbp["down"].isin([1,2])
    pbp["late"]  = pbp["down"].isin([3,4])
    pbp["rz"]    = pbp["yardline_100"]<=20

    g = pbp.groupby(["season","week","posteam"], as_index=False)
    off = g.agg(
        plays=("epa","size"),
        avg_epa=("epa","mean"),
        median_epa=("epa","median"),
        success_rate=("success","mean"),
        explosive_plays=("explosive_off","sum"),
        explosive_rate=("explosive_off","mean"),
        total_yards=("yards_gained","sum"),
        early_down_epa=("epa", lambda s: s[pbp.loc[s.index,"early"]].mean()),
        late_down_epa=("epa", lambda s: s[pbp.loc[s.index,"late"]].mean()),
        third_down_sr=("success", lambda s: s[pbp.loc[s.index,"down"]==3].mean()),
        fourth_down_sr=("success", lambda s: s[pbp.loc[s.index,"down"]==4].mean()),
        red_zone_epa=("epa", lambda s: s[pbp.loc[s.index,"rz"]].mean()),
        pass_epa_per_play=("epa", lambda s: s[pbp.loc[s.index,"pass_attempt"]==1].mean()),
        rush_epa_per_play=("epa", lambda s: s[pbp.loc[s.index,"rush_attempt"]==1].mean()),
        turnover_epa=("epa", lambda s: s[(pbp.loc[s.index,"interception"]==1) | (pbp.loc[s.index,"fumble_lost"]==1)].sum()),
        avg_start_yardline_100=("yardline_100","mean"),
    ).rename(columns={"posteam":"team"})
    return off.fillna(0.0)


def summarize_defense_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    # te same definicje, ale z perspektywy drużyny w obronie (defteam)
    pbp = pbp.copy()
    pbp["explosive_off"] = ((pbp["rush_attempt"]==1) & (pbp["yards_gained"]>=10)) | \
                           ((pbp["pass_attempt"]==1) & (pbp["yards_gained"]>=15))
    pbp["early"] = pbp["down"].isin([1,2])
    pbp["late"]  = pbp["down"].isin([3,4])
    pbp["rz"]    = pbp["yardline_100"]<=20

    g = pbp.groupby(["season","week","defteam"], as_index=False)
    deff = g.agg(
        plays_allowed=("epa","size"),
        avg_epa_allowed=("epa","mean"),
        median_epa_allowed=("epa","median"),
        success_rate_allowed=("success","mean"),
        explosive_allowed=("explosive_off","sum"),
        explosive_rate_allowed=("explosive_off","mean"),
        yards_allowed=("yards_gained","sum"),
        early_down_epa_allowed=("epa", lambda s: s[pbp.loc[s.index,"early"]].mean()),
        late_down_epa_allowed=("epa", lambda s: s[pbp.loc[s.index,"late"]].mean()),
        third_down_sr_allowed=("success", lambda s: s[pbp.loc[s.index,"down"]==3].mean()),
        fourth_down_sr_allowed=("success", lambda s: s[pbp.loc[s.index,"down"]==4].mean()),
        red_zone_epa_allowed=("epa", lambda s: s[pbp.loc[s.index,"rz"]].mean()),
        pass_epa_per_play_allowed=("epa", lambda s: s[pbp.loc[s.index,"pass_attempt"]==1].mean()),
        rush_epa_per_play_allowed=("epa", lambda s: s[pbp.loc[s.index,"rush_attempt"]==1].mean()),
        # turnover_epa_forced: korzyść obrony przy stratach ataku = ujemne EPA ataku → bierzemy wartość dodatnią dla defense
        turnover_epa_forced=("epa", lambda s: -s[(pbp.loc[s.index,"interception"]==1) | (pbp.loc[s.index,"fumble_lost"]==1)].sum()),
        avg_start_yardline_100_faced=("yardline_100","mean"),
    ).rename(columns={"defteam":"team"})
    return deff.fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    pbp = import_pbp(args.season)
    off_w = summarize_offense_weekly(pbp)
    def_w = summarize_defense_weekly(pbp)

    off_p = args.out_dir / f"epa_offense_summary_{args.season}_weekly.csv"
    def_p = args.out_dir / f"epa_defense_summary_{args.season}_weekly.csv"
    off_w.to_csv(off_p, index=False)
    def_w.to_csv(def_p, index=False)

    print(f"✅ Zapisano: {off_p}")
    print(f"✅ Zapisano: {def_p}")


if __name__ == "__main__":
    main()
