import pandas as pd
from pathlib import Path

SEASON = 2025
P = Path("data/processed")

def patch_third_down_alias():
    # Alias: def_3d_epa_allowed_per_play -> def_3d_epa_per_play
    for name in [f"third_down_team_{SEASON}.csv", f"third_down_weekly_{SEASON}.csv"]:
        fp = P / name
        if not fp.exists(): 
            continue
        df = pd.read_csv(fp)
        changed = False
        if "def_3d_epa_per_play" not in df.columns:
            cand = "def_3d_epa_allowed_per_play"
            if cand in df.columns:
                df["def_3d_epa_per_play"] = df[cand]
                changed = True
        if changed:
            df.to_csv(fp, index=False)
            print(f"[OK] {name}: aliased def_3d_epa_per_play")

def patch_field_pos_advantage_from_weekly():
    # Liczymy OFF/DEF start_yardline_100_avg ważone liczbą drives z weekly i z tego field_pos_advantage
    weekly = P / f"drive_efficiency_weekly_{SEASON}.csv"
    team_fp = P / f"drive_efficiency_team_{SEASON}.csv"
    if not (weekly.exists() and team_fp.exists()):
        return
    w = pd.read_csv(weekly)
    if not {"team","side","drives","start_yardline_100_avg"}.issubset(w.columns):
        return

    # ważona średnia start_yardline_100_avg po OFF i DEF
    w["w_start"] = w["start_yardline_100_avg"] * w["drives"].clip(lower=0).fillna(0)
    g = w.groupby(["team","side"], as_index=False).agg(w_sum=("w_start","sum"),
                                                      d_sum=("drives","sum"))
    g["avg_w"] = g["w_sum"] / g["d_sum"].replace(0, pd.NA)

    off = g[g["side"].str.lower()=="off"][["team","avg_w"]].rename(columns={"avg_w":"start100_off_w"})
    deff = g[g["side"].str.lower()=="def"][["team","avg_w"]].rename(columns={"avg_w":"start100_def_w"})
    m = off.merge(deff, on="team", how="outer")
    m["field_pos_advantage"] = m["start100_off_w"] - m["start100_def_w"]

    t = pd.read_csv(team_fp)
    if "field_pos_advantage" not in t.columns:
        t = t.merge(m[["team","field_pos_advantage"]], on="team", how="left")
        t.to_csv(team_fp, index=False)
        print(f"[OK] drive_efficiency_team_{SEASON}.csv: added field_pos_advantage (weighted)")

def patch_rz_td_rate():
    # Priorytet: redzone_team/weeky; fallback: policz z drive_efficiency_weekly
    red_team = P / f"redzone_team_{SEASON}.csv"
    red_week = P / f"redzone_weekly_{SEASON}.csv"
    changed = False

    # 1) Spróbuj dodać rz_td_rate w redzone_*
    for fp in [red_team, red_week]:
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if "rz_td_rate" not in df.columns:
            have = {"td_drives","rz_drives"}.issubset(df.columns)
            if have:
                df["rz_td_rate"] = df["td_drives"].astype(float) / df["rz_drives"].replace(0, pd.NA)
                df.to_csv(fp, index=False)
                print(f"[OK] {fp.name}: added rz_td_rate")
                changed = True

    # 2) Fallback: policz teamowy rz_td_rate z drive_efficiency_weekly i wlej do redzone_team
    if red_team.exists():
        df_team = pd.read_csv(red_team)
    else:
        df_team = pd.DataFrame({"season":[],"team":[]})

    if "rz_td_rate" not in df_team.columns:
        dew = P / f"drive_efficiency_weekly_{SEASON}.csv"
        if dew.exists():
            w = pd.read_csv(dew)
            if {"team","td_drives","rz_drives"}.issubset(w.columns):
                agg = w.groupby("team", as_index=False).agg(td_drives=("td_drives","sum"),
                                                            rz_drives=("rz_drives","sum"))
                agg["rz_td_rate"] = agg["td_drives"].astype(float) / agg["rz_drives"].replace(0, pd.NA)
                # wlej/merge
                if df_team.empty:
                    agg.insert(0, "season", SEASON)
                    df_team = agg
                else:
                    if "team" in df_team.columns:
                        df_team = df_team.merge(agg[["team","rz_td_rate"]], on="team", how="left")
                    else:
                        agg.insert(0, "season", SEASON)
                        df_team = agg
                df_team.to_csv(red_team, index=False)
                print(f"[OK] redzone_team_{SEASON}.csv: filled rz_td_rate from drive_efficiency_weekly fallback")

def main():
    patch_third_down_alias()
    patch_field_pos_advantage_from_weekly()
    patch_rz_td_rate()

if __name__ == "__main__":
    main()
