#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def _read_any(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix.lower() in {".parquet",".pq"} else pd.read_csv(p, low_memory=False)

def _explosive_mask(df: pd.DataFrame) -> pd.Series:
    yd = pd.to_numeric(df.get("yards_gained"), errors="coerce").fillna(0)
    is_pass = pd.to_numeric(df.get("is_pass"), errors="coerce").fillna(0).astype(int)
    is_rush = pd.to_numeric(df.get("is_rush"), errors="coerce").fillna(0).astype(int)
    return ((is_pass == 1) & (yd >= 20)) | ((is_rush == 1) & (yd >= 10))

def build_off_weekly(df: pd.DataFrame) -> pd.DataFrame:
    st = pd.to_numeric(df.get("st_play"), errors="coerce").fillna(0).astype(int)
    off = df.loc[st == 0].copy()
    off["success"] = pd.to_numeric(off.get("success"), errors="coerce")
    off["epa"] = pd.to_numeric(off.get("epa"), errors="coerce")
    off["down"] = pd.to_numeric(off.get("down"), errors="coerce")
    off["yardline_100"] = pd.to_numeric(off.get("yardline_100"), errors="coerce")
    off["is_pass"] = pd.to_numeric(off.get("is_pass"), errors="coerce").fillna(0).astype(int)
    off["is_rush"] = pd.to_numeric(off.get("is_rush"), errors="coerce").fillna(0).astype(int)
    off["interception"] = pd.to_numeric(off.get("interception"), errors="coerce").fillna(0).astype(int)
    off["fumble"] = pd.to_numeric(off.get("fumble"), errors="coerce").fillna(0).astype(int)
    off["yards_gained"] = pd.to_numeric(off.get("yards_gained"), errors="coerce").fillna(0)

    grp = off.groupby(["season","week","posteam"], dropna=False)
    def agg(g: pd.DataFrame) -> pd.Series:
        e = g["epa"]; s = g["success"]; d = g["down"]; y = g["yardline_100"]
        ip = g["is_pass"]; ir = g["is_rush"]
        inter = g["interception"]; fmb = g["fumble"]; yd = g["yards_gained"]
        expl = _explosive_mask(g)
        return pd.Series({
            "plays": int(len(g)),
            "avg_epa": float(e.mean(skipna=True)),
            "median_epa": float(e.median(skipna=True)),
            "success_rate": float(s.mean(skipna=True)),
            "explosive_plays": int(expl.sum()),
            "explosive_rate": float(expl.mean(skipna=True)),
            "total_yards": float(yd.sum()),
            "early_down_epa": float(e[d.isin([1,2])].mean(skipna=True)),
            "late_down_epa": float(e[d.isin([3,4])].mean(skipna=True)),
            "third_down_sr": float(s[d==3].mean(skipna=True)),
            "fourth_down_sr": float(s[d==4].mean(skipna=True)),
            "red_zone_epa": float(e[y<=20].mean(skipna=True)),
            "pass_epa_per_play": float(e[ip==1].mean(skipna=True)),
            "rush_epa_per_play": float(e[ir==1].mean(skipna=True)),
            "turnover_epa": float(e[(inter==1)|(fmb==1)].sum(skipna=True)),
            "avg_start_yardline_100": float(y.mean(skipna=True)),
        })
    out = grp.apply(agg).reset_index().rename(columns={"posteam":"team"})
    return out

def build_def_weekly(df: pd.DataFrame) -> pd.DataFrame:
    st = pd.to_numeric(df.get("st_play"), errors="coerce").fillna(0).astype(int)
    deff = df.loc[st == 0].copy()
    deff["epa"] = pd.to_numeric(deff.get("epa"), errors="coerce")
    deff["success"] = pd.to_numeric(deff.get("success"), errors="coerce")
    deff["down"] = pd.to_numeric(deff.get("down"), errors="coerce")
    deff["yardline_100"] = pd.to_numeric(deff.get("yardline_100"), errors="coerce")
    deff["is_pass"] = pd.to_numeric(deff.get("is_pass"), errors="coerce").fillna(0).astype(int)
    deff["is_rush"] = pd.to_numeric(deff.get("is_rush"), errors="coerce").fillna(0).astype(int)
    deff["interception"] = pd.to_numeric(deff.get("interception"), errors="coerce").fillna(0).astype(int)
    deff["fumble"] = pd.to_numeric(deff.get("fumble"), errors="coerce").fillna(0).astype(int)
    deff["yards_gained"] = pd.to_numeric(deff.get("yards_gained"), errors="coerce").fillna(0)

    grp = deff.groupby(["season","week","defteam"], dropna=False)
    def agg(g: pd.DataFrame) -> pd.Series:
        e = g["epa"]; s = g["success"]; d = g["down"]; y = g["yardline_100"]
        ip = g["is_pass"]; ir = g["is_rush"]
        inter = g["interception"]; fmb = g["fumble"]; yd = g["yards_gained"]
        expl = _explosive_mask(g)
        return pd.Series({
            "plays_allowed": int(len(g)),
            "avg_epa_allowed": float(e.mean(skipna=True)),
            "median_epa_allowed": float(e.median(skipna=True)),
            "success_rate_allowed": float(s.mean(skipna=True)),
            "explosive_allowed": int(expl.sum()),
            "explosive_rate_allowed": float(expl.mean(skipna=True)),
            "yards_allowed": float(yd.sum()),
            "early_down_epa_allowed": float(e[d.isin([1,2])].mean(skipna=True)),
            "late_down_epa_allowed": float(e[d.isin([3,4])].mean(skipna=True)),
            "third_down_sr_allowed": float(s[d==3].mean(skipna=True)),
            "fourth_down_sr_allowed": float(s[d==4].mean(skipna=True)),
            "red_zone_epa_allowed": float(e[y<=20].mean(skipna=True)),
            "pass_epa_per_play_allowed": float(e[ip==1].mean(skipna=True)),
            "rush_epa_per_play_allowed": float(e[ir==1].mean(skipna=True)),
            "turnover_epa_forced": float(e[(inter==1)|(fmb==1)].sum(skipna=True)),
            "avg_start_yardline_100_faced": float(y.mean(skipna=True)),
        })
    out = grp.apply(agg).reset_index().rename(columns={"defteam":"team"})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in_pbp", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    df = _read_any(args.in_pbp)
    df = df[df["season"] == args.season].copy()

    off = build_off_weekly(df)
    deff = build_def_weekly(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    off_p = args.out_dir / f"epa_offense_summary_{args.season}_weekly.csv"
    deff_p = args.out_dir / f"epa_defense_summary_{args.season}_weekly.csv"
    off.to_csv(off_p, index=False)
    deff.to_csv(deff_p, index=False)
    print(f"✅ Saved: {off_p}")
    print(f"✅ Saved: {deff_p}")

if __name__ == "__main__":
    main()
