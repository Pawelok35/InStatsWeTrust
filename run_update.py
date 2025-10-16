#!/usr/bin/env python3
import os, sys, subprocess, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ETL  = ROOT / "etl"
OUT  = ROOT / "data" / "processed"

def run(cmd, stage):
    os.environ["PYTHONPATH"] = str(ROOT)  # fix imports
    print(f"\n▶ {stage}: {' '.join(shlex.quote(c) for c in cmd)}")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        sys.exit(f"❌ {stage} failed (rc={rc})")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    args = p.parse_args()
    season = args.season

    ROOT = Path(__file__).resolve().parent
    ETL  = ROOT / "etl"
    OUT  = ROOT / "data" / "processed"

    def run(cmd, stage):
        os.environ["PYTHONPATH"] = str(ROOT)
        print(f"\n▶ {stage}: {' '.join(shlex.quote(c) for c in cmd)}")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(f"❌ {stage} failed (rc={rc})")

    pbp = OUT / f"pbp_clean_{season}.parquet"

    # A) Update PBP (weeks 1–5) — zostawiamy jak było
    if (ETL / "scratch_update_w5.py").exists() and season == 2025:
        run([sys.executable, str(ETL / "scratch_update_w5.py")], "A) Update PBP")

    if not pbp.exists():
        sys.exit(f"Brak {pbp} po kroku A.")

    # B) EPA (season)
    run([sys.executable, str(ETL / "transform_epa_season.py"), str(pbp), str(OUT)],
        "B) Build EPA season summaries")

    # C) Core12 (season) — bez clutch, stabilnie
    core12_out = OUT / f"team_core12_{season}.csv"
    run([sys.executable, str(ETL / "build_core12.py"),
         "--season", str(season),
         "--in_dir", str(OUT),
         "--out", str(core12_out)],
        "C) Build Core12 season (no clutch)")

    # D) Drive Efficiency
    weekly = OUT / f"drive_efficiency_weekly_{season}.csv"
    team   = OUT / f"drive_efficiency_team_{season}.csv"
    rc = subprocess.call([sys.executable, str(ETL / "build_drive_efficiency.py"),
                          "--season", str(season),
                          "--in_pbp", str(pbp),
                          "--out_weekly", str(weekly),
                          "--out_team", str(team)], cwd=str(ROOT))
    if rc != 0:
        run([sys.executable, str(ETL / "build_drive_efficiency.py"),
             "--season", str(season),
             "--pbp_csv", str(pbp),
             "--out_dir", str(OUT)],
            "D) Drive Efficiency (fallback)")

    print("\n🎉 DONE. Kluczowe pliki zaktualizowane w data/processed:")
    print(f" • {pbp.name}")
    print(f" • epa_offense_summary_{season}_season.csv")
    print(f" • epa_defense_summary_{season}_season.csv")
    print(f" • {core12_out.name}")
    print(f" • {weekly.name} / {team.name}")

    season = 2025
    pbp = OUT / f"pbp_clean_{season}.parquet"

    # A) Update PBP (weeks 1–5)
    if (ETL / "scratch_update_w5.py").exists():
        run([sys.executable, str(ETL / "scratch_update_w5.py")], "A) Update PBP")

    if not pbp.exists():
        sys.exit(f"Brak {pbp} po kroku A.")

    # B) EPA (season) from clean PBP
    run([sys.executable, str(ETL / "transform_epa_season.py"), str(pbp), str(OUT)],
        "B) Build EPA season summaries")

    # C) Core12 (season) — bez clutch, stabilnie
    core12_out = OUT / f"team_core12_{season}.csv"
    run([sys.executable, str(ETL / "build_core12.py"),
        "--season", str(season),
        "--in_dir", str(OUT),
        "--out", str(core12_out)],
        "C) Build Core12 season (no clutch)")


    # D) Drive Efficiency (try two common CLI variants)
    weekly = OUT / f"drive_efficiency_weekly_{season}.csv"
    team   = OUT / f"drive_efficiency_team_{season}.csv"

    # variant 1 (nowszy)
    rc = subprocess.call([sys.executable, str(ETL / "build_drive_efficiency.py"),
                          "--season", str(season),
                          "--in_pbp", str(pbp),
                          "--out_weekly", str(weekly),
                          "--out_team", str(team)], cwd=str(ROOT))
    if rc != 0:
        # variant 2 (starsze argumenty)
        run([sys.executable, str(ETL / "build_drive_efficiency.py"),
             "--season", str(season),
             "--pbp_csv", str(pbp),
             "--out_dir", str(OUT)],
            "D) Drive Efficiency (fallback)")


    #  E) EPA weekly (off/def) z PBP
    wk_off = OUT / f"epa_offense_summary_{season}_weekly.csv"
    wk_def = OUT / f"epa_defense_summary_{season}_weekly.csv"
    if (ETL / "build_weekly_summaries.py").exists():
        run([sys.executable, str(ETL / "build_weekly_summaries.py"),
             "--season", str(season),
             "--in_pbp", str(pbp),
             "--out_dir", str(OUT)],
            "E) Build EPA weekly")
        # sanity-check
        if not wk_off.exists() or not wk_def.exists():
            sys.exit("❌ Weekly EPA nie powstało – sprawdź log kroku E.")
        else:
            print(f"   • {wk_off.name}")
            print(f"   • {wk_def.name}")
    else:
        print("ℹ️ Pomijam weekly EPA (brak etl/build_weekly_summaries.py)")

    # F) Core12 weekly
    if (ETL / "build_core12_weekly.py").exists():
        run([sys.executable, str(ETL / "build_core12_weekly.py")],
            "F) Build Core12 weekly")
    else:
        print("ℹ️ Pomijam Core12 weekly (brak etl/build_core12_weekly.py)")

    # Podsumowanie
    print("\n🎉 DONE. Kluczowe pliki zaktualizowane w data/processed:")
    print(f" • {pbp.name}")
    print(f" • epa_offense_summary_{season}_season.csv")
    print(f" • epa_defense_summary_{season}_season.csv")
    print(f" • {core12_out.name}")
    print(f" • {weekly.name} / {team.name}")
    if wk_off.exists() and wk_def.exists():
        print(f" • {wk_off.name} / {wk_def.name}")
    wk_core12 = OUT / f"team_core12_weekly_{season}.csv"
    if wk_core12.exists():
        print(f" • {wk_core12.name}")

if __name__ == "__main__":
    main()