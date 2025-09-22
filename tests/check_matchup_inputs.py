#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys
from typing import List, Dict

def find_csvs(processed_dir: Path, season: int) -> List[Path]:
    patterns = [
        f"*_*_team_{season}.csv",
        f"*_*_weekly_{season}.csv",
        f"*team_{season}.csv",
        f"*weekly_{season}.csv",
    ]
    out = []
    for pat in patterns:
        out.extend(processed_dir.glob(pat))
    # Deduplicate while keeping order
    seen = set()
    unique = []
    for p in out:
        if p.name not in seen:
            unique.append(p)
            seen.add(p.name)
    return unique

EXPECTED_CORE = [
    # Drive / volume
    "yds","plays","drives","yds_per_drive","plays_per_drive",
    "ppd_basic","ppd_basic_sum",
    "start_own_yardline_avg","start_yardline_100_avg",
    # Red zone
    "rz_drives","td_drives","fg_drives","rz_td_rate","redzone_drive_rate",
    # Third down
    "off_3d_avg_togo","def_3d_allowed_conv","off_3d_conv","off_3d_att",
    "def_3d_epa_per_play","off_3d_epa_per_play","conv_rate",
    # Explosives / allowed
    "expl_pass_allowed","expl_rush_allowed","expl_total_allowed","expl_rate_allowed",
    # Hidden yards / field position
    "field_pos_advantage","avg_start_yardline_100_off",
    # Penalties
    "pen_per_100_plays_w","penalty_yds_pg",
    # Tempo / SR / EPA (generic)
    "sr","epa_per_play",
    # Special teams / turnovers
    "st_score","takeaways","pot_per_takeaway","pot_points",
    # Score / rate families (often used)
    "score_drives","score_rate","td_rate","fg_rate","punt_drives","punt_rate","turnover_drives","turnover_rate",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"[ERROR] Processed dir not found: {processed_dir}", file=sys.stderr)
        sys.exit(2)

    csvs = find_csvs(processed_dir, args.season)
    if not csvs:
        print(f"[WARN] No CSVs found under {processed_dir} for season {args.season}")
        sys.exit(0)

    # Collect columns per file
    rows = []
    col_to_files: Dict[str, set] = {}
    for fp in csvs:
        try:
            df_head = pd.read_csv(fp, nrows=0)
            cols = list(df_head.columns)
        except Exception as e:
            print(f"[WARN] Failed to read {fp.name}: {e}", file=sys.stderr)
            cols = []

        for c in cols:
            rows.append({"file": fp.name, "column": c})
            col_to_files.setdefault(c, set()).add(fp.name)

    inv = pd.DataFrame(rows).sort_values(["file","column"])

    # Coverage for expected core set
    expected_status = []
    for key in EXPECTED_CORE:
        hit_files = sorted(col_to_files.get(key, []))
        expected_status.append({
            "column": key,
            "present": bool(hit_files),
            "files": ", ".join(hit_files) if hit_files else ""
        })
    coverage = pd.DataFrame(expected_status).sort_values(["present","column"], ascending=[False, True])

    # Print summary to console
    print(f"=== Inventory: files found ({len(csvs)}) ===")
    for fp in csvs:
        print(" -", fp.name)
    print("\n=== Coverage of expected core columns ===")
    print(coverage.to_string(index=False))

    # Save full inventory to CSV
    out_dir = Path("tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"columns_inventory_{args.season}.csv"
    inv.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote detailed inventory to: {out_csv}")

if __name__ == "__main__":
    main()
