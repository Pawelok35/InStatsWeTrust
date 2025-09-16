import pandas as pd
import nfl_data_py as nfl

SEASON = 2024
pbp = nfl.import_pbp_data([SEASON], downcast=False, cache=False)

cols = [
    "season","week","game_id","game_date","posteam","defteam",
    "home_team","away_team","down","ydstogo","yardline_100",
    "qtr","time","play_id","play_type","yards_gained","desc",
    "epa","wp","wpa","first_down","first_down_rush","first_down_pass",
    "first_down_penalty","qb_dropback","qb_spike","qb_kneel",
    "touchdown","pass","rush","penalty","penalty_team","no_play",
    "field_goal_result","interception","fumble_lost","turnover_on_downs","punt",
    # —— drive metadata (very helpful) ——
    "drive","drive_id","drive_play_id_started","drive_play_id_ended",
    "drive_start_yard_line","drive_end_yard_line"
]

have = [c for c in cols if c in pbp.columns]
pbp = pbp[have].copy()

out_path = "data/processed/pbp_clean_2024_new.parquet"
pbp.to_parquet(out_path, index=False)
print(f"✅ Zapisano: {out_path} (rows={len(pbp):,})")
