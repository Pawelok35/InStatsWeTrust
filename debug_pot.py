import pandas as pd, random
df = pd.read_parquet("data/processed/pbp_clean_2024.parquet")
from etl.build_points_off_turnovers import detect_takeaways, build_series, compute_play_points
df = detect_takeaways(df)
df = build_series(df)
pts_scorer = df.apply(compute_play_points, axis=1)
df["play_points"] = [t[0] for t in pts_scorer]
df["play_scorer"] = [t[1] for t in pts_scorer]
cand = df.loc[df["is_takeaway"], ["game_id","week","posteam","defteam","takeaway_team","play_type",
                                  "desc","play_description","series_id","series_owner"]].copy()
print("Sample takeaways:", len(cand))
print(cand.sample(min(8, len(cand))).to_string(index=False))
