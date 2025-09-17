import pandas as pd

team = pd.read_csv("data/processed/special_teams_team_2024.csv")
assert len(team)==32, f"Expect 32 teams, got {len(team)}"

cols = ["st_score","st_epa_per_play","ko_tb_rate","ko_opp_start_yd100","punt_opp_start_yd100","st_pen_rate"]
missing = [c for c in cols if c not in team.columns]
assert not missing, f"Missing columns: {missing}"

print("Rows:", len(team))
print("\nTop 5 ST Score:")
print(team.sort_values("st_score", ascending=False).head(5)[["team","st_score","st_epa_per_play"]])
print("\nBest (low) opp start after punts:")
print(team.sort_values("punt_opp_start_yd100").head(5)[["team","punt_opp_start_yd100"]])
print("\nKickoff TB% leaders:")
print(team.sort_values("ko_tb_rate", ascending=False).head(5)[["team","ko_tb_rate"]])
