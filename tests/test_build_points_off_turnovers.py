import pandas as pd
from etl.build_points_off_turnovers import compute_pot

def test_pot_basic_pick_six():
    # Minimalny syntetyczny game z pick-six
    data = [
        # prelude
        dict(season=2024, week=1, game_id="G1", row=0, home_team="H", away_team="A",
             posteam="H", defteam="A", drive=1, total_home_score=0, total_away_score=0,
             interception=0, fumble_lost=0),
        # takeaway + pick-six by A (defteam)
        dict(season=2024, week=1, game_id="G1", row=1, home_team="H", away_team="A",
             posteam="H", defteam="A", drive=1, total_home_score=0, total_away_score=6,
             interception=1, fumble_lost=0),
        # XP (osobny play, nie liczymy do same-play)
        dict(season=2024, week=1, game_id="G1", row=2, home_team="H", away_team="A",
             posteam="A", defteam="H", drive=2, total_home_score=0, total_away_score=7,
             interception=0, fumble_lost=0),
    ]
    df = pd.DataFrame(data)
    df["play_id"] = df["row"]
    weekly, team = compute_pot(df, 2024)

    # Team A powinien mieć 6 punktów PoT (pick-six same play)
    t = team.loc[team["team"]=="A"].iloc[0]
    assert t["pot_points"] == 6
    assert t["takeaways"] == 1
    assert t["pot_per_takeaway"] == 6
