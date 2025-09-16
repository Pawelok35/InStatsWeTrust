import pandas as pd
from etl.build_explosives_allowed import build

def test_build_explosives_allowed_basic():
    # Minimalny syntetyczny set
    data = [
        # def plays vs TEAMA
        {"season": 2024, "week": 1, "defteam": "TEAMA", "play_type": "run",  "yards_gained": 9},
        {"season": 2024, "week": 1, "defteam": "TEAMA", "play_type": "run",  "yards_gained": 10},
        {"season": 2024, "week": 1, "defteam": "TEAMA", "play_type": "pass", "yards_gained": 14},
        {"season": 2024, "week": 1, "defteam": "TEAMA", "play_type": "pass", "yards_gained": 15},
        {"season": 2024, "week": 1, "defteam": "TEAMA", "play_type": "pass", "yards_gained": 21},
        # drugi tydzien
        {"season": 2024, "week": 2, "defteam": "TEAMA", "play_type": "run",  "yards_gained": 25},
    ]
    df = pd.DataFrame(data)

    weekly, team = build(df, rush_yards=10, pass_yards=15, track_chunk20=True)

    # Week 1: rush_expl=1 (10+), pass_expl=2 (15+, 21), chunk20=1 (21)
    w1 = weekly[(weekly.defteam=="TEAMA") & (weekly.week==1)].iloc[0]
    assert w1.expl_rush_allowed == 1
    assert w1.expl_pass_allowed == 2
    assert w1.expl_total_allowed == 3
    assert w1.chunk20_allowed == 1
    assert w1.def_plays == 5

    # Team totals: + week 2 rush 25 => extra rush_expl + chunk20
    t = team[team.defteam=="TEAMA"].iloc[0]
    assert t.expl_rush_allowed == 2
    assert t.expl_pass_allowed == 2
    assert t.expl_total_allowed == 4
    assert t.chunk20_allowed == 2
    assert t.def_plays == 6
