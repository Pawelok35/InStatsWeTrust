# tests/test_points_off_turnovers_debug.py
import pandas as pd
from etl.build_points_off_turnovers import (
    compute_pot, detect_takeaways, build_series, compute_play_points
)

SEASON = 2024

def make_df(rows):
    df = pd.DataFrame(rows)
    if "play_id" not in df.columns:
        df["play_id"] = range(1, len(df) + 1)
    return df

def test_pick_six_same_play_only_6_points():
    # A przechwytuje i zdobywa pick-six w tym samym playu (6).
    # XP jest OSOBNYM playem i nie liczy się do same-play PoT.
    rows = [
        # prelude
        dict(season=SEASON, week=1, game_id="G1", home_team="H", away_team="A",
             posteam="H", defteam="A", desc="1&10 H 25", touchdown=0, field_goal_result=None),
        # interception + TD return (same play)
        dict(season=SEASON, week=1, game_id="G1", home_team="H", away_team="A",
             posteam="H", defteam="A", desc="Pass intercepted and returned for TD by A",
             touchdown=1, td_team="A"),
        # XP (nie liczymy do same-play)
        dict(season=SEASON, week=1, game_id="G1", home_team="H", away_team="A",
             posteam="A", defteam="H", desc="XP good", extra_point_result="good"),
    ]
    df = make_df(rows)
    weekly, team = compute_pot(df, SEASON)
    a = team.loc[team["team"] == "A"].iloc[0]
    assert a["pot_points"] == 6
    assert a["takeaways"] == 1
    assert a["pot_per_takeaway"] == 6

def test_takeaway_then_first_off_series_counts_td_plus_xp():
    # A przechwytuje (bez same-play TD), potem w pierwszej ofensywnej serii A: TD + XP → 7 PoT.
    rows = [
        # H ma piłkę
        dict(season=SEASON, week=1, game_id="G2", home_team="H", away_team="A",
             posteam="H", defteam="A", desc="1&10 H 20"),
        # interception (bez TD)
        dict(season=SEASON, week=1, game_id="G2", home_team="H", away_team="A",
             posteam="H", defteam="A", desc="Pass intercepted by A", touchdown=0),
        # zaczyna się ofensywna seria A
        dict(season=SEASON, week=1, game_id="G2", home_team="H", away_team="A",
             posteam="A", defteam="H", desc="A gains 8 yards"),
        dict(season=SEASON, week=1, game_id="G2", home_team="H", away_team="A",
             posteam="A", defteam="H", desc="A TD", touchdown=1, td_team="A"),
        dict(season=SEASON, week=1, game_id="G2", home_team="H", away_team="A",
             posteam="A", defteam="H", desc="XP good", extra_point_result="good"),
    ]
    df = make_df(rows)
    weekly, team = compute_pot(df, SEASON)
    a = team.loc[team["team"] == "A"].iloc[0]
    assert a["pot_points"] == 7
    assert a["takeaways"] == 1

def test_punt_change_of_possession_is_not_takeaway():
    # Punt z COP nie powinien być liczony jako takeaway.
    rows = [
        dict(season=SEASON, week=1, game_id="G3", home_team="H", away_team="A",
             posteam="H", defteam="A", play_type="punt", desc="Punt, possession changes"),
        # Po puncie A ma piłkę, ale nie był to takeaway w rozumieniu PoT.
        dict(season=SEASON, week=1, game_id="G3", home_team="H", away_team="A",
             posteam="A", defteam="H", desc="A drive starts"),
    ]
    df = make_df(rows)
    df = detect_takeaways(df)
    # Zero takeawayów:
    assert int(df["is_takeaway"].sum()) == 0

def test_infer_takeaway_team_from_next_series_when_defteam_missing():
    # Brak defteam na playu przechwytu → inferujemy zespół przechwytu na podstawie
    # właściciela następnej serii. H ma piłkę, dochodzi do przechwytu (w opisie),
    # następna seria należy do A → takeaway_team = A → potem A zdobywa FG=3 → PoT=3.
    rows = [
        dict(season=SEASON, week=1, game_id="G4", home_team="H", away_team="A",
             posteam="H", desc="1&10 H 25"),
        dict(season=SEASON, week=1, game_id="G4", home_team="H", away_team="A",
             posteam="H", desc="Pass intercepted by A at midfield"),
        # Nowa seria A
        dict(season=SEASON, week=1, game_id="G4", home_team="H", away_team="A",
             posteam="A", desc="A short gain"),
        dict(season=SEASON, week=1, game_id="G4", home_team="H", away_team="A",
             posteam="A", desc="FG good", field_goal_result="made"),
    ]
    df = make_df(rows)
    weekly, team = compute_pot(df, SEASON)
    a = team.loc[team["team"] == "A"].iloc[0]
    assert a["takeaways"] == 1
    assert a["pot_points"] == 3
