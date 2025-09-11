def test_momentum_columns_exist(weekly_csv):
    df = weekly_csv
    required = {
        "season","week","team",
        "momentum_3w",
        "net_epa_roll3","net_epa_delta3",
        "off_epa_per_play_roll3","off_epa_per_play_delta3",
        "def_epa_per_play_allowed_roll3","def_epa_per_play_allowed_delta3",
    }
    assert required.issubset(df.columns), f"Missing: {required - set(df.columns)}"

def test_week_monotonic_per_team(weekly_csv):
    df = weekly_csv
    ok = True
    for t, g in df.groupby("team"):
        if not g["week"].is_monotonic_increasing:
            ok = False
            break
    assert ok, "Weeks must be monotonic increasing within each team"
