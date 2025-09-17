def test_hidden_yards_column_present():
    import pandas as pd
    t = pd.read_csv('data/processed/drive_efficiency_team_2024.csv')
    assert 'hidden_yards_per_drive' in t.columns
    assert t['hidden_yards_per_drive'].notna().mean() == 1.0
