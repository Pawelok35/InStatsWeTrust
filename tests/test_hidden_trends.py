from pathlib import Path
import pandas as pd
import pytest

from analysis.utils_hidden_trends import load_hidden_trends, compute_hidden_trends_edges, HIDDEN_TREND_COLS


BASE = Path(".")
SEASON = 2025


def _csv_path():
    return BASE / f"data/processed/team_hidden_trends_{SEASON}.csv"


@pytest.mark.skipif(not _csv_path().exists(), reason="Hidden Trends CSV not present")
def test_load_hidden_trends_has_expected_columns_and_unique_index():
    df = load_hidden_trends(SEASON, BASE)
    for c in HIDDEN_TREND_COLS:
        assert c in df.columns, f"Missing column: {c}"
    assert df.index.is_unique, "Index 'team' must be unique"
    # No bad aliases
    bad = {"LA", "NAN"}
    assert bad.isdisjoint(set(df.index)), f"Team alias leakage: {sorted(set(df.index) & bad)}"


@pytest.mark.skipif(not _csv_path().exists(), reason="Hidden Trends CSV not present")
def test_team_count_32_or_report_missing():
    df = load_hidden_trends(SEASON, BASE)
    n = df.index.nunique()
    assert n == 32, f"Expected 32 teams, got {n}. Missing: {sorted(set([t for t in df.index if isinstance(t, str)]) )}"


def test_compute_edges_nan_safe():
    # Minimal synthetic df to test NaN handling
    data = {
        "team": ["AAA", "BBB"],
        "game_rhythm_q4": [1.0, None],
        "play_call_entropy_neutral": [0.5, 0.4],
        "neutral_pass_rate": [0.6, 0.5],
        "neutral_plays": [100, 90],
        "drive_momentum_3plus": [0.3, 0.2],
        "drives_with_3plus": [10, 9],
        "drives_total": [30, 31],
        "field_flip_eff": [0.12, 0.15],
        "punts_tracked": [40, 38],
    }
    df = pd.DataFrame(data).set_index("team")
    comp = compute_hidden_trends_edges(df, "AAA", "BBB")
    assert comp["edges"]["play_call_entropy_neutral"] == pytest.approx(0.1)
    assert comp["edges"]["game_rhythm_q4"] is None  # NaN -> None

