import numpy as np

DIFFS = ["net_epa","off_epa_per_play","def_epa_per_play_allowed","momentum_3w"]

def test_game_features_has_diffs(game_features_csv):
    gf = game_features_csv
    for c in DIFFS:
        assert f"diff_{c}" in gf.columns, f"Missing diff_{c}"

def test_diff_values_where_available(game_features_csv):
    gf = game_features_csv.copy()
    # Sprawdź na podzbiorze bez NaN w kolumnach bazowych
    mask = np.ones(len(gf), dtype=bool)
    for c in DIFFS:
        mask &= gf[f"home_{c}"].notna() & gf[f"away_{c}"].notna()

    sample = gf[mask].head(50)  # wystarczy próbka
    for c in DIFFS:
        expected = sample[f"home_{c}"] - sample[f"away_{c}"]
        assert np.allclose(sample[f"diff_{c}"].values, expected.values, equal_nan=True)
