def test_core12_has_key_columns(core12_csv):
    df = core12_csv
    required = {
        "season","team","rank_net_epa","net_epa",
        "off_epa_per_play","def_epa_per_play_allowed",
        "net_early_down_epa","net_late_down_epa",
        "net_third_down_sr","net_fourth_down_sr",
        "net_explosive_rate","net_red_zone_epa",
        "pass_rush_delta_off","turnover_epa_net","field_pos_advantage",
    }
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

def test_power_ranking_sorted(power_csv):
    pr = power_csv.copy()
    by_season_ok = True
    for season, g in pr.groupby("season"):
        if not g["net_epa"].is_monotonic_decreasing:
            by_season_ok = False
            break
    assert by_season_ok, "power_ranking not sorted by net_epa desc within season"

