$SEASON = 2025
$WEEK   = 4
$IN_PBP = "data/processed/pbp_clean_${SEASON}.parquet"
$OUT_DIR = "data/processed"

Write-Host "=== Building Core 12 metrics ==="

python etl/build_drive_efficiency.py   --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/drive_efficiency_weekly_${SEASON}.csv" --out_team "$OUT_DIR/drive_efficiency_team_${SEASON}.csv"
python etl/build_redzone.py            --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/redzone_weekly_${SEASON}.csv" --out_team "$OUT_DIR/redzone_team_${SEASON}.csv"
python etl/build_success_rate.py       --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/success_rate_weekly_${SEASON}.csv" --out_team "$OUT_DIR/success_rate_team_${SEASON}.csv"
python etl/build_tempo.py              --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/tempo_weekly_${SEASON}.csv" --out_team "$OUT_DIR/tempo_team_${SEASON}.csv"
python etl/build_fourth_down.py        --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/fourth_down_weekly_${SEASON}.csv" --out_team "$OUT_DIR/fourth_down_team_${SEASON}.csv"
python etl/build_field_position.py     --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/field_position_weekly_${SEASON}.csv" --out_team "$OUT_DIR/field_position_team_${SEASON}.csv"
python etl/build_hidden_yards.py       --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/hidden_yards_weekly_${SEASON}.csv" --out_team "$OUT_DIR/hidden_yards_team_${SEASON}.csv"
python etl/build_penalty_discipline.py --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/penalty_discipline_weekly_${SEASON}.csv" --out_team "$OUT_DIR/penalty_discipline_team_${SEASON}.csv"
python etl/build_qb_pressure.py        --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/qb_pressure_weekly_${SEASON}.csv" --out_team "$OUT_DIR/qb_pressure_team_${SEASON}.csv"
python etl/build_run_block.py          --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/run_block_weekly_${SEASON}.csv" --out_team "$OUT_DIR/run_block_team_${SEASON}.csv"
python etl/build_pass_block.py         --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/pass_block_weekly_${SEASON}.csv" --out_team "$OUT_DIR/pass_block_team_${SEASON}.csv"
python etl/build_rolling_form.py       --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/rolling_form_weekly_${SEASON}.csv" --out_team "$OUT_DIR/rolling_form_team_${SEASON}.csv"

Write-Host "=== Building Bonus 4 metrics ==="

python etl/build_explosive_plays.py    --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/explosive_plays_weekly_${SEASON}.csv" --out_team "$OUT_DIR/explosive_plays_team_${SEASON}.csv"
python etl/build_turnover_luck.py      --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/turnover_luck_weekly_${SEASON}.csv" --out_team "$OUT_DIR/turnover_luck_team_${SEASON}.csv"
python etl/build_clutch_index.py       --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/clutch_index_weekly_${SEASON}.csv" --out_team "$OUT_DIR/clutch_index_team_${SEASON}.csv"
python etl/build_special_teams.py      --season $SEASON --in_pbp $IN_PBP --out_weekly "$OUT_DIR/special_teams_weekly_${SEASON}.csv" --out_team "$OUT_DIR/special_teams_team_${SEASON}.csv"

Write-Host "=== Building matchup for Week $WEEK ==="


