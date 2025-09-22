param(
  [int]$Season = 2025,
  [string]$PbpOut = "data/processed/pbp_clean_2025.parquet"
)

Write-Host "▶ Ingest PBP for season $Season..." -ForegroundColor Cyan
& python etl/ingest.py --season $Season --out $PbpOut --mode full
if ($LASTEXITCODE -ne 0) { throw "Ingest failed" }

# === Bonus 9 / rozszerzenia — uruchom to, co masz w repo ===
# 3rd Down Efficiency
& python etl/build_third_down.py --season $Season --in_pbp $PbpOut `
  --out_weekly "data/processed/third_down_weekly_${Season}.csv" `
  --out_team   "data/processed/third_down_team_${Season}.csv"

# Red Zone Efficiency
& python etl/build_redzone.py --season $Season --in_pbp $PbpOut `
  --out_weekly "data/processed/redzone_weekly_${Season}.csv" `
  --out_team   "data/processed/redzone_team_${Season}.csv"

# Points off Turnovers
& python etl/build_points_off_turnovers.py --season $Season --in_pbp $PbpOut `
  --out_weekly "data/processed/points_off_turnovers_weekly_${Season}.csv" `
  --out_team   "data/processed/points_off_turnovers_team_${Season}.csv"

# Second-Half Adjustments
& python etl/build_second_half.py --season $Season --in_pbp $PbpOut `
  --out_weekly "data/processed/second_half_weekly_${Season}.csv" `
  --out_team   "data/processed/second_half_team_${Season}.csv"

# Special Teams Impact
& python etl/build_special_teams.py --season $Season --in_pbp $PbpOut `
  --out_weekly "data/processed/special_teams_weekly_${Season}.csv" `
  --out_team   "data/processed/special_teams_team_${Season}.csv"

# 1st Down Success  (jeśli masz ten plik w etl/)
if (Test-Path "etl/build_first_down_success.py") {
  & python etl/build_first_down_success.py --season $Season --in_pbp $PbpOut `
    --out_weekly "data/processed/first_down_success_weekly_${Season}.csv" `
    --out_team   "data/processed/first_down_success_team_${Season}.csv"
}

# Penalty Discipline  (jeśli masz ten plik w etl/)
if (Test-Path "etl/build_penalty_discipline.py") {
  & python etl/build_penalty_discipline.py --season $Season --in_pbp $PbpOut `
    --out_weekly "data/processed/penalty_discipline_weekly_${Season}.csv" `
    --out_team   "data/processed/penalty_discipline_team_${Season}.csv"
}

# Drive Efficiency  (jeśli masz ten plik w etl/)
if (Test-Path "etl/build_drive_efficiency.py") {
  & python etl/build_drive_efficiency.py --season $Season --in_pbp $PbpOut `
    --out_weekly "data/processed/drive_efficiency_weekly_${Season}.csv" `
    --out_team   "data/processed/drive_efficiency_team_${Season}.csv"
}

# Explosives Allowed (Defense)  (jeśli masz ten plik w etl/)
if (Test-Path "etl/build_explosives_allowed.py") {
  & python etl/build_explosives_allowed.py --season $Season --in_pbp $PbpOut `
    --out_weekly "data/processed/explosives_allowed_weekly_${Season}.csv" `
    --out_team   "data/processed/explosives_allowed_team_${Season}.csv"
}

Write-Host "`n✅ Season ${Season}: metrics built." -ForegroundColor Green

