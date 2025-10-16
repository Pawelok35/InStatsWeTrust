param(
  [int]$SEASON = 2025,
  [int]$WEEK   = 5,
  [string]$PY  = "python"
)

$ErrorActionPreference = "Stop"

# --- Paths ---
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$Root\.."
$Repo = Get-Location
$InDir = "data/processed"
$NextW = $WEEK + 1
$MatchDir = Join-Path $InDir "matchups"
if (-not (Test-Path $MatchDir)) { New-Item -ItemType Directory -Path $MatchDir | Out-Null }

# --- Virtual Env ---
$Venv = Join-Path $Repo ".venv\Scripts\Activate.ps1"
if (Test-Path $Venv) { & $Venv }

# --- Python imports ---
$env:PYTHONPATH = "$Repo"

function Invoke-Step {
  param([string]$Title, [string[]]$Cmd)
  Write-Host "=== $Title ===" -ForegroundColor Cyan
  & $PY $Cmd
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title" }
  Write-Host "OK: $Title" -ForegroundColor Green
}

# --- 1) PBP update ---
$scratch = "etl/scratch_update_w$WEEK.py"
if (-not (Test-Path $scratch)) {
  $prev = "etl/scratch_update_w$($WEEK-1).py"
  if (Test-Path $prev) {
    Copy-Item $prev $scratch -Force
    (Get-Content $scratch -Raw) -replace "WEEK\s*=\s*$($WEEK-1)", "WEEK = $WEEK" | Set-Content $scratch -Encoding UTF8
    Write-Host "Created $scratch from $prev" -ForegroundColor Yellow
  } else {
    throw "Missing $scratch and $prev. Create etl/scratch_update_w$WEEK.py."
  }
}
Invoke-Step "PBP update (Week $WEEK)" @("$scratch")

# --- 2) Drive Efficiency ---
Invoke-Step "Build Drive Efficiency (team and weekly)" @(
  "-m","etl.build_drive_efficiency",
  "--season",$SEASON,
  "--in_pbp","$InDir/pbp_clean_$SEASON.parquet",
  "--out_weekly","$InDir/drive_efficiency_weekly_$SEASON.csv",
  "--out_team","$InDir/drive_efficiency_team_$SEASON.csv"
)

# --- 3) Core Metrics ---
$metrics = @(
  @{ title="Core12";          mod="etl.build_core12";                out="team_core12_$SEASON.csv" },
  @{ title="3rd Down";        mod="etl.build_third_down";            out="team_third_down_$SEASON.csv" },
  @{ title="Red Zone";        mod="etl.build_redzone";               out="team_redzone_$SEASON.csv" },
  @{ title="Explosives";      mod="etl.build_explosives_allowed";    out="team_explosives_allowed_$SEASON.csv" },
  @{ title="Penalties";       mod="etl.build_penalty_discipline";    out="team_penalties_$SEASON.csv" },
  @{ title="Special Teams";   mod="etl.build_special_teams";         out="team_special_teams_$SEASON.csv" },
  @{ title="Points Off TO";   mod="etl.build_points_off_turnovers";  out="team_points_off_turnovers_$SEASON.csv" }
)
foreach ($m in $metrics) {
  Invoke-Step ("Build " + $m.title) @(
    "-m",$m.mod,
    "--season",$SEASON,
    "--in_pbp","$InDir/pbp_clean_$SEASON.parquet",
    "--out","$InDir/$($m.out)"
  )
}

# --- 4) Season Summary ---
Invoke-Step "Build Season Summary" @(
  "-m","etl.build_season_summary",
  "--season",$SEASON,
  "--in_dir",$InDir,
  "--out_csv","$InDir/season_summary_$SEASON.csv"
)

# --- 5) Schedule & Matchups (for Week+1) ---
$schedCsv = "$InDir/schedule_w$NextW" + "_$SEASON.csv"
Invoke-Step "Build Schedule (Week $NextW)" @(
  "-m","etl.build_schedule",
  "--season",$SEASON,
  "--week",$NextW,
  "--out",$schedCsv
)
if (-not (Test-Path $schedCsv)) { throw "Schedule not generated: $schedCsv" }

# Regex for tables inside matchup export (kept as a single string)
$tableRegex = "^(Offense vs Opponent Defense|Defense vs Opponent Offense|Team vs Team)"

$rows = Import-Csv $schedCsv
foreach ($r in $rows) {
  $homeTeam = $r.home
  $awayTeam = $r.away
  if ([string]::IsNullOrWhiteSpace($homeTeam) -or [string]::IsNullOrWhiteSpace($awayTeam)) { continue }

  Write-Host ("-> Matchup {0} vs {1} (Week {2})" -f $homeTeam,$awayTeam,$NextW) -ForegroundColor Magenta

  $exportBase = "$homeTeam" + "_" + "$awayTeam" + "_w$NextW" + "_$SEASON"
  $pyArgs = @(
    "-m","etl.build_matchup",
    "--season",$SEASON,
    "--week",$NextW,
    "--home",$homeTeam,
    "--away",$awayTeam,
    "--in_dir",$InDir,
    "--normalize","per_game",
    "--sort","zscore",
    "--min_abs","0.15",
    "--table_regex",$tableRegex,
    "--table_top","200",
    "--table_sort","zscore",
    "--export_table",(Join-Path $MatchDir ($exportBase + "_table.csv")),
    "--export_md",(Join-Path $MatchDir ($exportBase + "_scorecard.md")),
    "--export_html",(Join-Path $MatchDir ($exportBase + "_scorecard.html"))
  )

  & $PY $pyArgs
  if ($LASTEXITCODE -ne 0) { throw "Matchup failed: $homeTeam vs $awayTeam" }
}

Write-Host ""
Write-Host "================ DONE ================" -ForegroundColor Green
Write-Host ("PBP        : {0}\pbp_clean_{1}.parquet" -f $InDir,$SEASON)
Write-Host ("Drive Team : {0}\drive_efficiency_team_{1}.csv" -f $InDir,$SEASON)
Write-Host ("Summary    : {0}\season_summary_{1}.csv" -f $InDir,$SEASON)
Write-Host ("Matchups   : {0}\*w{1}_{2}*" -f $MatchDir,$NextW,$SEASON)
Write-Host "======================================"
