param(
  [int]$SEASON = 2025,
  [int]$WEEK   = 6,
  [string]$IN_DIR = "data/processed"
)

# ⬇️ Uzupełnij listę meczów (home = gospodarz, away = gość)
$games = @(
  @{ home = "CIN"; away = "PIT" }   # 02:15
  @{ home = "JAX"; away = "LAR" }   # 15:30
  @{ home = "CHI"; away = "NO"  }   # 19:00
  @{ home = "CLE"; away = "MIA" }   # 19:00
  @{ home = "KC";  away = "LV"  }   # 19:00
  @{ home = "MIN"; away = "PHI" }   # 19:00
  @{ home = "NYJ"; away = "CAR" }   # 19:00
  @{ home = "TEN"; away = "NE"  }   # 19:00
  @{ home = "DEN"; away = "NYG" }   # 22:05
  @{ home = "LAC"; away = "IND" }   # 22:05
  @{ home = "ARI"; away = "GB"  }   # 22:25
  @{ home = "DAL"; away = "WAS" }   # 22:25
  @{ home = "SF";  away = "ATL" }   # 02:20
  @{ home = "DET"; away = "TB"  }   # 01:00
  @{ home = "SEA"; away = "HOU" }   # 04:00
)



$OUT_DIR = Join-Path $IN_DIR "matchups"
if (-not (Test-Path $OUT_DIR)) { New-Item -ItemType Directory -Path $OUT_DIR | Out-Null }

# regex filtrujący kolumny do tabeli (zostaw pusty, żeby brać wszystkie)
$tableRegex = "^(off_|def_|net_|third_down|fourth_down|red_zone|explosive|ppd|success_rate)"

# wybór Pythona (preferuj venv)
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# zadbaj o importy etl/
$env:PYTHONPATH = (Get-Location).Path

foreach ($g in $games) {
  $HomeTeam = $g.home
  $AwayTeam = $g.away

  Write-Host "== Matchup $HomeTeam vs $AwayTeam (W$WEEK $SEASON) ==" -ForegroundColor Cyan

  & $py "etl\build_matchup.py" `
    --season $SEASON `
    --week   $WEEK `
    --home   $HomeTeam `
    --away   $AwayTeam `
    --in_dir $IN_DIR `
    --normalize per_game `
    --sort zscore `
    --min_abs 0.15 `
    --table_regex $tableRegex `
    --table_top 200 `
    --table_sort zscore `
    --print_table `
    --export_table (Join-Path $OUT_DIR "${HomeTeam}_${AwayTeam}_w${WEEK}_${SEASON}_table.csv") `
    --export_md    (Join-Path $OUT_DIR "${HomeTeam}_${AwayTeam}_w${WEEK}_${SEASON}_scorecard.md") `
    --export_html  (Join-Path $OUT_DIR "${HomeTeam}_${AwayTeam}_w${WEEK}_${SEASON}_scorecard.html")

  if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Błąd dla $HomeTeam-$AwayTeam (kod $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
  }
}

Write-Host "`n✅ Gotowe. Pliki w: $OUT_DIR" -ForegroundColor Green
