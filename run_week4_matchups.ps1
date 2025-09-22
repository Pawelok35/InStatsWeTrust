param(
  [int]$SEASON = 2025,
  [int]$WEEK   = 4,
  [string]$IN_DIR = "data/processed"
)

$games = @(
  @{ home = "ARI"; away = "SEA" },  # Arizona Cardinals vs Seattle Seahawks
  @{ home = "PIT"; away = "MIN" },  # Pittsburgh Steelers vs Minnesota Vikings
  @{ home = "ATL"; away = "WAS" },  # Atlanta Falcons vs Washington Commanders
  @{ home = "BUF"; away = "NO"  },  # Buffalo Bills vs New Orleans Saints
  @{ home = "DET"; away = "CLE" },  # Detroit Lions vs Cleveland Browns
  @{ home = "HOU"; away = "TEN" },  # Houston Texans vs Tennessee Titans
  @{ home = "NE";  away = "CAR" },  # New England Patriots vs Carolina Panthers
  @{ home = "NYG"; away = "LAC" },  # New York Giants vs Los Angeles Chargers
  @{ home = "TB";  away = "PHI" },  # Tampa Bay Buccaneers vs Philadelphia Eagles
  @{ home = "LAR"; away = "IND" },  # Los Angeles Rams vs Indianapolis Colts
  @{ home = "SF";  away = "JAX" },  # San Francisco 49ers vs Jacksonville Jaguars
  @{ home = "KC";  away = "BAL" },  # Kansas City Chiefs vs Baltimore Ravens
  @{ home = "LV";  away = "CHI" }   # Las Vegas Raiders vs Chicago Bears
)


$OUT_DIR = Join-Path $IN_DIR "matchups"
if (-not (Test-Path $OUT_DIR)) { New-Item -ItemType Directory -Path $OUT_DIR | Out-Null }

foreach ($g in $games) {
  # UWAGA: NIE używamy $home – to koliduje z wbudowanym $HOME!
  $HomeTeam = $g.home
  $AwayTeam = $g.away

  Write-Host "`n=== Building $HomeTeam vs $AwayTeam (Week $WEEK) ===" -ForegroundColor Cyan

  python etl/build_matchup.py `
    --season $SEASON `
    --week   $WEEK `
    --home   $HomeTeam `
    --away   $AwayTeam `
    --in_dir $IN_DIR `
    --normalize per_game `
    --sort zscore `
    --min_abs 0.25 `
    --table_regex "(Offense vs Opponent Defense|Defense vs Opponent Offense|Team vs Team · (Red Zone|Expl|Takeaways|Offense · 3rd Down|Defense · 3rd Down|Defense · Plays|Offense · Plays))" `
    --table_top 150 `
    --print_table `
    --export_table (Join-Path $OUT_DIR "${HomeTeam}_${AwayTeam}_w${WEEK}_${SEASON}_table.csv") `
    --export_md    (Join-Path $OUT_DIR "${HomeTeam}_${AwayTeam}_w${WEEK}_${SEASON}_scorecard.md") `
    --export_html  (Join-Path $OUT_DIR "${HomeTeam}_${AwayTeam}_w${WEEK}_${SEASON}_scorecard.html")
}
