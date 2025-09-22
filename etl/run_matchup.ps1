param(
  [Parameter(Mandatory=$true)][int]$Season = 2025,
  [Parameter(Mandatory=$true)][int]$Week,          # używamy danych z tygodni < Week
  [Parameter(Mandatory=$true)][string]$HomeTeam,
  [Parameter(Mandatory=$true)][string]$AwayTeam,

  [string]$InDir = "data/processed",
  [string]$Normalize = "per_game",                  # none | per_game
  [string]$SortMode = "zscore",                     # abs | zscore
  [double]$MinAbs = 0.5,
  [int]$TopEdges = 20,

  # Tabela porównawcza – domyślny filtr na kluczowe metryki
  [string]$TableRegex = "(epa|sr|third_down|rz(_td_rate|_drives)?|ppd|explosive)",
  [int]$TableTop = 80,

  # Dodatkowy podgląd surowych wartości (opcjonalnie)
  [string]$ShowRaw = ""
)

# ── Sanitizacja kodów drużyn ───────────────────────────────────────────────────
$HomeTeam = $HomeTeam.Trim().ToUpper()
$AwayTeam = $AwayTeam.Trim().ToUpper()

# ── Ścieżki wyników ───────────────────────────────────────────────────────────
$matchupDir = Join-Path $InDir "matchups"
if (-not (Test-Path $matchupDir)) { New-Item -ItemType Directory -Path $matchupDir | Out-Null }

$tag = "${HomeTeam}_${AwayTeam}_w${Week}_${Season}"
$outCsv    = Join-Path $matchupDir "$tag.csv"
$outTable  = Join-Path $matchupDir "${tag}_table.csv"
$outMd     = Join-Path $matchupDir "${tag}_scorecard.md"
$outHtml   = Join-Path $matchupDir "${tag}_scorecard.html"

Write-Host "▶ Running matchup for $HomeTeam vs $AwayTeam (Week $Week, $Season)..." -ForegroundColor Cyan
Write-Host "   Output: $matchupDir" -ForegroundColor DarkCyan

# ── Argumenty do Pythona ──────────────────────────────────────────────────────
$argsList = @(
  "etl/build_matchup.py",
  "--season", $Season,
  "--week",   $Week,
  "--home",   $HomeTeam,
  "--away",   $AwayTeam,
  "--in_dir", $InDir,
  "--out",    $outCsv,
  "--normalize", $Normalize,
  "--sort",      $SortMode,
  "--min_abs",   $MinAbs,
  "--top",       $TopEdges,
  "--print_table",
  "--table_regex", $TableRegex,
  "--table_top",   $TableTop,
  "--export_table", $outTable,
  "--export_md",    $outMd,
  "--export_html",  $outHtml
)

if ($ShowRaw -and $ShowRaw.Trim().Length -gt 0) {
  $argsList += @("--show_raw", $ShowRaw)
}

# ── Run ───────────────────────────────────────────────────────────────────────
& python @argsList
$ec = $LASTEXITCODE

if ($ec -eq 0) {
  Write-Host "`n✅ Done. Files saved:" -ForegroundColor Green
  Write-Host " - $outCsv"
  Write-Host " - $outTable"
  Write-Host " - $outMd"
  Write-Host " - $outHtml"
} else {
  Write-Host "`n❌ Python exited with code $ec" -ForegroundColor Red
}
