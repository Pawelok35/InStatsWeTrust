<# 
.SYNOPSIS
  One-click aktualizacja Hidden Trends (+ opcjonalnie scorecardy) dla ISTW.

.PARAMETER Season
  Sezon NFL (domyślnie 2025).

.PARAMETER Week
  Numer kolejki (wymagany, np. 7).

.PARAMETER PbpPath
  Ścieżka do pliku PBP .parquet. Jeśli nie podasz, skrypt znajdzie najnowszy dla danego sezonu.

.PARAMETER OutPath
  Ścieżka wyjściowa CSV (domyślnie: data\processed\team_hidden_trends_<Season>.csv).

.PARAMETER RunScorecards
  Jeśli dodasz ten switch, skrypt po zbudowaniu HT odpali .\run_week_matchups.ps1 -SEASON <Season> -WEEK <Week>.

.PARAMETER SkipVenv
  Pomiń próbę użycia .venv\Scripts\python.exe i użyj domyślnego `python` z PATH.

.EXAMPLE
  .\Update-HiddenTrends.ps1 -Season 2025 -Week 7 -RunScorecards
#>

[CmdletBinding()]
param(
  [int]$Season = 2025,
  [Parameter(Mandatory = $true)][int]$Week,
  [string]$PbpPath,
  [string]$OutPath,
  [switch]$RunScorecards,
  [switch]$SkipVenv
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSCommandPath
Set-Location $root

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# 1) Walidacje podstawowe
if (-not (Test-Path "etl\build_hidden_trends.py")) {
  Write-Err "Nie znaleziono etl\build_hidden_trends.py. Uruchom skrypt z katalogu głównego repo."
  exit 1
}

# 2) Ustal Python
$pythonExe = "python"
if (-not $SkipVenv) {
  $venvPython = Join-Path $root ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) { $pythonExe = $venvPython }
}
Write-Info "Używam Pythona: $pythonExe"

# 3) Znajdź PBP jeśli nie podano
if (-not $PbpPath) {
  $pbpCandidate = Get-ChildItem "data\processed\pbp" -Filter "*$Season*.parquet" -ErrorAction SilentlyContinue |
                  Sort-Object LastWriteTime -Desc | Select-Object -First 1
  if (-not $pbpCandidate) {
    Write-Err "Nie znaleziono pliku PBP dla sezonu $Season w data\processed\pbp. Podaj -PbpPath."
    exit 1
  }
  $PbpPath = $pbpCandidate.FullName
}

if (-not (Test-Path $PbpPath)) {
  Write-Err "Podany plik PBP nie istnieje: $PbpPath"
  exit 1
}
Write-Info "PBP: $PbpPath"

# 4) Ustal OutPath
if (-not $OutPath) {
  $OutPath = "data\processed\team_hidden_trends_{0}.csv" -f $Season
}
$OutDir = Split-Path $OutPath -Parent
if ($OutDir -and -not (Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

# 5) Log file (opcjonalny)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logPath = Join-Path $logDir ("Update-HiddenTrends_{0}_w{1}_{2}.log" -f $Season, $Week, $stamp)
Write-Info "Log: $logPath"
Start-Transcript -Path $logPath -Append | Out-Null

try {
  # 6) Run build_hidden_trends.py
  Write-Info "Buduję Hidden Trends (Season=$Season, Week=$Week)..."
  & $pythonExe "etl\build_hidden_trends.py" `
      --season $Season `
      --in_pbp $PbpPath `
      --out $OutPath `
      --week $Week

  if ($LASTEXITCODE -ne 0) {
    throw "build_hidden_trends.py zwrócił kod $LASTEXITCODE"
  }

  if (-not (Test-Path $OutPath)) {
    throw "Nie znaleziono pliku wyjściowego: $OutPath"
  }
  Write-Ok "Zapisano: $OutPath"

  # 7) (opcjonalnie) Scorecardy
  if ($RunScorecards) {
    if (-not (Test-Path ".\run_week_matchups.ps1")) {
      Write-Warn "Brak .\run_week_matchups.ps1 – pomijam scorecardy."
    } else {
      Write-Info "Odpalam scorecardy (Week=$Week)..."
      & ".\run_week_matchups.ps1" -SEASON $Season -WEEK $Week
      if ($LASTEXITCODE -ne 0) {
        Write-Warn "run_week_matchups.ps1 zwrócił kod $LASTEXITCODE"
      } else {
        Write-Ok "Scorecardy wygenerowane."
      }
    }
  }

  # 8) Szybki podgląd CSV (pierwsze 5 wierszy)
  try {
    Write-Info "Podgląd CSV (pierwsze 5 wierszy):"
    Import-Csv $OutPath | Select-Object -First 5 | Format-Table | Out-Host
  } catch {
    Write-Warn "Nie udało się wczytać CSV do podglądu: $($_.Exception.Message)"
  }

  Write-Ok "Hidden Trends gotowe."
}
catch {
  Write-Err $_.Exception.Message
  exit 1
}
finally {
  Stop-Transcript | Out-Null
}
