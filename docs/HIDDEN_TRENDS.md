Hidden Trends (micro-edges)

Overview
- Purpose: capture subtle, situational advantages not covered by standard summaries.
- Source: derived from play-by-play (PBP) – neutral situations, drive sequences, punts.
- Output files:
  - `data/processed/team_hidden_trends_{season}.csv`
  - Optional per-team updates under `data/processed/teams/{TEAM}.csv` (if week provided)

Metrics
- Q4 Game Rhythm (`game_rhythm_q4`): plays per minute in 4Q when score within ±7. Higher = better closing tempo.
- Neutral Entropy (`play_call_entropy_neutral`): unpredictability of run/pass in neutral context. Higher = harder to defend.
- Neutral Pass Rate (`neutral_pass_rate`): share of passes in neutral context. Higher = more aggressive through the air.
- Neutral Plays (`neutral_plays`): count of neutral-context snaps observed.
- Sustained Drives (`drive_momentum_3plus`): share of drives with ≥3 consecutive positive plays.
- Drives With ≥3 (`drives_with_3plus`): count of such drives.
- Drives Total (`drives_total`): total drives.
- Field Flip Efficiency (`field_flip_eff`): average improvement in opponent start field position after punts.
- Punts Tracked (`punts_tracked`): number of punts used in field flip estimate.

Interpretation
- Higher is better for: Q4 Game Rhythm, Neutral Entropy, Neutral Pass Rate (mild plus), Sustained Drives, Field Flip Efficiency.
- Use deltas (HOME – AWAY) to identify micro-edges for the matchup.

Pipeline
1) ETL Hidden Trends
   - Reads: `data/processed/pbp_clean_{season}.parquet`
   - Writes: `data/processed/team_hidden_trends_{season}.csv`
   - Normalizes team codes; removes aliases like `LA` or `NAN`.
   - Optionally updates per-team histories `data/processed/teams/{TEAM}.csv` when `--week` supplied.

2) Analysis Integration (JSON)
   - `analysis/build_week_analysis.py` augments week JSON with:
     - `hidden_trends`: `{home: {...}, away: {...}}`
     - `hidden_trends_edges`: `{metric: delta_home_minus_away}`
     - `hidden_trends_meta`: `labels` and `tooltips`
   - Appends top-3 hidden edges to the existing `edges` list for TL;DR.

3) GUI (Streamlit)
   - New tab: `8) Hidden Trends (micro-edges)`
   - Renders HOME vs AWAY, Δ (HOME–AWAY), direction icons (↑/↓/≈), and top-3 badges.
   - Tooltips show brief descriptions of the metrics.

Run Commands (PowerShell)
```
& .\.venv\Scripts\Activate.ps1
Set-Location "C:\Users\Daniel\OneDrive\Desktop\InStatsWeTrust"

# 1) ETL Hidden Trends
python etl/build_hidden_trends.py `
  --season 2025 `
  --in_pbp data/processed/pbp_clean_2025.parquet `
  --out data/processed/team_hidden_trends_2025.csv

# 2) Regeneracja analizy tygodniowej
python analysis/build_week_analysis.py --season 2025 --week 6

# 3) Uruchom GUI
streamlit run app/app.py
```

Sanity Checks
- `team_hidden_trends_{season}.csv` includes exactly 32 teams; no `LA`/`NAN` aliases.
- Week JSON contains `hidden_trends`, `hidden_trends_edges`, and `hidden_trends_meta`.
- App shows the “Hidden Trends” tab with correct deltas and icons.

