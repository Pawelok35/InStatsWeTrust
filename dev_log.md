# 📔 Development Journal – InStatsWeTrust

## 📅 2024-04-24
### ✅ Completed:
- Introduced structured match analysis concept.
- Defined input data format (matches, HOME/AWAY tables, dominance, xG).
- Created base functions `get_recent_form()` and `analyze_match()`.

### 💡 Ideas:
- Opponent difficulty based on league position.
- Colored terminal output for clarity.

---

## 📅 2024-04-25
### ✅ Completed:
- Added xG averages for last 5 matches for home and away teams.
- Calculated average opponent position.
- Displayed formatted match results.

### 🐞 Fixed:
- Handled `None` values for missing xG.
- Fixed missing opponent position in `get_recent_form()`.

---

## 📅 2024-04-26
### ✅ Completed:
- Compared average opponent strength for both teams.
- Enhanced formatting with section headers and colors.
- Added interpretation of opponent difficulty.

### 💡 Ideas:
- Add form rating: great/good/average/poor.

---

## 📅 2024-04-27
### ✅ Completed:
- Added average xG of opponents.
- Grouped opponents into tiers: Top 6, 7–10, 11–14, 15–20.
- Calculated win/draw/loss records and average points per tier.

### 🐞 Fixed:
- Fixed `ValueError` when converting NaN to int.
- Improved text alignment and spacing for opponent info.

---

## 📅 2024-04-28
### ✅ Completed:
- Displayed team name and current HOME/AWAY league position.
- Cleaned up output formatting and spacing.
- Moved dev journal entries into `dev_log.md`.

---

## 📅 2024-04-29
### ✅ Completed:
- Organized repository: renamed Python and Excel files.
- Added README and pushed repo to GitHub.
- Excluded `.venv` and committed `.gitignore`.

### 💡 Ideas:
- Add result export to `.txt` or `.csv`.

---

## 📅 2024-04-30
### ✅ Completed:
- Fixed issues with opponent match lists in form section.
- Ensured proper alignment of opponent rows in match logs.
- Improved error handling for `None` and non-dict `result` values.
- Synced team positions with HOME or AWAY tables depending on match context.

### 🐞 Fixed:
- Prevented crash on missing data in opponent tier classification.
- Corrected handling of float NaNs in league positions.

## 📅 2025-05-01
### ✅ Completed:
- Renamed all Polish column headers in Excel files to English equivalents (`TEAM`, `Opponent`, `Type`, `Round`, etc.).
- Updated `update_premier_league_match_data_detailed.py` to match new column names and xG/Score logic.
- Refactored `analyze_match.py` to fully support English naming across all data sources.
- Fixed `KeyError` issues related to outdated column names (`TEAM`, `Position`, etc.).
- Displayed team name and current HOME/AWAY table position in form output.

### 🐞 Fixed:
- Crash caused by mismatched column names (`Pozycja` → `Position`).
- Misalignment between Excel sheet formats and Python expectations.

---

## 📅 2025-05-02

### ✅ Completed:
- Fully refactored `analyze_match.py` to align with the new English column naming convention (`Points`, `xG`, `Score`, `Opponent`, etc.).
- Rewrote the `get_recent_form`, `analyze_match`, and match output sections to support English-based data structures.
- Confirmed that reversed match score logic is applied correctly for away teams.
- Ensured consistency between detailed match data and team tables (`HomeTable` / `AwayTable`).
- Modularized summary statistics: avg points, avg xG, and opponent positions/xG.

### 🐞 Fixed:
- Missing data in match analysis summary (average points, opponent data) caused by outdated Polish keys (`Śr. Punkty (5m)`, `Pozycja`, etc.).
- Corrected logic to retrieve and display xG values for both teams and their opponents across last 5 matches.
- Resolved inconsistencies between match type references (`home`, `away`) and team normalization.

---

## 📅 2025-05-05

### ✅ Completed:
- Fully integrated and synchronized match data across `match_data_detailed.xlsx`, `team_tables_home_away.xlsx`, and `raw_match_data.xlsx`.
- Refactored the `get_recent_form()` function to reliably calculate:
  - average points and xG over the last 5 matches,
  - average opponent positions,
  - average xG of opponents.
- Improved logic for identifying opponent match type (`home`/`away`) to ensure accurate data retrieval.
- Applied green ANSI color highlighting for `"Statystyka Gospodarza"` and `"Statystyka Gościa"` headers in the console output.
- Verified that the entire match analysis works correctly across all 34 matchdays and for all 20 teams.

### 🐞 Fixed:
- Fixed missing data in summaries (opponent xG and position) caused by incomplete filtering or lack of name normalization.
- Replaced `None`/`NaN` outputs with actual numerical values when data is available.
- Reactivated and correctly used `home_stats` and `away_stats` to display table positions.
- Simplified and unified match summary logic and final output structure.

---

## 📅 2025-05-06

### ✅ Completed:
- Implemented and integrated **Power Rating** logic based on average points and number of “Domination” wins over the last 5 matches.
- Created `calculate_power_rating()` function to produce a single score summarizing recent team strength.
- Designed and documented Power Rating interpretation scale with 7 detailed tiers, from “Elite” to “Crisis”.
- Displayed Power Ratings clearly within match summaries, just after average points.
- Refactored output section to make rating comparison and summaries cleaner and easier to interpret.
- Enhanced `analyze_round()` to use Power Rating differences for quick visual comparison of team strength.
- Updated markdown documentation and appended Power Rating explanation to the `README.md`.

### 🐞 Fixed:
- Corrected uninitialized variables (`sr_home`, `sr_away`) in round analysis that caused inconsistencies.
- Moved Power Rating values from a hard-to-read location to a more logical position just under average points.
- Validated that Power Ratings and opponent stats are calculated properly for all teams with complete data.
- Reorganized summary output for better readability and separation of stats, insights, and tier summaries.

---

## 📅 2025-05-07

### ✅ Completed:
- Reworked `analyze_round()` to use **difference in average points** from the last 5 home/away matches instead of Power Rating.
- Adjusted output formatting with **fixed-width columns** to align all match data (matchup, favorite, signal, difference) perfectly.
- Created a new version of `ocena_sygnalu()` to reflect **point-based differences**, including a 5-tier interpretation with emojis.
- Integrated point difference presentation: "ADVANTAGE FOR HOME TEAM IS X" styled and aligned consistently across all matches.
- Ensured signal and advantage columns remain **vertically aligned** even with long labels like "✅ WARTO ZAGRAĆ...".
- Kept Power Rating logic and related functions intact, but decoupled from round-level analysis for clarity and precision.

### 🐞 Fixed:
- Eliminated misalignment in visual output when long signal texts caused the final column to shift.
- Resolved confusion between Power Rating difference and average point difference by introducing explicit separation of logic.
- Replaced incorrect use of Power Rating in signal generation with accurate point-based logic using recent form data.
- Simplified logic for clarity by pulling `Śr. Punkty (5m)` directly from recent form instead of relying on derived scores.

---
