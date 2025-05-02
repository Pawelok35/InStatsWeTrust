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
