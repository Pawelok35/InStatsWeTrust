
# InStatsWeTrust – Premier League Match Analyzer

This project provides tools to analyze English Premier League football matches using statistical data such as expected goals (xG), results, team form, and a calculated Power Rating system.

---

## 🧠 Key Features

- Compare home and away teams based on last 5 matches
- Evaluate **Power Rating** (form based on xG, points and dominance)
- Highlight match dominance (wins by 3+ goals)
- Analyze xG production and xG allowed
- Measure average opponent strength (position and xG)
- Full round analysis with color-coded differences in form
- Update local dataset with new match results

---

## 📈 Power Rating Scale

| Power Rating       | Interpretation              | Description                                                                      |
|--------------------|-----------------------------|----------------------------------------------------------------------------------|
| **≥ +2.5**         | **Elite**                   | Exceptional form, frequent dominant wins, peak performance.                     |
| **+1.5 to +2.49**  | **Very Strong Form**        | Clearly better than average, consistently wins, dominates weaker teams.         |
| **+0.5 to +1.49**  | **Good Form**               | Above average performance, stable and solid.                                    |
| **-0.49 to +0.49** | **Average Form**            | Balanced team, can win or lose, equal level.                                    |
| **-0.5 to -1.49**  | **Weak Form**               | Below average, frequent point losses, little control over matches.              |
| **-1.5 to -2.49**  | **Very Weak Form**          | Frequent defeats, poor attack/defense, rarely dominates.                        |
| **≤ -2.5**         | **Crisis**                  | Terrible form, heavy and frequent losses, lacks competitiveness.                |

---

## 📁 Project Structure

```
InStatsWeTrust/
│
├── analyze_match.py                                  # Interactive CLI tool for match analysis
├── data/
│   ├── premier_league_match_data_detailed.xlsx       # Processed match data
│   ├── premier_league_raw_match_data.xlsx            # Raw data (xG, result, date)
│   ├── premier_league_team_tables_home_away.xlsx     # Standings by round (Home/Away)
│   └── update_premier_league_match_data_detailed.py  # Script to update enhanced dataset
└── README.md
```

---

## ⚙️ How to Use

1. Ensure all `.xlsx` files are placed inside the `data/` folder.
2. To analyze a specific match:
   ```bash
   python analyze_match.py
   ```
   You will be prompted to choose home/away teams and round number. Output includes form, stats, power rating, and interpretation.

3. To analyze an entire round (all matches in a matchweek):
   ```bash
   python analyze_match.py
   ```
   Then type the round number when prompted (e.g., "Analyze full round? 35").

4. To update the dataset with new matches:
   ```bash
   python update_premier_league_match_data_detailed.py
   ```

---

## 📌 Example Output

```
FORMA GOSPODARZA:
Chelsea (6th in home table)
Avg. Points: 2.6
Power Rating: +1.3
Last 5 matches:
- Ipswich Town (19) | 2–2 | Draw
- Tottenham (14)    | 1–0 | Win
...

FORMA GOŚCIA:
Tottenham (17th away)
Avg. Points: 1.2
Power Rating: -0.8
...

========== SUMMARY ==========
🏠 Home advantage: +1.4 points
⚖️ Power Rating difference: +2.1
📉 Opponent strength: 17.2 (home) vs 11.2 (away)
```

---

## 🛠 Dependencies

- Python 3.8+
- `pandas`
- `openpyxl`

Install with:

```bash
pip install pandas openpyxl
```

---

## 📬 Contact

Maintained by [Pawelok35](https://github.com/Pawelok35) – feel free to fork and contribute!
