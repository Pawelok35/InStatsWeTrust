# InStatsWeTrust – Premier League Match Analyzer

This project provides tools to analyze English Premier League football matches using statistical data such as expected goals (xG), results, and recent team form.

## 🧠 Key Features

- Compare home and away teams based on last 5 matches
- Analyze xG production and xG allowed
- Evaluate strength of opponents and form (Win/Draw/Loss)
- Highlight which team has a statistical edge
- Update local dataset with fresh match results and xG data

## 📁 Project Structure

```
InStatsWeTrust/
│
├── analyze_match.py                       # Interactive CLI tool for analyzing a match
├── update_premier_league_match_data_detailed.py  # Script to update detailed match dataset
├── data/
│   ├── premier_league_match_data_detailed.xlsx   # Main dataset (team, opponent, result, xG, etc.)
│   ├── premier_league_raw_match_data.xlsx        # Raw match data with xG and results
│   └── premier_league_team_tables_home_away.xlsx # League standings (HomeTable and AwayTable)
└── README.md
```

## ⚙️ How to Use

1. Make sure all Excel files are placed inside the `data/` folder.
2. Run the match analyzer:
   ```bash
   python analyze_match.py
   ```
   - You’ll be asked to select home team, away team, and match round.
   - The tool will display recent form, xG metrics, opponent strength, and predictions.

3. To update the dataset with new results and xG values:
   ```bash
   python update_premier_league_match_data_detailed.py
   ```

   This script merges raw match data (`premier_league_raw_match_data.xlsx`) into the structured match dataset.

## 📝 Data Requirements

The analysis requires these Excel files to be placed in the `data/` folder:

- `premier_league_match_data_detailed.xlsx`: Enhanced match data (team, opponent, type, round, score, xG).
- `premier_league_raw_match_data.xlsx`: Raw source with `Home`, `Away`, `xG`, `xG.1`, and `Score`.
- `premier_league_team_tables_home_away.xlsx`: Contains standings (HomeTable and AwayTable) with team performance stats per round.

## 📌 Example Output

```
FORMA GOSPODARZA:
Średnia punktów: 1.6
Średnie xG: 1.45
Ostatnie mecze gospodarza:
- Chelsea (4)                 Wynik: 2–1 | Wygrana
- Arsenal (2)                Wynik: 1–1 | Remis
...

FORMA GOŚCIA:
Średnia punktów: 2.0
Średnie xG: 1.80
...

========== PODSUMOWANIE ==========
RÓŻNICA NA KORZYŚĆ GOŚCIA WYNOSI 0.4
...
```

## 🛠 Dependencies

- Python 3.8+
- pandas
- openpyxl (for `.xlsx` support)

Install via pip:

```bash
pip install pandas openpyxl
```

## 📬 Contact

Maintained by [Pawelok35](https://github.com/Pawelok35) – feel free to fork and contribute!