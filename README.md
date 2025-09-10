# 🏈 In Stats We Trust

**InStatsWeTrust** is an NFL analytics engine built on *play-by-play (PBP)* data.  
The project is developed in Python and uses [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) to fetch NFL play-by-play data.  

The goal is to build a modular **ETL pipeline** and a set of **team metrics** that will serve as the foundation for predictive models and visualizations.

---

## 🚀 Features

- 📥 **Ingest**
  - Weekly PBP data (`etl/ingest.py`).
  - Full season PBP data (`etl/ingest_season.py`).

- 🔄 **Transform**
  - `etl/transform_epa.py` – offense/defense summary for a selected week.
  - `etl/transform_epa_season.py` – season-level aggregations and rankings.

- 📊 **Metrics**
  - Offensive and Defensive EPA.
  - Success rate.
  - Explosive plays (≥20 yards).
  - Early down and 3rd down efficiency.
  - **Power Signal** = Offensive EPA – Defensive EPA Allowed.

- 📂 **Data structure**
  - `data/raw/` – raw PBP CSVs (excluded from repo).
  - `data/processed/` – processed CSVs (summaries, matchups, rankings).

---

## 🛠️ Installation

Requires **Python 3.10+**.

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Pawelok35/InStatsWeTrust.git
   cd InStatsWeTrust
2. **Create and activate a virtual environment:**
    py -3.10 -m venv .venv
    .venv\Scripts\Activate.ps1   # Windows PowerShell
    # or
    source .venv/bin/activate    # Linux/Mac
3. **Install dependencies:**
    pip install -r requirements.txt

▶️ Usage
Weekly data
python etl/ingest.py
python etl/transform_epa.py


Full season data
python etl/ingest_season.py
python etl/transform_epa_season.py


Processed results will appear in:

data/processed/


📊 Example Results
Offense summary (Week 18, 2024)
team	plays	avg_epa	success_rate	explosive_rate
CAR	85	0.3302	55.3%	4.7%
DEN	87	0.3149	56.3%	6.9%
ARI	86	0.2470	57.0%	7.0%
Defense summary (Week 18, 2024)
team	plays	avg_epa_allowed	success_rate_allowed
DEN	50	-0.3102	30.0%
BAL	77	-0.2554	33.8%
DET	81	-0.2497	35.8%


📌 Roadmap

 ETL pipeline (ingest + transform).

 Offense/Defense metrics + Power Signal.

 Add final scores and winners to matchups.

 Visualizations (matplotlib).

 CLI parameters (choose season via command line).

 Predictive models (models/).

 Web dashboard


---

