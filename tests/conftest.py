
import pytest
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

@pytest.fixture(scope="session")
def core12_csv():
    p = DATA_DIR / "team_core12_2024.csv"
    if not p.exists():
        pytest.skip(f"Missing {p}. Run: make build-core12")
    return pd.read_csv(p)

@pytest.fixture(scope="session")
def power_csv():
    p = DATA_DIR / "power_ranking_2024.csv"
    if not p.exists():
        pytest.skip(f"Missing {p}. Run: make build-core12")
    return pd.read_csv(p)

@pytest.fixture(scope="session")
def weekly_csv():
    p = DATA_DIR / "team_core12_weekly_2024.csv"
    if not p.exists():
        pytest.skip(f"Missing {p}. Run: make weekly")
    return pd.read_csv(p)

@pytest.fixture(scope="session")
def game_features_csv():
    p = DATA_DIR / "game_features_2024.csv"
    if not p.exists():
        pytest.skip(f"Missing {p}. Run: make features")
    return pd.read_csv(p)