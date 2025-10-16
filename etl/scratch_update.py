# etl/_scratch_update_w4.py
import pandas as pd
from pathlib import Path

OUT = Path("data/processed/pbp_clean_2025.parquet")

# 1) Wczytaj istniejący PBP (jeśli jest)
if OUT.exists():
    base = pd.read_parquet(OUT)
else:
    base = pd.DataFrame()

# 2) Dociągnij Week 1–4 2025 (własny loader / nfl_data_py)
try:
    import nfl_data_py as nfl
    new = nfl.import_pbp_data([2025], downcast=True)
    new = new[new["week"].between(1,6)]
except Exception as e:
    raise SystemExit(f"Brak nfl_data_py lub błąd pobierania: {e}")

# 3) Połącz + deduplikacja
all_df = pd.concat([base, new], ignore_index=True) if len(base) else new.copy()
all_df = all_df.drop_duplicates(subset=["game_id","play_id"]).reset_index(drop=True)

# 4) Zapis
OUT.parent.mkdir(parents=True, exist_ok=True)
all_df.to_parquet(OUT, index=False)
print(f"Zapisano: {OUT}  | rows={len(all_df)}  | weeks={sorted(all_df['week'].unique())}")
