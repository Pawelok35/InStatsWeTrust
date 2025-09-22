import pandas as pd
from pathlib import Path

SEASON = 2025
P = Path("data/processed")

def safe_div(a, b):
    return a / b.replace(0, pd.NA)

def patch_redzone():
    for name in [f"redzone_team_{SEASON}.csv", f"redzone_weekly_{SEASON}.csv"]:
        fp = P / name
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        changed = False
        if "rz_td_rate" not in df.columns and {"td_drives","rz_drives"}.issubset(df.columns):
            df["rz_td_rate"] = df["td_drives"].astype(float) / df["rz_drives"].replace(0, pd.NA)
            changed = True
        if changed:
            df.to_csv(fp, index=False)
            print(f"[OK] Patched {name}: rz_td_rate")

def patch_third_down():
    for name in [f"third_down_team_{SEASON}.csv", f"third_down_weekly_{SEASON}.csv"]:
        fp = P / name
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        changed = False
        if "conv_rate" not in df.columns and {"off_3d_conv","off_3d_att"}.issubset(df.columns):
            df["conv_rate"] = df["off_3d_conv"].astype(float) / df["off_3d_att"].replace(0, pd.NA)
            changed = True
        # jeżeli masz defensywne epa per play na 3rd down w innych kolumnach, zmapuj je tutaj:
        # np. jeśli kolumna nazywa się 'def_3d_epa_pp', zrób alias:
        cand = [c for c in df.columns if c.lower() in ("def_3d_epa_per_play","def_3d_epa_pp","def_third_down_epa_per_play")]
        if "def_3d_epa_per_play" not in df.columns and cand:
            df["def_3d_epa_per_play"] = df[cand[0]]
            changed = True
        if changed:
            df.to_csv(fp, index=False)
            print(f"[OK] Patched {name}: conv_rate/def_3d_epa_per_play")

def patch_drive_efficiency():
    for name in [f"drive_efficiency_team_{SEASON}.csv", f"drive_efficiency_weekly_{SEASON}.csv"]:
        fp = P / name
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        changed = False
        # alias offense-specific start yardline jeśli masz tylko jedną wersję
        if "avg_start_yardline_100_off" not in df.columns:
            # jeśli istnieje 'start_yardline_100_avg', użyj jej jako proxy dla OFF
            if "start_yardline_100_avg" in df.columns:
                df["avg_start_yardline_100_off"] = df["start_yardline_100_avg"]
                changed = True
            elif "start_own_yardline_avg" in df.columns:
                # przelicz na skalę 100y jeżeli masz tylko 'start_own_yardline_avg'
                # (tu zostawiamy proxy  = start_own_yardline_avg jako przybliżenie)
                df["avg_start_yardline_100_off"] = df["start_own_yardline_avg"]
                changed = True

        # field_pos_advantage: jeśli masz 'start_yardline_100_avg' i np. średnią dla przeciwników,
        # to policz różnicę. Jeśli nie masz defensywnego odpowiednika – zostaw pustą.
        if "field_pos_advantage" not in df.columns:
            # heurystyka: jeśli są kolumny off/def dla start yardline, policz różnicę
            off_cand = None
            def_cand = None
            for c in df.columns:
                cl = c.lower()
                if "start" in cl and "100" in cl and "off" in cl: off_cand = c
                if "start" in cl and "100" in cl and ("def" in cl or "opp" in cl): def_cand = c
            if off_cand and def_cand:
                df["field_pos_advantage"] = df[off_cand].astype(float) - df[def_cand].astype(float)
                changed = True

        # turnover_drives: jeśli mamy drives i turnover_rate
        if "turnover_drives" not in df.columns and {"drives","turnover_rate"}.issubset(df.columns):
            df["turnover_drives"] = df["drives"].astype(float) * df["turnover_rate"].astype(float)
            changed = True

        if changed:
            df.to_csv(fp, index=False)
            print(f"[OK] Patched {name}: avg_start_yardline_100_off/field_pos_advantage/turnover_drives")

def main():
    patch_redzone()
    patch_third_down()
    patch_drive_efficiency()

if __name__ == "__main__":
    main()
