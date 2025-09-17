import pandas as pd
import click

def _detect_half_column(df: pd.DataFrame) -> pd.Series:
    if "quarter" in df.columns:
        q = df["quarter"]
    elif "qtr" in df.columns:
        q = df["qtr"]
    else:
        raise KeyError("Brak kolumny quarter/qtr w PBP.")
    # H1: 1–2, H2: 3–4; inne (OT/NaN) wyrzucamy
    return q.map(lambda x: 1 if x in (1, 2) else (2 if x in (3, 4) else pd.NA))

def _filter_runs_passes(df: pd.DataFrame) -> pd.DataFrame:
    if "play_type" in df.columns:
        return df[df["play_type"].isin(["run", "pass"])]
    conds = []
    if "rush" in df.columns:
        conds.append(df["rush"] == 1)
    if "pass" in df.columns:
        conds.append(df["pass"] == 1)
    if conds:
        return df[pd.concat(conds, axis=1).any(axis=1)]
    raise KeyError("Nie znaleziono play_type ani flag pass/rush w PBP.")

def _success_series(df: pd.DataFrame) -> pd.Series:
    if "success" in df.columns:
        return df["success"].astype(float)
    # fallback: sukces gdy EPA > 0
    return (df["epa"] > 0).astype(float)

def _aggregate_by_side(df: pd.DataFrame, team_col: str, side_label: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["team"] = tmp[team_col]
    tmp["half"] = _detect_half_column(tmp)
    tmp = tmp.dropna(subset=["half"])
    tmp["success_local"] = _success_series(tmp)
    out = (
        tmp.groupby(["team", "week", "half"])
           .agg(
               plays=("epa", "size"),
               epa_avg=("epa", "mean"),
               success_rate=("success_local", "mean"),
           )
           .reset_index()
    )
    out["side"] = side_label
    return out

@click.command()
@click.option("--season", required=True, type=int, help="Sezon NFL (np. 2024)")
@click.option("--in_pbp", required=True, type=click.Path(exists=True), help="Wejściowy plik PBP (parquet)")
@click.option("--out_weekly", required=True, type=click.Path(), help="Plik wyjściowy – weekly CSV")
@click.option("--out_team", required=True, type=click.Path(), help="Plik wyjściowy – team-level CSV")
def main(season, in_pbp, out_weekly, out_team):
    print(f"📥 Wczytuję PBP: {in_pbp}")
    pbp = pd.read_parquet(in_pbp)

    # sezon
    if "season" not in pbp.columns:
        raise KeyError("Brak kolumny 'season' w PBP.")
    df = pbp[pbp["season"] == season].copy()

    # tylko akcje run/pass
    df = _filter_runs_passes(df)

    # wymagane kolumny (dla OFF i DEF)
    required = ["epa", "posteam", "defteam", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Brak wymaganych kolumn w PBP: {missing}")

    # --- OFF (z perspektywy posteam) ---
    off = _aggregate_by_side(df, "posteam", "off")

    # --- DEF (z perspektywy defteam) — odwracamy znak EPA (EPA przeciwnika to koszt obrony) ---
    defn = df.copy()
    defn["epa"] = -defn["epa"]
    defn = _aggregate_by_side(defn, "defteam", "def")

    grouped = pd.concat([off, defn], ignore_index=True)

    # pivot H1 vs H2
    pivoted = grouped.pivot_table(
        index=["team", "week", "side"],
        columns="half",
        values=["plays", "epa_avg", "success_rate"],
    )
    pivoted.columns = [f"{m}_h{int(h)}" for m, h in pivoted.columns]
    pivoted = pivoted.reset_index()

    # upewnij się, że kolumny istnieją (jeśli ktoś nie miał np. zagrań w H1/H2)
    for col in ["epa_avg_h1", "epa_avg_h2", "success_rate_h1", "success_rate_h2"]:
        if col not in pivoted.columns:
            pivoted[col] = pd.NA

    # różnice (H2 - H1)
    pivoted["adj_epa"] = pivoted["epa_avg_h2"] - pivoted["epa_avg_h1"]
    pivoted["adj_sr"]  = pivoted["success_rate_h2"] - pivoted["success_rate_h1"]

    pivoted.to_csv(out_weekly, index=False)
    print(f"✅ Zapisano weekly: {out_weekly} (rows={len(pivoted)})")

    # agregacja team-level
    team = (
        pivoted.groupby(["team", "side"], dropna=False)
               .agg(
                   epa_h1=("epa_avg_h1", "mean"),
                   epa_h2=("epa_avg_h2", "mean"),
                   sr_h1=("success_rate_h1", "mean"),
                   sr_h2=("success_rate_h2", "mean"),
                   adj_epa=("adj_epa", "mean"),
                   adj_sr=("adj_sr", "mean"),
               )
               .reset_index()
    )
    team.to_csv(out_team, index=False)
    print(f"🏁 Zapisano team:   {out_team} (rows={len(team)})")

if __name__ == "__main__":
    main()
