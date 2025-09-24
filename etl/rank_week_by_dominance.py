#!/usr/bin/env python3
import re
import sys
import csv
from pathlib import Path
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING, Any

# --- Typy tylko dla analizatora (Pylance/mypy), bez twardej zależności w runtime
if TYPE_CHECKING:
    import pandas as pd
    DataFrame = pd.DataFrame
else:
    DataFrame = object  # placeholder w runtime

# ===== WAGI KATEGORII (dopasowanie po fragmencie nazwy metryki w kolumnie "Metric") =====
WEIGHTS: List[Tuple[str, float]] = [
    (r"PPD", 1.0),
    (r"Yards Per Drive|Yards per Drive|yds_per_drive", 0.8),
    (r"Red Zone TD|rz_td_rate|Red Zone", 0.9),
    (r"3rd Down|Third Down|conv_rate|third_down", 0.9),
    (r"Explosive|explosive|chunk20", 0.7),
    (r"Hidden Yards|hidden|field pos|avg start yardline|avg_start_yardline_100", 0.6),
    (r"Second[- ]?Half|adj_epa", 0.5),
    (r"Points off Turnovers|avg_pts_per_takeaway|pot_", 0.5),
    (r"Penalty|pen_per_100|penalty_yds", 0.4),
    (r"Special Teams|st_score|ko_tb_rate|fg_rate|xp_rate", 0.3),
]

# Wiersz tabeli scorecard .md: | Metric | Edge | Dir |
MD_ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*([-\d.]+)\s*\|\s*(HOME ↑|AWAY ↑)\s*\|$")
HEADER_ROW = re.compile(r"^\|\s*Metric\s*\|\s*Edge\s*\|\s*Dir\s*\|$", re.I)

def weight_for_metric(metric: str) -> float:
    for pat, w in WEIGHTS:
        if re.search(pat, metric, flags=re.I):
            return w
    return 0.2  # niska domyślna waga, jeśli nie rozpoznano metryki

def parse_match_name_from_first_line(lines: List[str]) -> Tuple[str, str, str]:
    """
    Zwraca (teamA, teamB, match_label) jeżeli w pierwszej niepustej linii jest ' vs ',
    w przeciwnym razie zwraca ('','','').
    """
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if " vs " in s:
            a, b = s.split(" vs ", 1)
            return a.strip(), b.strip(), s.strip()
        if HEADER_ROW.match(s) or s.startswith("|---"):
            break
        break
    return "", "", ""

def match_label_from_filename(path: Path) -> str:
    """
    Ekstrahuje przyjazną nazwę meczu z nazwy pliku, np.:
    ARI_SEA_w4_2025_scorecard.md -> ARI vs SEA (W4 2025)
    """
    stem = path.name
    if stem.lower().endswith("_scorecard.md"):
        stem = stem[:-len("_scorecard.md")]
    elif stem.lower().endswith(".md"):
        stem = stem[:-3]
    m = re.match(r"^([A-Z]{2,4})_([A-Z]{2,4})_w(\d+)_([12]\d{3})$", stem)
    if m:
        home, away, w, y = m.groups()
        return f"{home} vs {away} (W{w} {y})"
    parts = stem.split("_")
    if len(parts) >= 2 and all(re.fullmatch(r"[A-Za-z0-9]{2,5}", p) for p in parts[:2]):
        return f"{parts[0]} vs {parts[1]}"
    return stem

def try_import_pandas() -> Optional[Any]:
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None

def load_table_csv_for_match(pd_mod: Optional[Any], md_path: Path) -> Optional["DataFrame"]:
    """Próbuje znaleźć obok plik *_table.csv i zwrócić DataFrame z kolumnami [metric,home,away,edge,dir]."""
    if pd_mod is None:
        return None
    candidate = md_path.with_name(md_path.name.replace("_scorecard.md", "_table.csv"))
    if not candidate.exists():
        return None
    try:
        df = pd_mod.read_csv(candidate)
        lower = {c.lower(): c for c in df.columns}
        need = ["metric", "home", "away", "edge", "dir"]
        if not all(col in lower for col in need):
            return None
        df = df.rename(columns={lower[c]: c for c in need})
        return df  # type: ignore[return-value]
    except Exception:
        return None

def round1(x: Optional[float]) -> Optional[float]:
    try:
        return round(float(x), 1)
    except Exception:
        return None

def compute_scores_and_contribs(
    md_text: str,
    fallback_label: str,
    table_df: Optional["DataFrame"]
) -> Dict[str, Any]:
    lines = [ln.rstrip("\n") for ln in md_text.splitlines()]
    teamA, teamB, label_from_text = parse_match_name_from_first_line(lines)
    match_label = label_from_text if label_from_text else fallback_label

    scoreA = 0.0
    scoreB = 0.0
    contribsA: List[Dict[str, Any]] = []
    contribsB: List[Dict[str, Any]] = []

    for ln in lines:
        s = ln.strip()
        if not s or HEADER_ROW.match(s) or s.startswith("|---"):
            continue
        m = MD_ROW.match(s)
        if not m:
            continue
        metric, edge_str, dir_tag = m.groups()
        try:
            edge = float(edge_str)
        except ValueError:
            continue
        w = weight_for_metric(metric)
        val = abs(edge) * w

        home_val = away_val = None
        if table_df is not None:
            try:
                row = table_df.loc[table_df["metric"] == metric].head(1)  # type: ignore[index]
                if getattr(row, "empty", True) is False:
                    home_val = round1(row["home"].iloc[0])   # type: ignore[index]
                    away_val = round1(row["away"].iloc[0])   # type: ignore[index]
            except Exception:
                pass

        record = {
            "metric": metric,
            "edge": edge,
            "weight": w,
            "score": val,
            "dir": dir_tag,
            "home": home_val,
            "away": away_val,
        }

        if "HOME" in dir_tag:
            scoreA += val
            contribsA.append(record)
        else:
            scoreB += val
            contribsB.append(record)

    winner_is_A = scoreA >= scoreB
    if winner_is_A:
        winner = teamA if teamA else "HOME"
        loser  = teamB if teamB else "AWAY"
        top_contribs = sorted(contribsA, key=lambda r: r["score"], reverse=True)[:3]
        win_score, lose_score = scoreA, scoreB
    else:
        winner = teamB if teamB else "AWAY"
        loser  = teamA if teamA else "HOME"
        top_contribs = sorted(contribsB, key=lambda r: r["score"], reverse=True)[:3]
        win_score, lose_score = scoreB, scoreA

    dominance = win_score - lose_score

    return {
        "match": match_label,
        "winner": winner or "",
        "loser": loser or "",
        "winner_score": round(win_score, 3),
        "loser_score": round(lose_score, 3),
        "dominance_index": round(dominance, 3),
        "top3": top_contribs,
    }

def main() -> None:
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/matchups")
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else in_dir / "week_dominance_ranking.csv"

    pd_mod = try_import_pandas()

    rows: List[Dict[str, Any]] = []
    details: List[Tuple[str, str, float, List[Dict[str, Any]]]] = []

    for p in sorted(in_dir.glob("*_scorecard.md")):
        md = p.read_text(encoding="utf-8")
        table_df = load_table_csv_for_match(pd_mod, p)
        res = compute_scores_and_contribs(md, fallback_label=match_label_from_filename(p), table_df=table_df)
        rows.append({k: res[k] for k in ["match","winner","loser","winner_score","loser_score","dominance_index"]})
        details.append((res["match"], res["winner"], float(res["dominance_index"]), res["top3"]))

    rows.sort(key=lambda r: r["dominance_index"], reverse=True)
    details.sort(key=lambda r: r[2], reverse=True)

    # Zapis CSV (ranking)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["match","winner","loser","winner_score","loser_score","dominance_index"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✅ Zapisano ranking: {out_csv}")

    # --- TOP 3 w konsoli ---
    top_n = min(3, len(details))
    if top_n == 0:
        return
    print("\n🏆 TOP 3 typy kolejki:")
    for i in range(top_n):
        match, winner, dom_idx, top3 = details[i]
        print(f"{i+1}. {match} → {winner or 'LEPSZA STRONA'} (Dominance Index {dom_idx})")
        for rec in top3:
            metric = rec["metric"]
            edge = rec["edge"]
            dir_tag = rec["dir"]
            tail = f"edge {edge:+.3f} ({dir_tag})"
            if rec.get("home") is not None and rec.get("away") is not None:
                tail += f" · home {rec['home']:.1f} vs away {rec['away']:.1f}"
            print(f"   - {metric}: {tail}")
    print("")

if __name__ == "__main__":
    main()
