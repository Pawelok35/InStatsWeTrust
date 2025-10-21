#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from analysis.utils_hidden_trends import load_hidden_trends, compute_hidden_trends_edges, HIDDEN_TREND_COLS


def _load_week_json(path: Path) -> Dict[str, Any] | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_week_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _abbr_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    # Import late to avoid Streamlit dependency when running headless
    from app.utils import NFL_ABBR_TO_NAME, NAME_ALIASES
    abbr_to_name = {k: v for k, v in NFL_ABBR_TO_NAME.items()}
    name_to_abbr = {v.upper(): k for k, v in NFL_ABBR_TO_NAME.items()}
    # Add alias variants
    for canon, aliases in NAME_ALIASES.items():
        for a in aliases:
            name_to_abbr[a.upper()] = name_to_abbr.get(canon.upper(), name_to_abbr.get(a.upper(), None)) or \
                                       next((abbr for abbr, nm in abbr_to_name.items() if nm.upper() == canon.upper()), None)
    return abbr_to_name, name_to_abbr


def _resolve_abbr(full_name: str, name_to_abbr: Dict[str, str]) -> str | None:
    return name_to_abbr.get(full_name.upper())


def _labels_and_tooltips() -> Dict[str, Dict[str, str]]:
    labels = {
        "game_rhythm_q4": "Q4 Game Rhythm",
        "play_call_entropy_neutral": "Neutral Entropy",
        "neutral_pass_rate": "Neutral Pass Rate",
        "neutral_plays": "Neutral Plays",
        "drive_momentum_3plus": "Sustained Drives (>=3)",
        "drives_with_3plus": "Drives with >=3",
        "drives_total": "Total Drives",
        "field_flip_eff": "Field Flip Efficiency",
        "punts_tracked": "Punts Tracked",
    }
    tooltips = {
        "game_rhythm_q4": "Tempo/rytm w Q4 – lepsze zamykanie meczów.",
        "play_call_entropy_neutral": "Entropia play-call w neutralnych sytuacjach – trudniejsza do przewidzenia ofensywa.",
        "neutral_pass_rate": "% podań w neutralnych sytuacjach – wyższe = agresywniejsza gra po powietrzu.",
        "field_flip_eff": "Zdolność do odwracania pozycji na boisku po puncie.",
        "drive_momentum_3plus": "Odsetek drivów z ≥3 kolejnymi udanymi akcjami.",
    }
    return {"labels": labels, "tooltips": tooltips}


def _top3_hidden_edges(edges: Dict[str, float | None], home_abbr: str, away_abbr: str) -> List[Dict[str, Any]]:
    # Return Edge objects compatible with existing JSON schema
    # winner team = home if delta>0 else away
    def _friendly_name(k: str) -> str:
        return {
            "game_rhythm_q4": "Hidden: Q4 Game Rhythm",
            "play_call_entropy_neutral": "Hidden: Neutral Entropy",
            "neutral_pass_rate": "Hidden: Neutral Pass Rate",
            "field_flip_eff": "Hidden: Field Flip Efficiency",
            "drive_momentum_3plus": "Hidden: Sustained Drives",
        }.get(k, f"Hidden: {k}")

    items = [(k, v) for k, v in edges.items() if v is not None]
    items.sort(key=lambda kv: abs(float(kv[1])), reverse=True)
    top = items[:3]
    out: List[Dict[str, Any]] = []
    for k, v in top:
        team = home_abbr if float(v) > 0 else (away_abbr if float(v) < 0 else None)
        out.append({"name": _friendly_name(k), "team": team, "value": float(v)})
    return out


def build(season: int, week: int, base_dir: Path) -> Path:
    analyses_dir = base_dir / "data/processed/analyses"
    out_path = analyses_dir / f"week_{week}_{season}_analysis.json"
    base = _load_week_json(out_path) or {"season": season, "week": week, "generated_at": None, "games": []}

    # Prepare games list: either from JSON, or from PS1
    games = base.get("games", [])
    if not games:
        # fallback: derive from PS1
        from app.utils import load_games_from_ps1
        ps_games = load_games_from_ps1(base_dir / "run_week_matchups.ps1")
        # Convert to full-name entries to be consistent with JSON
        from app.utils import NFL_ABBR_TO_NAME
        games = [{
            "home": NFL_ABBR_TO_NAME.get(g["home_abbr"], g["home_abbr"]),
            "away": NFL_ABBR_TO_NAME.get(g["away_abbr"], g["away_abbr"]),
        } for g in ps_games]
        base["games"] = games

    # Load Hidden Trends df
    ht = load_hidden_trends(season=season, base_dir=base_dir)
    _, name_to_abbr = _abbr_maps()

    meta = _labels_and_tooltips()

    for g in games:
        home_name = g.get("home")
        away_name = g.get("away")
        if not home_name or not away_name:
            continue
        home_abbr = _resolve_abbr(home_name, name_to_abbr)
        away_abbr = _resolve_abbr(away_name, name_to_abbr)
        if not home_abbr or not away_abbr:
            # cannot resolve -> skip hidden trends for this game
            continue

        comp = compute_hidden_trends_edges(ht, home_abbr, away_abbr)
        g["hidden_trends"] = {"home": comp["home"], "away": comp["away"]}
        g["hidden_trends_edges"] = comp["edges"]
        g["hidden_trends_meta"] = meta

        # Augment top edges list with top-3 hidden edges
        existing_edges: List[Dict[str, Any]] = g.get("edges", []) or []
        hidden_top = _top3_hidden_edges(comp["edges"], home_abbr, away_abbr)
        # Merge, preferring existing first
        g["edges"] = existing_edges + hidden_top

    # Stamp generation time if not present
    if not base.get("generated_at"):
        from datetime import datetime, timezone
        base["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    _save_week_json(base, out_path)
    # Sanity: print top-3 deltas for first game
    try:
        first = base.get("games", [])[0]
        if first and first.get("hidden_trends_edges"):
            edges = first["hidden_trends_edges"]
            top = sorted([(k, v) for k, v in edges.items() if v is not None], key=lambda kv: abs(kv[1]), reverse=True)[:3]
            print("[SANITY] Top-3 Hidden Trends deltas (HOME–AWAY) for first game:")
            for k, v in top:
                print(f"  - {k}: {v:+.3f}")
    except Exception:
        pass
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()
    out = build(season=args.season, week=args.week, base_dir=Path("."))
    print(f"[OK] Updated week analysis with Hidden Trends: {out}")


if __name__ == "__main__":
    main()
