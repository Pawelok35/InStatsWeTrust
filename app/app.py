import streamlit as st
from datetime import datetime
from pathlib import Path
import pandas as pd

# ====== importy z utils ======
from utils import (
    parse_week_analysis, game_label,
    load_games_from_ps1, game_key_from_abbr,
    confidence_badge, detail_md_path, equal_names,
    parse_analysis_sections, TAB_TITLES   # <-- parser wklejki + tytuły zakładek
)
from analysis.utils_hidden_trends import (
    load_hidden_trends, compute_hidden_trends_edges, HIDDEN_TREND_COLS
)

# ============== USTAWIENIA STRONY ==============
st.set_page_config(
    page_title="In Stats We Trust – NFL Insights Dashboard",
    page_icon="🏈",
    layout="wide",
)

# ============== STYL (CSS) ==============
st.markdown("""
<style>
/* przyciski secondary */
div[data-testid="baseButton-secondary"] > button {
  padding: 10px 22px !important;
  font-size: 1.05rem !important;
  border: 2px solid #00ff99 !important;
  color: #00ff99 !important;
  background: transparent !important;
  border-radius: 8px !important;
}
div[data-testid="baseButton-secondary"] > button:hover {
  background: #00ff99 !important;
  color: #000 !important;
}

/* header */
h1, .h1 { color:#00FF88; text-align:center; }
hr { border:1px solid #00FF88; }

/* KPI & cards */
.kpi { border:1px solid #00ff99; border-radius:12px; padding:10px 12px; margin:6px 0; }
.kpi .label { color:#99ffd6; font-size:0.85rem; }
.kpi .value { color:#fff; font-size:1.15rem; font-weight:700; }
.cardgrid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:8px; }
.card { border:1px solid #114; border-radius:10px; padding:10px; background:#0a0f0a; }
.card .name { color:#9deabf; font-size:0.85rem; }
.card .val  { color:#fff; font-size:1.05rem; font-weight:700; }
.tldr { background:#09130f; border:1px solid #0a3; border-radius:10px; padding:10px 14px; }
.tabnote { color:#bbb; font-size:0.9rem; }

/* Oddziel zakładki pionową kreską i nadaj kontrast */
div[data-baseweb="tab-list"] button {
    border-right: 1px solid rgba(0,255,136,0.3);
    padding-right: 12px;
    margin-right: 6px;
    font-weight: 600;
    color: #00FF88;
}
/* Ostatnia zakładka bez prawej kreski */
div[data-baseweb="tab-list"] button:last-child { border-right: none; }
/* Aktywna zakładka wyróżniona */
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: rgba(0,255,136,0.08);
    border-bottom: 2px solid #00FF88;
}
/* Hover efekt */
div[data-baseweb="tab-list"] button:hover { background-color: rgba(0,255,136,0.15); }
</style>
""", unsafe_allow_html=True)

# ============== HEADER ==============
st.markdown("""
<h1>In Stats We Trust</h1>
<p style='text-align: center; color: #cccccc;'>NFL Insights Dashboard</p>
<hr>
""", unsafe_allow_html=True)

# ============== ŚCIEŻKI ==============
DATA_PATH = Path("data/processed/analyses/week_6_2025_analysis.json")
PS1_PATH = Path("run_week_matchups.ps1")  # plik z $games
CSV_2024_PATH = Path("data/processed/season_summary_2024_clean.csv")

SEASON = 2025
WEEK = 6

# ============== STAN WIDOKU ==============
if "analysis_view" not in st.session_state:
    st.session_state.analysis_view = False
    st.session_state.analysis_row = None
    st.session_state.analysis_game = None

def open_analysis(row: dict, game_obj):
    st.session_state.analysis_view = True
    st.session_state.analysis_row = row
    st.session_state.analysis_game = game_obj

def close_analysis():
    st.session_state.analysis_view = False
    st.session_state.analysis_row = None
    st.session_state.analysis_game = None

# ============== CACHE LOADERS ==============
@st.cache_data(show_spinner=False)
def load_week():
    return parse_week_analysis(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_games():
    return load_games_from_ps1(PS1_PATH)

@st.cache_data(show_spinner=False)
def load_csv_2024():
    if not CSV_2024_PATH.exists():
        raise FileNotFoundError(f"2024 CSV not found at: {CSV_2024_PATH}")
    df = pd.read_csv(CSV_2024_PATH)
    # normalizacja nazw: lower + underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_hidden_trends_df():
    base = Path(".")
    try:
        return load_hidden_trends(SEASON, base)
    except Exception as e:
        st.warning(f"Hidden Trends CSV not loaded: {e}")
        return None

# ============== LOGIKA DANYCH (status) ==============
st.subheader("Week Analysis – Data status")
try:
    wa = load_week()
    st.success(f"Loaded: {DATA_PATH}  |  Games in JSON: {len(wa.games)}  |  Generated: {wa.generated_at}")
except Exception as e:
    st.warning(f"Could not load analysis JSON: {e}")
    wa = None

try:
    ps_games = load_games()
    st.success(f"Loaded matchups from PowerShell: {PS1_PATH}  |  Games: {len(ps_games)}")
except Exception as e:
    st.error(str(e))
    st.stop()

def find_analysis_for(home_full: str, away_full: str):
    if not wa:
        return None, "No JSON loaded."
    for g in wa.games:
        if equal_names(g.home, home_full) and equal_names(g.away, away_full):
            return g, None
    return None, "No detailed analysis found for this matchup in week JSON."

# ============== HELPERY (lokalne) ==============
FIELDS_MAP = {
    "PPD_off": "ppd_offense",
    "Yds/Drive": "yards_per_drive_offense",
    "EPA/play (off)": "epa_per_play_offense",
    "3rd Down SR (off)": "third_down_sr_offense",
    "RZ EPA (off)": "redzone_epa_offense",
    "Explosive Plays (off)": "explosive_plays_offense",
    "Explosive Allowed (def)": "explosive_allowed_defense",
    "EPA/play allowed (def)": "epa_per_play_defense",
    "3rd Down SR allowed (def)": "third_down_sr_allowed",
    "RZ EPA allowed (def)": "redzone_epa_allowed",
    "Plays/Drive": "plays_per_drive",
    "Start FP": "start_fp",
    "Hidden Yards/Drive": "hidden_yards_per_drive",
}

def _fmt_pct(v):
    if v is None: return "N/A"
    try:
        if 0 <= float(v) <= 1:
            return f"{float(v)*100:.1f}%"
        return f"{float(v):.1f}%"
    except Exception:
        return str(v)

def _fmt_num(v, nd=2):
    if v is None: return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

def _team_row_2024(df, team_name: str):
    team_col = "team" if "team" in df.columns else ("team_name" if "team_name" in df.columns else None)
    if not team_col:
        raise KeyError("CSV 2024 must contain 'team' or 'team_name' column.")
    r = df[df[team_col].str.lower() == team_name.lower()]
    # proste aliasy LA Rams/Chargers – jeśli masz inne, dopisz
    if r.empty and team_name in {"LA Rams", "Los Angeles Rams"}:
        r = df[df[team_col].str.lower() == "los angeles rams"]
    if r.empty and team_name in {"LA Chargers", "Los Angeles Chargers"}:
        r = df[df[team_col].str.lower() == "los angeles chargers"]
    return r.iloc[0].to_dict() if not r.empty else {}

@st.cache_data(show_spinner=False)
def build_profile_cards_2024_inline(home_team: str, away_team: str):
    df = load_csv_2024()
    rh = _team_row_2024(df, home_team)
    ra = _team_row_2024(df, away_team)

    def as_cards(row_dict: dict):
        cards = []
        for ui_key, csv_col in FIELDS_MAP.items():
            val = row_dict.get(csv_col)
            if "SR" in ui_key:
                disp = _fmt_pct(val)
            elif "PPD" in ui_key:
                disp = _fmt_num(val, nd=2)
            else:
                disp = _fmt_num(val, nd=2)
            cards.append({"name": ui_key, "value": disp})
        return cards
    return as_cards(rh), as_cards(ra)

def summarize_top_edges_inline(game_obj, n=3):
    """Top n edge'ów po bezwzględnej wartości."""
    if not game_obj or not getattr(game_obj, "edges", None):
        return []
    edges = [e for e in game_obj.edges if getattr(e, "value", None) is not None]
    edges.sort(key=lambda x: abs(float(x.value)), reverse=True)
    out = []
    for e in edges[:n]:
        dir_txt = f" → edge **{getattr(e, 'team', '—')}**" if getattr(e, "team", None) else ""
        try:
            val_txt = f"{float(e.value):+,.2f}"
        except Exception:
            val_txt = str(e.value)
        out.append(f"**{e.name}**: {val_txt}{dir_txt}")
    return out

# ============== WIDOK ANALIZY (FULL PAGE) ==============
def render_analysis_view(row: dict, game_obj):
    st.markdown("<br>", unsafe_allow_html=True)
    cols_header = st.columns([0.2, 0.8])
    with cols_header[0]:
        if st.button("← Back to games", key="back_top", type="secondary"):
            close_analysis()
            st.stop()

    st.markdown(f"## Analiza: {row.get('away','AWAY')} @ {row.get('home','HOME')}")
    st.markdown("---")

    # --- Paste-in analiza (opcjonalne nadpisanie zakładek) ---
    st.markdown("##### Wklej analizę (opcjonalnie, auto-rozrzut do 7 zakładek)")
    paste = st.text_area(
        "Wklej tutaj cały tekst analizy (z nagłówkami 1) … 7) albo tymi nazwami):",
        value="",
        height=220,
        help="Parser wykrywa zarówno numerowane nagłówki, jak i aliasy (PL/EN).",
        key="paste_area",
    )
    use_paste = st.button("Analizuj (rozrzuć do zakładek)", key="btn_paste_parse")
    sections_from_paste = parse_analysis_sections(paste) if (use_paste and paste.strip()) else None

    # === TABS (7 zakładek, stałe tytuły) ===
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(TAB_TITLES)

    # ---- 1) Profil drużyn [2024 → 2025] ----
    with t1:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[0], "—")
            st.markdown(md if md.strip() else "—")
        else:
            st.markdown("<span class='tabnote'>Dane 2024 z CSV. 2025: tylko trendy wywnioskowane z #2 (Edges).</span>", unsafe_allow_html=True)
            try:
                cards_home, cards_away = build_profile_cards_2024_inline(row.get('home',''), row.get('away',''))
                st.markdown(f"### {row.get('away','AWAY')} @ {row.get('home','HOME')} — Profil 2024")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{row.get('home','HOME')} (2024)**")
                    st.markdown("<div class='cardgrid'>" + "".join(
                        [f"<div class='card'><div class='name'>{c['name']}</div><div class='val'>{c['value']}</div></div>"
                         for c in cards_home]) + "</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{row.get('away','AWAY')} (2024)**")
                    st.markdown("<div class='cardgrid'>" + "".join(
                        [f"<div class='card'><div class='name'>{c['name']}</div><div class='val'>{c['value']}</div></div>"
                         for c in cards_away]) + "</div>", unsafe_allow_html=True)
                st.caption("2025: „N/A” (chyba że wynika wprost z #2).")
            except Exception as e:
                st.warning(f"Profil 2024 niezaładowany: {e}")

    # ---- 2) Top przewagi (Top 12 Edges — Week 6, 2025) ----
    with t2:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[1], "—")
            st.markdown(md if md.strip() else "—")
        else:
            tldr = summarize_top_edges_inline(game_obj, n=3) if game_obj else []
            if tldr:
                st.markdown("#### TL;DR – 3 rzeczy, które mają znaczenie")
                st.markdown("<div class='tldr'>" + "<br>".join([f"• {x}" for x in tldr]) + "</div>", unsafe_allow_html=True)
            else:
                st.info("Brak edges do podsumowania.")
            if game_obj and getattr(game_obj, "edges", None):
                st.markdown("#### Edge breakdown")
                rows = [{"Edge": e.name, "Team": (getattr(e, 'team', '—') or "—"), "Value": getattr(e, 'value', None)} for e in game_obj.edges]
                st.dataframe(rows, hide_index=True, use_container_width=True)
                st.caption("Największe wartości bezwzględne zwykle determinują dynamikę meczu.")
            else:
                st.info("No edges.")

    # ---- 3) Diagnoza matchupu (Game Dynamics) ----
    with t3:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[2], "—")
            st.markdown(md if md.strip() else "—")
        else:
            st.markdown("#### Dlaczego to działa (diagnosis)")
            if game_obj and getattr(game_obj, "signals", None):
                sig = game_obj.signals
                badge = confidence_badge(getattr(sig, "confidence", None))
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='kpi'><div class='label'>Side</div><div class='value'>%s</div></div>" % (getattr(sig, "side", None) or "—"), unsafe_allow_html=True)
                with c2:
                    st.markdown("<div class='kpi'><div class='label'>Total</div><div class='value'>%s</div></div>" % (getattr(sig, "total", None) or "—"), unsafe_allow_html=True)
                with c3:
                    st.markdown("<div class='kpi'><div class='label'>Confidence</div><div class='value'>%s</div></div>" % badge, unsafe_allow_html=True)
            if game_obj and getattr(game_obj, "why", None):
                for w in game_obj.why:
                    st.markdown(f"• {w}")
            else:
                st.caption("Dodaj listę 'why' w JSON, aby zobaczyć diagnozę.")
            md_path = detail_md_path(row.get("home_abbr",""), row.get("away_abbr",""), WEEK, SEASON)
            if md_path.exists():
                with st.expander(f"Notatki szczegółowe ({md_path.name})"):
                    st.markdown(md_path.read_text(encoding="utf-8"))

    # ---- 4) Model punktacji i prognoza ----
    with t4:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[3], "—")
            st.markdown(md if md.strip() else "—")
        else:
            st.markdown("#### Założenia modelu & wyniki")
            st.caption("Drives, opponent-adjusted PPD, 3D/RZ multipliers, Hidden Yards/ST, HFA, Monte Carlo.")
            md_path = detail_md_path(row.get("home_abbr",""), row.get("away_abbr",""), WEEK, SEASON)
            if md_path.exists():
                with st.expander("Opis modelu (markdown)"):
                    st.markdown(md_path.read_text(encoding="utf-8"))

    # ---- 5) Scenariusze gry (Game Scripts) ----
    with t5:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[4], "—")
            st.markdown(md if md.strip() else "—")
        else:
            st.markdown("#### Scenariusze")
            st.caption("Neutral / Positive HOME / Positive AWAY / Late-game.")
            # if game_obj and getattr(game_obj,'scripts',None):
            #     for k, lst in game_obj.scripts.items():
            #         st.markdown(f"**{k.capitalize()}**")
            #         for s in lst: st.markdown(f"• {s}")

    # ---- 6) Ryzyka i punkty krytyczne (Swing Factors) ----
    with t6:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[5], "—")
            st.markdown(md if md.strip() else "—")
        else:
            st.markdown("#### Ryzyka / Swing Factors")
            if game_obj and getattr(game_obj, "risks", None):
                for r in game_obj.risks:
                    st.markdown(f"• {r}")
            else:
                st.caption("Dodaj 'risks' do JSON, aby wyświetlić listę.")

    # ---- 7) Prognoza i typowanie (Final pick) ----
    with t7:
        if sections_from_paste:
            md = sections_from_paste.get(TAB_TITLES[6], "—")
            st.markdown(md if md.strip() else "—")
        else:
            st.markdown("#### Final pick (model / sygnały)")
            if game_obj and getattr(game_obj, "signals", None):
                sig = game_obj.signals
                badge = confidence_badge(getattr(sig, "confidence", None))
                st.markdown(f"- **Side:** {getattr(sig, 'side', None) or '—'}")
                st.markdown(f"- **Total:** {getattr(sig, 'total', None) or '—'}")
                st.markdown(f"- **Confidence:** {badge}", unsafe_allow_html=True)
            else:
                st.info("Brak sygnałów końcowych w JSON (side/total/confidence).")
                st.caption("Docelowo można tu pokazać: fair spread/total, medianę punktów, P20–P80, value vs market itp.")

    # ---- 8) Hidden Trends (micro-edges) ----
    with t8:
        # Prefer JSON if present, else compute from CSV on the fly
        meta = getattr(game_obj, "hidden_trends_meta", None)
        labels = meta.labels if (meta and getattr(meta, "labels", None)) else {
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
        tooltips = meta.tooltips if (meta and getattr(meta, "tooltips", None)) else {
            "game_rhythm_q4": "Tempo/rytm w Q4 – lepsze zamykanie meczów.",
            "field_flip_eff": "Zdolność do przesuwania pozycji startowej boiska.",
        }

        ht_df = load_hidden_trends_df()
        data_json = getattr(game_obj, "hidden_trends", None)
        edges_json = getattr(game_obj, "hidden_trends_edges", None)

        if data_json and edges_json:
            home_vals = data_json.get("home", {})
            away_vals = data_json.get("away", {})
            deltas = edges_json
        elif ht_df is not None:
            home_abbr = row.get("home_abbr")
            away_abbr = row.get("away_abbr")
            comp = compute_hidden_trends_edges(ht_df, home_abbr, away_abbr)
            home_vals = comp["home"]
            away_vals = comp["away"]
            deltas = comp["edges"]
        else:
            st.info("No Hidden Trends available.")
            home_vals, away_vals, deltas = {}, {}, {}

        # Build UI table rows
        rows = []
        for k in HIDDEN_TREND_COLS:
            hv = home_vals.get(k)
            av = away_vals.get(k)
            dv = deltas.get(k)
            # Directional emoji
            if dv is None:
                arrow = "≈"
            else:
                arrow = "↑" if dv > 0 else ("↓" if dv < 0 else "≈")
            label = labels.get(k, k)
            rows.append({
                "Metric": label,
                "Home": hv,
                "Away": av,
                "Δ (H–A)": dv,
                "Dir": arrow,
            })

        # Find top-3 absolute diffs to mark
        top3 = sorted(
            [(r["Metric"], abs(r["Δ (H–A)"]) if r["Δ (H–A)"] is not None else -1) for r in rows],
            key=lambda t: t[1], reverse=True
        )[:3]
        top_names = {name for name, v in top3 if v is not None and v >= 0}
        for r in rows:
            if r["Metric"] in top_names:
                r["Δ (H–A)"] = r["Δ (H–A)"] if r["Δ (H–A)"] is None else float(r["Δ (H–A)"])
                r["Edge"] = "🏷️ Edge"
            else:
                r["Edge"] = ""

        st.markdown("#### Hidden Trends (micro-edges)")
        st.caption("HOME vs AWAY with Δ = HOME – AWAY. Icons: ↑ Home edge, ↓ Away edge, ≈ small/none.")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if tooltips:
            st.caption("Tooltips:")
            for k, tip in tooltips.items():
                st.markdown(f"- **{labels.get(k, k)}**: {tip}")

    # Dolny przycisk powrotu
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    cols_footer = st.columns([0.4, 0.2, 0.4])
    with cols_footer[1]:
        if st.button("← Back to games", key="back_bottom", type="secondary"):
            close_analysis()
            st.stop()

# === Wywołanie widoku analizy (tylko jeśli aktywny) ===
if st.session_state.analysis_view and st.session_state.analysis_row is not None:
    render_analysis_view(st.session_state.analysis_row, st.session_state.analysis_game)
    st.stop()

# ============== LISTA GIER ==============
st.subheader(f"Week {WEEK} Games")
for i, row in enumerate(ps_games):
    cols = st.columns([0.07, 0.53, 0.20, 0.20])
    with cols[0]:
        st.markdown(f"**{i+1}.**")
    with cols[1]:
        st.markdown(f"**{row['away']} @ {row['home']}**")
    with cols[2]:
        key = game_key_from_abbr(row["home_abbr"], row["away_abbr"])
        st.caption(f"`{key}`")
    with cols[3]:
        if st.button("Analizuj", key=f"btn_analyze_{i}"):
            g_obj, _ = find_analysis_for(row["home"], row["away"])
            open_analysis(row, g_obj)

# ============== STOPKA ==============
st.markdown(
    f"""
    <hr>
    <div style='text-align:center; color:gray; font-size:0.9em;'>
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """,
    unsafe_allow_html=True
)
