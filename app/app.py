import streamlit as st
from datetime import datetime
from pathlib import Path

from utils import (
    parse_week_analysis, game_label,
    load_games_from_ps1, game_key_from_abbr,
    confidence_badge, detail_md_path, equal_names
)


# === Full-page analysis view state ===
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



st.set_page_config(
    page_title="In Stats We Trust – NFL Insights Dashboard",
    page_icon="🏈",
    layout="wide",
)

# --- HEADER ---
# --- Global CSS: większe przyciski secondary ---
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <h1 style='text-align: center; color:#00FF88;'>In Stats We Trust</h1>
    <p style='text-align: center; color: #cccccc;'>NFL Insights Dashboard</p>
    <hr style='border:1px solid #00FF88'>
    """,
    unsafe_allow_html=True
)

DATA_PATH = Path("data/processed/analyses/week_6_2025_analysis.json")
PS1_PATH = Path("run_week_matchups.ps1")  # <- tu czytamy Twoje $games

@st.cache_data(show_spinner=False)
def load_week():
    return parse_week_analysis(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_games():
    return load_games_from_ps1(PS1_PATH)

# ---- Data status ----
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

# Helper: znajdź analizę dla meczu po pełnych nazwach


SEASON = 2025
WEEK = 6

def find_analysis_for(home_full: str, away_full: str):
    if not wa:
        return None, "No JSON loaded."
    for g in wa.games:
        if equal_names(g.home, home_full) and equal_names(g.away, away_full):
            return g, None
    return None, "No detailed analysis found for this matchup in week JSON."


# ======= FULL PAGE ANALYSIS VIEW (zamiast modala) =======
if st.session_state.analysis_view and st.session_state.analysis_row is not None:
    row = st.session_state.analysis_row
    game_obj = st.session_state.analysis_game

    # Górny przycisk powrotu (większy dzięki CSS + type="secondary")
    st.markdown("<br>", unsafe_allow_html=True)
    cols_header = st.columns([0.2, 0.8])
    with cols_header[0]:
        if st.button("← Back to games", key="back_top", type="secondary"):
            close_analysis()
            st.stop()

    # Tytuł analizy
    st.markdown(f"## Analiza: {row['away']} @ {row['home']}")
    st.markdown("---")

    left, right = st.columns([1.2, 1.8], gap="large")

    with left:
        st.markdown("### Signals"); st.markdown("<br>", unsafe_allow_html=True)
        if game_obj and game_obj.signals:
            sig = game_obj.signals
            badge = confidence_badge(sig.confidence)
            st.markdown(
                f"""
                <div style="padding:10px;border:1px solid #114;border-radius:12px;">
                    <div><b>Side:</b> {sig.side or '—'}</div>
                    <div><b>Total:</b> {sig.total or '—'}</div>
                    <div><b>Confidence:</b> {badge}</div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.info("No signals.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Edges"); st.markdown("<br>", unsafe_allow_html=True)
        if game_obj and game_obj.edges:
            rows = [{"Edge": e.name, "Team": e.team or "—", "Value": e.value} for e in game_obj.edges]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No edges.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Risks"); st.markdown("<br>", unsafe_allow_html=True)
        if game_obj and game_obj.risks:
            for r in game_obj.risks:
                st.markdown(f"• {r}")
        else:
            st.info("No risks.")

    with right:
        st.markdown("---")
        st.markdown("### Why it matters"); st.markdown("<br>", unsafe_allow_html=True)
        if game_obj and game_obj.why:
            for w in game_obj.why:
                st.markdown(f"✅ {w}")
        else:
            st.info("No 'why' notes.")

        st.markdown("<br>", unsafe_allow_html=True); st.markdown("---"); st.markdown("<br>", unsafe_allow_html=True)
        md_path = detail_md_path(row["home_abbr"], row["away_abbr"], WEEK, SEASON)
        if md_path.exists():
            st.markdown(f"**Detailed analysis** (`{md_path.name}`)")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(md_path.read_text(encoding="utf-8"))
        else:
            st.caption(f"_No markdown file found at_ `{md_path}`")

    # Dolny przycisk powrotu
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    cols_footer = st.columns([0.4, 0.2, 0.4])
    with cols_footer[1]:
        if st.button("← Back to games", key="back_bottom", type="secondary"):
            close_analysis()
            st.stop()

    st.stop()



# ---- RENDER LISTY GIER ----
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

# --- FOOTER ---
st.markdown(
    f"""
    <hr style='border:1px solid #00FF88'>
    <div style='text-align:center; color:gray; font-size:0.9em;'>
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """,
    unsafe_allow_html=True
)
