import streamlit as st
import pandas as pd
from analyze_match import analyze_match

# === Wczytanie danych ===
df_matches = pd.read_excel("data/premier_league_match_data_detailed.xlsx")
wszystkie_druzyny = sorted(df_matches['Drużyna'].dropna().unique())
min_kolejka = int(df_matches['Kolejka'].min())
max_kolejka = int(df_matches['Kolejka'].max())

# === Tytuł aplikacji ===
st.title("⚽ InStatsWeTrust – Match Analyzer")

# === Wybór parametrów ===
col1, col2 = st.columns(2)
with col1:
    gospodarz = st.selectbox("Choose Home Team", wszystkie_druzyny)
with col2:
    gosc = st.selectbox("Choose Away Team", wszystkie_druzyny)

kolejka = st.slider("Select Matchday (Round)", min_kolejka, max_kolejka, min_kolejka)

# === Przycisk uruchamiający analizę ===
if st.button("Analyze Match"):
    wynik = analyze_match(gospodarz, gosc, kolejka)
    fg = wynik['Forma Gospodarza']
    fgosc = wynik['Forma Gościa']

    st.subheader(f"🔵 Home Team Form – {gospodarz}")
    st.write(f"**Average Points:** {fg.get('Śr. Punkty (5m)', 'N/A')}")
    st.write(f"**Average xG:** {fg.get('Śr. xG (5m)', 'N/A')}")
    st.write("**Recent Matches:**")
    for item in fg.get("Przeciwnicy Info", []):
        st.markdown(f"- {item}")

    st.subheader(f"🔴 Away Team Form – {gosc}")
    st.write(f"**Average Points:** {fgosc.get('Śr. Punkty (5m)', 'N/A')}")
    st.write(f"**Average xG:** {fgosc.get('Śr. xG (5m)', 'N/A')}")
    st.write("**Recent Matches:**")
    for item in fgosc.get("Przeciwnicy Info", []):
        st.markdown(f"- {item}")

    st.subheader("📊 Match Summary")
    st.write(f"**Result:** {wynik.get('Wynik meczu', 'N/A')}")
    st.write(f"**xG Home:** {wynik.get('xG Gospodarz', 'N/A')}")
    st.write(f"**xG Away:** {wynik.get('xG Gość', 'N/A')}")