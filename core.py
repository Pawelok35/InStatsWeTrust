# core.py — logika analizy meczów i kolejek

import pandas as pd
from utils import calculate_power_score, interpretuj_power_score, ocena_sygnalu
from data_loader import get_recent_form_with_xpts

# === ANALIZA JEDNEGO MECZU ===
def analyze_match(gospodarz, gosc, kolejka):
    from data_loader import load_data
    df_matches, df_long, _, _, _, _, _ = load_data()

    form_home = get_recent_form_with_xpts(df_long, gospodarz, 'home', kolejka)
    form_away = get_recent_form_with_xpts(df_long, gosc, 'away', kolejka)

    power_home = calculate_power_score(form_home)
    power_away = calculate_power_score(form_away)

    print("\n--- ANALIZA MECZU ---")
    print(f"{gospodarz} (HOME) PowerRating: {power_home}")
    print(f"{gosc} (AWAY) PowerRating: {power_away}")
    print(f"Sygnał: {ocena_sygnalu(power_home, power_away)}")

    return {
        'Gospodarz': gospodarz,
        'Gość': gosc,
        'Round': kolejka,
        'Forma Gospodarza': form_home,
        'Forma Gościa': form_away,
        'PowerRating Gospodarza': power_home,
        'PowerRating Gościa': power_away
    }


# === ANALIZA CAŁEJ KOLEJKI ===
def analyze_round(kolejka):
    from data_loader import load_data
    df_matches, df_long, _, _, _, _, _ = load_data()
    print(f"\n\033[95m📊 ANALIZA KOLEJKI {kolejka}\033[0m\n")
    df_kolejka = df_matches[df_matches['Round'] == kolejka]

    if df_kolejka.empty:
        print("⚠️ Brak meczów w tej kolejce.")
        return

    for _, row in df_kolejka.iterrows():
        home = row['Home']
        away = row['Away']
        form_home = get_recent_form_with_xpts(df_long, home, 'home', kolejka)
        form_away = get_recent_form_with_xpts(df_long, away, 'away', kolejka)
        power_home = calculate_power_score(form_home)
        power_away = calculate_power_score(form_away)

        diff = round((form_home.get('Śr. Punkty (5m)', 0) or 0) - (form_away.get('Śr. Punkty (5m)', 0) or 0), 2)
        sygnal = ocena_sygnalu(power_home, power_away)

        print(f"{home:<20} vs {away:<20} → Sygnał: {sygnal:<55} || RÓŻNICA: {diff:>5}")


# === POWER RANKING ===
def generate_power_ranking(kolejka):
    from data_loader import load_data
    df_matches, df_long, _, _, _, _, _ = load_data()
    teams = sorted(set(df_long['TEAM'].dropna().unique()))
    print(f"\n\033[96m📊 POWER RANKING PRZED KOLEJKĄ {kolejka}\033[0m\n")
    ranking = []

    for team in teams:
        home_form = get_recent_form_with_xpts(df_long, team, 'home', kolejka)
        away_form = get_recent_form_with_xpts(df_long, team, 'away', kolejka)

        power_home = calculate_power_score(home_form)
        power_away = calculate_power_score(away_form)

        if power_home is not None and power_away is not None:
            avg_power = round((power_home + power_away) / 2, 2)
        elif power_home is not None:
            avg_power = power_home
        elif power_away is not None:
            avg_power = power_away
        else:
            avg_power = None

        label = interpretuj_power_score(avg_power) if avg_power is not None else "brak danych"
        ranking.append((team, avg_power, label))

    ranking = sorted([r for r in ranking if r[1] is not None], key=lambda x: x[1], reverse=True)

    for idx, (team, power, opis) in enumerate(ranking, start=1):
        print(f"{idx:>2}. {team:<20} → {power:>5}  {opis}")


# === KOLEJKA Z WYNIKAMI ===
def analyze_round_with_results_xpts(kolejka, filtruj_silne_sygnaly=False):
    from data_loader import load_data, get_recent_form_with_xpts
    from utils import calculate_power_score, ocena_sygnalu

    df_matches, df_long, *_ = load_data()
    df_kolejka = df_matches[df_matches['Round'] == kolejka]

    for _, row in df_kolejka.iterrows():
        home = row['Home']
        away = row['Away']
        goals_home = row['Goals_For']
        goals_away = row['Goals_Against']


        form_home = get_recent_form_with_xpts(df_long, home, 'home', kolejka)
        form_away = get_recent_form_with_xpts(df_long, away, 'away', kolejka)

        pr_home = calculate_power_score(form_home)
        pr_away = calculate_power_score(form_away)
        diff = round(pr_home - pr_away, 1) if pr_home is not None and pr_away is not None else None
        signal = ocena_sygnalu(pr_home, pr_away)

        # Filtrowanie na żądanie
        if filtruj_silne_sygnaly and not ("(4/5)" in signal or "(5/5)" in signal):
            continue

        # Wynik meczu
        wynik = f"{int(goals_home)}–{int(goals_away)}" if pd.notna(goals_home) and pd.notna(goals_away) else "brak"

        print(f"{home:<21} vs {away:<21} → {signal:<60} || RÓŻNICA: {diff:>5} || Wynik: {wynik}")
