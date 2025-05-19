# data_loader.py — ładowanie danych i drużyn

import pandas as pd
from utils import calculate_expected_points

def load_data():
    # Wczytaj szczegółowe dane meczowe
    df_long = pd.read_excel("data/premier_league_match_data_detailed.xlsx")

    # Wczytaj plik z tabelami HOME i AWAY
    home_table = pd.read_excel("data/premier_league_team_tables_home_away.xlsx", sheet_name="HomeTable")
    away_table = pd.read_excel("data/premier_league_team_tables_home_away.xlsx", sheet_name="AwayTable")

    # Wczytaj dane surowe (mecze z wynikami)
    df_matches = pd.read_excel("data/premier_league_raw_match_data.xlsx")

    # Lista drużyn
    druzyny_home = sorted(df_matches['Home'].dropna().unique())
    druzyny_away = sorted(df_matches['Away'].dropna().unique())
    wszystkie_druzyny = sorted(set(druzyny_home) | set(druzyny_away))

    # Zakres kolejek
    min_kolejka = int(df_matches['Round'].min())
    max_kolejka = int(df_matches['Round'].max())

    return df_matches, df_long, home_table, away_table, wszystkie_druzyny, min_kolejka, max_kolejka


def wybierz_druzyne(lista, prompt_text):
    while True:
        team = input(prompt_text)
        if team in lista:
            return team
        else:
            print("\033[91mNie ma takiej drużyny. Spróbuj ponownie.\033[0m")


def wybierz_kolejke(min_kolejka, max_kolejka, prompt_text):
    while True:
        try:
            kolejka = int(input(prompt_text))
            if min_kolejka <= kolejka <= max_kolejka:
                return kolejka
            else:
                print(f"\033[91mPodana kolejka jest poza zakresem ({min_kolejka}-{max_kolejka}).\033[0m")
        except ValueError:
            print("\033[91mMusisz wpisać liczbę!\033[0m")


def get_recent_form_with_xpts(df_detailed, druzyna, typ, kolejka):
    def normalize_team_name(name):
        return str(name).lower().strip().replace(" fc", "").replace("\u00a0", "")

    team_norm = normalize_team_name(druzyna)
    type_norm = typ.lower().strip()

    df = df_detailed[
        (df_detailed['TEAM'].apply(normalize_team_name) == team_norm) &
        (df_detailed['Type'].str.lower().str.strip() == type_norm) &
        (df_detailed['Round'] < kolejka)
    ].sort_values(by='Round', ascending=False).head(5)

    if df.empty:
        print(f"[⚠️ BRAK DANYCH] {druzyna} ({typ}) do kolejki {kolejka}")
        return {
            'Śr. xPTS (5m)': None,
            'Śr. xG (5m)': None,
            'Śr. xG Przeciwników (5m)': None,
            'Śr. Punkty (5m)': None,
            'Śr. Pozycja Przeciwników (5m)': None,
            'Avg_Goal_Diff_5m': None
        }

    # Oblicz xPTS dla każdego meczu na żywo
    df['xPTS'] = df.apply(lambda row: calculate_expected_points(row['xG'], row['xG_Opponent']), axis=1)

    xpts = df['xPTS'].mean()
    xg = df['xG'].mean()
    xga = df['xG_Opponent'].mean()
    punkty = df['Points'].mean()
    pozycja = df['Opponent_Position'].mean() if 'Opponent_Position' in df.columns else None
    goal_diff = round((df['Goals_For'] - df['Goals_Against']).mean(), 2)

    return {
        'Śr. xPTS (5m)': xpts,
        'Śr. xG (5m)': xg,
        'Śr. xG Przeciwników (5m)': xga,
        'Śr. Punkty (5m)': punkty,
        'Śr. Pozycja Przeciwników (5m)': pozycja,
        'Avg_Goal_Diff_5m': goal_diff
    }


def get_all_played_rounds(df_matches):
    return sorted(df_matches['Round'].dropna().unique())