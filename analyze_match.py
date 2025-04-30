import pandas as pd

# === KROK 1: WCZYTANIE DANYCH ===

df_long = pd.read_excel("data/premier_league_match_data_detailed.xlsx")
df_matches = pd.read_excel("data/premier_league_raw_match_data.xlsx")

home_table_path = "data/premier_league_team_tables_home_away.xlsx"
home_table = pd.read_excel(home_table_path, sheet_name="HomeTable")
away_table = pd.read_excel(home_table_path, sheet_name="AwayTable")

# === KROK 2: FUNKCJE POMOCNICZE ===
def normalize_team_name(name):
    return str(name).lower().strip().replace(" fc", "").replace("\u00a0", "")

def get_home_stats(druzyna, kolejka):
    norm_name = normalize_team_name(druzyna)
    home_table['Norm'] = home_table['Drużyna'].apply(normalize_team_name)
    df = home_table[(home_table['Round'] == kolejka - 1) & (home_table['Norm'] == norm_name)]
    if not df.empty:
        w = df.iloc[0]
        return {
            'Pozycja': int(w['Pozycja']),
            'M': int(w['M']),
            'W': int(w['W']),
            'D': int(w['D']),
            'L': int(w['L']),
            'GOALS': str(w['GOALS']),
            'DIF': int(w['DIF']),
            'PT': int(w['PT'])
        }
    return None

def get_latest_away_stats(druzyna, kolejka):
    norm_name = normalize_team_name(druzyna)
    away_table['Norm'] = away_table['Drużyna'].apply(normalize_team_name)
    df = away_table[(away_table['Round'] < kolejka) & (away_table['Norm'] == norm_name)]
    if not df.empty:
        w = df.sort_values(by='Round', ascending=False).iloc[0]
        return {
            'Pozycja': int(w['Pozycja']),
            'M': int(w['M']),
            'W': int(w['W']),
            'D': int(w['D']),
            'L': int(w['L']),
            'GOALS': str(w['GOALS']),
            'DIF': int(w['DIF']),
            'PT': int(w['PT'])
        }
    return None

def get_recent_form(druzyna, typ, kolejka):
    norm_name = normalize_team_name(druzyna)
    filt = (df_long['Drużyna'].apply(normalize_team_name) == norm_name) & \
           (df_long['Typ'] == typ) & \
           (df_long['Kolejka'] < kolejka)
    ostatnie = df_long[filt].sort_values(by='Kolejka', ascending=False).head(5)

    przeciwnicy_info = []
    pozycje_przeciwnikow = []
    xg_przeciwnikow = []
    tabela = home_table if typ == 'home' else away_table
    tabela['Norm'] = tabela['Drużyna'].apply(normalize_team_name)

    for _, row in ostatnie.iterrows():
        przeciwnik = row['Przeciwnik']
        kolejka_rywala = row['Kolejka']
        norm = normalize_team_name(przeciwnik)

        mecz_home = df_long[
            (df_long['Kolejka'] == kolejka_rywala) &
            (df_long['Drużyna'].apply(normalize_team_name) == norm) &
            (df_long['Typ'] == 'home')
        ]
        mecz_away = df_long[
            (df_long['Kolejka'] == kolejka_rywala) &
            (df_long['Drużyna'].apply(normalize_team_name) == norm) &
            (df_long['Typ'] == 'away')
        ]

        if not mecz_home.empty:
            typ_przeciwnika = 'home'
        elif not mecz_away.empty:
            typ_przeciwnika = 'away'
        else:
            przeciwnicy_info.append(f"{przeciwnik} (brak danych) | Wynik: brak danych | brak danych")
            continue

        tabela_przeciwnika = home_table if typ_przeciwnika == 'home' else away_table
        tabela_przeciwnika['Norm'] = tabela_przeciwnika['Drużyna'].apply(normalize_team_name)
        df_pos = tabela_przeciwnika[
            (tabela_przeciwnika['Round'] == kolejka_rywala) &
            (tabela_przeciwnika['Norm'] == norm)
        ]

        if df_pos.empty:
            pozycja = None
        else:
            poz_val = df_pos.iloc[0]['Pozycja']
            pozycja = int(poz_val) if pd.notna(poz_val) else None

        if pozycja is not None:
            pozycje_przeciwnikow.append(pozycja)

        # Wynik meczu
        wynik_meczu = row.get('Score') if 'Score' in row else 'brak wyniku'
        punkty = row.get('Punkty', 0)
        if punkty == 3:
            rezultat = "Wygrana"
        elif punkty == 1:
            rezultat = "Remis"
        else:
            rezultat = "Przegrana"

        przeciwnicy_info.append(f"{przeciwnik} ({pozycja if pozycja is not None else 'brak'}) | Wynik: {wynik_meczu} | {rezultat}")

        # xG przeciwnika
        xg_przeciwnika = row.get('xG.1') if typ == 'home' else row.get('xG')
        if pd.notna(xg_przeciwnika):
            xg_przeciwnikow.append(xg_przeciwnika)

    sr_pkt = round(float(ostatnie['Punkty'].mean()), 2) if not ostatnie.empty else None
    sr_xg = round(ostatnie['xG'].mean(), 2) if not ostatnie.empty and 'xG' in ostatnie.columns else None
    sr_pozycja = round(sum(pozycje_przeciwnikow) / len(pozycje_przeciwnikow), 2) if pozycje_przeciwnikow else None
    sr_xg_przeciwnikow = round(sum(xg_przeciwnikow) / len(xg_przeciwnikow), 2) if xg_przeciwnikow else None

    return {
        'Śr. Punkty (5m)': sr_pkt,
        'Śr. xG (5m)': sr_xg,
        'Śr. Pozycja Przeciwników (5m)': sr_pozycja,
        'Śr. xG Przeciwników (5m)': sr_xg_przeciwnikow,
        'Przeciwnicy Info': przeciwnicy_info
    }



# === KROK 3: ANALIZA MECZU ===

def analyze_match(gospodarz, gosc, kolejka):
    home_form = get_recent_form(gospodarz, 'home', kolejka)
    away_form = get_recent_form(gosc, 'away', kolejka)
    home_stats = get_home_stats(gospodarz, kolejka)
    away_stats = get_latest_away_stats(gosc, kolejka)
    df_match = df_matches[(df_matches['Round'] == kolejka) & (df_matches['Home'] == gospodarz) & (df_matches['Away'] == gosc)]
    xg_home = float(df_match['xG'].values[0]) if not df_match.empty and 'xG' in df_match.columns else None
    xg_away = float(df_match['xG.1'].values[0]) if not df_match.empty and 'xG.1' in df_match.columns else None
    wynik_meczu = str(df_match['Score'].values[0]) if not df_match.empty else None
    return {
        'Gospodarz': gospodarz,
        'Gość': gosc,
        'Kolejka': kolejka,
        'Forma Gospodarza': home_form,
        'Forma Gościa': away_form,
        'xG Gospodarz': xg_home,
        'xG Gość': xg_away,
        'Wynik meczu': wynik_meczu
    }

# === KROK 4: URUCHOMIENIE ANALIZY ===

if __name__ == "__main__":
    # Lista drużyn i kolejek
    druzyny_home = sorted(df_matches['Home'].dropna().unique())
    druzyny_away = sorted(df_matches['Away'].dropna().unique())
    wszystkie_druzyny = sorted(set(druzyny_home) | set(druzyny_away))
    min_kolejka = int(df_matches['Round'].min())
    max_kolejka = int(df_matches['Round'].max())

    # Wyświetl drużyny w kolumnach
    print("\nDostępne drużyny:")
    kolumny = 4
    licznik = 0
    linia = ""
    for team in wszystkie_druzyny:
        linia += f"{team:<25}"
        licznik += 1
        if licznik % kolumny == 0:
            print(linia)
            linia = ""
    if linia:
        print(linia)

    print(f"\nDostępny zakres kolejek: {min_kolejka} - {max_kolejka}")
    print("----------------------------------------")

    # Walidacja inputu
    def wybierz_druzyne(prompt_text):
        while True:
            team = input(prompt_text)
            if team in wszystkie_druzyny:
                return team
            else:
                print("\033[91mNie ma takiej drużyny. Spróbuj ponownie.\033[0m")

    def wybierz_kolejke(prompt_text):
        while True:
            try:
                kolejka = int(input(prompt_text))
                if min_kolejka <= kolejka <= max_kolejka:
                    return kolejka
                else:
                    print(f"\033[91mPodana kolejka jest poza zakresem ({min_kolejka}-{max_kolejka}).\033[0m")
            except ValueError:
                print("\033[91mMusisz wpisać liczbę!\033[0m")

    # Pobieranie danych
    gospodarz = wybierz_druzyne("Podaj nazwę drużyny gospodarzy: ")
    gosc = wybierz_druzyne("Podaj nazwę drużyny gości: ")
    kolejka = wybierz_kolejke("Podaj numer kolejki: ")

    # Analiza meczu
    wynik = analyze_match(gospodarz, gosc, kolejka)
    fg = wynik['Forma Gospodarza']
    fgosc = wynik['Forma Gościa']

      # FORMA GOSPODARZA
    print("\n\033[94m----------------------------------------")
    print("FORMA GOSPODARZA:")
    print(f"Średnia punktów: {fg.get('Śr. Punkty (5m)', 'brak danych')}")
    print(f"Średnie xG: {fg.get('Śr. xG (5m)', 'brak danych')}")
    print("Ostatnie mecze gospodarza:")
    for przeciwnik_info in fg.get('Przeciwnicy Info', []):
        parts = przeciwnik_info.split("|")
        if len(parts) == 3:
            nazwa = parts[0].strip()
            wynik_meczowy = parts[1].replace("Wynik:", "").strip()
            rezultat = parts[2].strip()
            print(f"- {nazwa:<30} Wynik: {wynik_meczowy} | {rezultat}")
        else:
            print(f"- {przeciwnik_info}")
    print("\033[0m")

    # FORMA GOŚCIA
    print("\n\033[91m----------------------------------------")
    print("FORMA GOŚCIA:")
    print(f"Średnia punktów: {fgosc.get('Śr. Punkty (5m)', 'brak danych')}")
    print(f"Średnie xG: {fgosc.get('Śr. xG (5m)', 'brak danych')}")
    print("Ostatnie mecze gościa:")
    for przeciwnik_info in fgosc.get('Przeciwnicy Info', []):
        parts = przeciwnik_info.split("|")
        if len(parts) == 3:
            nazwa = parts[0].strip()
            wynik_meczowy = parts[1].replace("Wynik:", "").strip()
            rezultat = parts[2].strip()
            print(f"- {nazwa:<30} Wynik: {wynik_meczowy} | {rezultat}")
        else:
            print(f"- {przeciwnik_info}")
    print("\033[0m")

    # === PODSUMOWANIE ===
    print("\n\033[93m========== PODSUMOWANIE ==========\033[0m")

    sr_pkt_home = fg.get('Śr. Punkty (5m)')
    sr_pkt_away = fgosc.get('Śr. Punkty (5m)')
    roznica_pkt = None
    faworyt_text = ""
    if sr_pkt_home is not None and sr_pkt_away is not None:
        roznica_pkt = round(sr_pkt_home - sr_pkt_away, 2)
        if roznica_pkt > 0:
            faworyt_text = f"RÓŻNICA NA KORZYŚĆ GOSPODARZA WYNOSI {roznica_pkt}"
        elif roznica_pkt < 0:
            faworyt_text = f"RÓŻNICA NA KORZYŚĆ GOŚCIA WYNOSI {abs(roznica_pkt)}"
        else:
            faworyt_text = "BRAK RÓŻNICY MIĘDZY GOSPODARZEM A GOŚCIEM."

    print(f"Śr. punkty Gospodarza (dom): {sr_pkt_home if sr_pkt_home is not None else 'brak danych'}")
    print(f"Śr. punkty Gościa (wyjazd): {sr_pkt_away if sr_pkt_away is not None else 'brak danych'}\n")
    print(f"\033[1m{faworyt_text}\033[0m\n")

    print(f"Średnie xG Gospodarza (z 5 meczy): {fg.get('Śr. xG (5m)', 'brak danych')}")
    print(f"Średnie xG przeciwników Gospodarza (z 5 meczy): {fg.get('Śr. xG Przeciwników (5m)', 'brak danych')}\n")
    print(f"Średnie xG Gościa (z 5 meczy): {fgosc.get('Śr. xG (5m)', 'brak danych')}")
    print(f"Średnie xG przeciwników Gościa (z 5 meczy): {fgosc.get('Śr. xG Przeciwników (5m)', 'brak danych')}\n")
    print(f"Średnia pozycja przeciwników Gospodarza: {fg.get('Śr. Pozycja Przeciwników (5m)', 'brak danych')}")
    print(f"Średnia pozycja przeciwników Gościa: {fgosc.get('Śr. Pozycja Przeciwników (5m)', 'brak danych')}\n")
    if isinstance(wynik, dict):
        print(f"Wynik meczu: {wynik.get('Wynik meczu', 'brak danych')}")
    else:
        print(f"Błąd: zmienna 'wynik' nie jest słownikiem, tylko: {type(wynik)}")


    # === DODATKOWE WNIOSKI ===
    print("\n\033[92m========== DODATKOWE WNIOSKI ==========\033[0m")

    def ocena_formy(srednia_punktow):
        if srednia_punktow is None:
            return "Brak danych"
        elif srednia_punktow > 2.0:
            return "Świetna forma"
        elif srednia_punktow > 1.5:
            return "Dobra forma"
        elif srednia_punktow > 1.0:
            return "Średnia forma"
        else:
            return "Słaba forma"

    forma_gospodarza = ocena_formy(sr_pkt_home)
    forma_goscia = ocena_formy(sr_pkt_away)
    print(f"Forma Gospodarza: {forma_gospodarza}")
    print(f"Forma Gościa: {forma_goscia}\n")

    xg_home = fg.get('Śr. xG (5m)')
    xg_away = fgosc.get('Śr. xG (5m)')
    xg_przeciwnicy_home = fg.get('Śr. xG Przeciwników (5m)')
    xg_przeciwnicy_away = fgosc.get('Śr. xG Przeciwników (5m)')

    if xg_home is not None and xg_away is not None:
        if xg_home > xg_away:
            print(f"Gospodarz generuje więcej xG (średnio {round(xg_home - xg_away, 2)} więcej).")
        elif xg_home < xg_away:
            print(f"Gość generuje więcej xG (średnio {round(xg_away - xg_home, 2)} więcej).")
        else:
            print("Obie drużyny generują podobne xG.")

    if xg_przeciwnicy_home is not None and xg_przeciwnicy_away is not None:
        if xg_przeciwnicy_home < xg_przeciwnicy_away:
            print(f"Gospodarz lepiej ogranicza przeciwników pod względem xG (średnio {round(xg_przeciwnicy_away - xg_przeciwnicy_home, 2)} mniej straconego xG).")
        elif xg_przeciwnicy_home > xg_przeciwnicy_away:
            print(f"Gość lepiej ogranicza przeciwników pod względem xG (średnio {round(xg_przeciwnicy_home - xg_przeciwnicy_away, 2)} mniej straconego xG).")
        else:
            print("Obie drużyny równie dobrze ograniczają przeciwników pod względem xG.")
