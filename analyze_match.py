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
    home_table['Norm'] = home_table['TEAM'].apply(normalize_team_name)
    df = home_table[(home_table['Round'] == kolejka - 1) & (home_table['Norm'] == norm_name)]
    if not df.empty:
        w = df.iloc[0]
        return {
            'Position': int(w['Position']),
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
    away_table['Norm'] = away_table['TEAM'].apply(normalize_team_name)
    df = away_table[(away_table['Round'] < kolejka) & (away_table['Norm'] == norm_name)]
    if not df.empty:
        w = df.sort_values(by='Round', ascending=False).iloc[0]
        return {
            'Position': int(w['Position']),
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

    filt = (df_long['TEAM'].apply(normalize_team_name) == norm_name) & \
           (df_long['Type'].str.lower() == typ.lower()) & \
           (df_long['Round'] < kolejka)

    ostatnie = df_long[filt].sort_values(by='Round', ascending=False).head(5)

    przeciwnicy_info = []
    pozycje_przeciwnikow = []
    xg_przeciwnikow = []

    tabela = home_table if typ == 'home' else away_table
    tabela['Norm'] = tabela['TEAM'].apply(normalize_team_name)

    for _, row in ostatnie.iterrows():
        przeciwnik = row['Opponent']
        kolejka_rywala = row['Round']
        norm = normalize_team_name(przeciwnik)

        # Szukamy pozycji przeciwnika w jego ostatnim meczu (home/away)
        tabela_home = home_table.copy()
        tabela_away = away_table.copy()
        tabela_home['Norm'] = tabela_home['TEAM'].apply(normalize_team_name)
        tabela_away['Norm'] = tabela_away['TEAM'].apply(normalize_team_name)

        tabela_mecz = pd.concat([
            tabela_home[(tabela_home['Round'] == kolejka_rywala) & (tabela_home['Norm'] == norm)],
            tabela_away[(tabela_away['Round'] == kolejka_rywala) & (tabela_away['Norm'] == norm)]
        ])

        if not tabela_mecz.empty:
            pozycja = tabela_mecz.iloc[0]['Position']
            if pd.notna(pozycja):
                pozycje_przeciwnikow.append(int(pozycja))
        else:
            pozycja = 'brak'

        # Wynik meczu i punktacja
        wynik_meczu = row.get('Score') if 'Score' in row else 'brak wyniku'
        punkty = row.get('Points', 0)
        rezultat = "Wygrana" if punkty == 3 else "Remis" if punkty == 1 else "Przegrana"
        przeciwnicy_info.append(f"{przeciwnik} ({pozycja}) | Wynik: {wynik_meczu} | {rezultat}")

        # xG przeciwnika z df_long
        opp_norm = normalize_team_name(przeciwnik)
        opp_typ = 'home' if typ == 'away' else 'away'
        opp_match = df_long[
            (df_long['TEAM'].apply(normalize_team_name) == opp_norm) &
            (df_long['Round'] == kolejka_rywala) &
            (df_long['Type'].str.lower() == opp_typ)
        ]

        xg_val = None
        if not opp_match.empty:
            if opp_typ == 'home':
                xg_val = opp_match.iloc[0].get('xG')
            else:
                xg_val = opp_match.iloc[0].get('xG_Opponent')

        if pd.notna(xg_val):
            xg_przeciwnikow.append(xg_val)

    sr_pkt = round(float(ostatnie['Points'].mean()), 2) if not ostatnie.empty else None
    sr_xg = round(ostatnie['xG'].mean(), 2) if not ostatnie.empty and 'xG' in ostatnie.columns else None
    sr_pozycja = round(sum(pozycje_przeciwnikow) / len(pozycje_przeciwnikow), 2) if pozycje_przeciwnikow else None
    sr_xg_przeciwnikow = round(sum(xg_przeciwnikow) / len(xg_przeciwnikow), 2) if xg_przeciwnikow else None

    dom_count = int(ostatnie['Domination'].sum()) if 'Domination' in ostatnie.columns else 0
    return {
        'Śr. Punkty (5m)': sr_pkt,
        'Śr. xG (5m)': sr_xg,
        'Śr. Pozycja Przeciwników (5m)': sr_pozycja,
        'Śr. xG Przeciwników (5m)': sr_xg_przeciwnikow,
        'Przeciwnicy Info': przeciwnicy_info
    }

def calculate_power_rating(form_data):
    """
    Oblicza wskaźnik siły drużyny (Power Rating) na podstawie formy z ostatnich 5 meczów.
    Uwzględnia średnie punkty, różnicę xG, jakość przeciwników i dominacje.
    """
    pkt = form_data.get('Śr. Punkty (5m)', 0) or 0
    xg = form_data.get('Śr. xG (5m)', 0) or 0
    xga = form_data.get('Śr. xG Przeciwników (5m)', 0) or 0
    poz = form_data.get('Śr. Pozycja Przeciwników (5m)', 20) or 20
    dominacje = form_data.get('Domination Count', 0) or 0

    diff_xg = xg - xga
    rating = (pkt * 2.0) + (diff_xg * 1.5) - (poz * 0.3) + (dominacje * 0.5)
    return round(rating, 2)


def get_average_opponent_position(mecze, team_tables_path, typ):
    import pandas as pd

    # Wczytaj plik z tabelami
    xls = pd.ExcelFile(team_tables_path)
    home_table = xls.parse(xls.sheet_names[0])
    away_table = xls.parse(xls.sheet_names[1])

    przeciwnicy = []

    for _, row in mecze.iterrows():
        round_nr = int(row['Round']) - 1  # patrzymy na poprzednią kolejkę
        opponent = row['Opponent']

        if typ.lower() == 'home':
            table = away_table
        else:
            table = home_table

        team_row = table[(table['Round'] == round_nr) & (table['TEAM'].str.strip() == opponent.strip())]

        if not team_row.empty:
            pos = team_row.iloc[0]['Position']
            try:
                pos = int(pos)
                przeciwnicy.append(pos)
            except:
                continue

    if przeciwnicy:
        return round(sum(przeciwnicy) / len(przeciwnicy), 2)
    else:
        return None


def get_season_form_vs_opponent_tiers(druzyna, typ, kolejka):
    """
    Analizuje wszystkie mecze danej drużyny (home/away) rozegrane do wskazanej kolejki.
    Dzieli wyniki względem pozycji przeciwników w tabeli w momencie danego meczu.
    """
    from collections import defaultdict

    norm_name = normalize_team_name(druzyna)

    # Filtrujemy mecze tej drużyny i typu (home/away) przed wskazaną kolejką
    filt = (
    (df_long['TEAM'].apply(normalize_team_name) == norm_name) &
    (df_long['Type'].str.lower() == typ.lower()) &
    (df_long['Round'] < kolejka)
    )
    mecze = df_long[filt]

    if mecze.empty:
        return {}

    tabela = home_table if typ == 'home' else away_table
    tabela['Norm'] = tabela['TEAM'].apply(normalize_team_name)

    grupy = {
        'Top 6': (1, 6),
        '7–10': (7, 10),
        '11–14': (11, 14),
        '15–20': (15, 20)
    }

    wynik = {k: {'Mecze': 0, 'Pkt': 0, 'W': 0, 'R': 0, 'P': 0, 'xG': 0.0, 'xGA': 0.0} for k in grupy}

    for _, row in mecze.iterrows():
        przeciwnik = normalize_team_name(row['Opponent'])
        runda = row['Round']

        # Znajdź pozycję przeciwnika z odpowiedniej tabeli z poprzedniej kolejki
        df_tabela = tabela[(tabela['Round'] == runda) & (tabela['Norm'] == przeciwnik)]

        if df_tabela.empty:
            continue

        poz_val = df_tabela.iloc[0]['Position']
        if pd.isna(poz_val):
            continue  # pomiń mecz bez pozycji
        pozycja = int(poz_val)


        # Określ grupę
        grupa_docelowa = None
        for nazwa, (low, high) in grupy.items():
            if low <= pozycja <= high:
                grupa_docelowa = nazwa
                break
        if grupa_docelowa is None:
            continue

        # Zbierz dane
        wynik[grupa_docelowa]['Mecze'] += 1
        wynik[grupa_docelowa]['Pkt'] += row.get('Points', 0)

        pkt = row.get('Points', 0)
        if pkt == 3:
            wynik[grupa_docelowa]['W'] += 1
        elif pkt == 1:
            wynik[grupa_docelowa]['R'] += 1
        else:
            wynik[grupa_docelowa]['P'] += 1

        if 'xG' in row and pd.notna(row['xG']):
            wynik[grupa_docelowa]['xG'] += row['xG']
        if 'xG_Opponent' in row and pd.notna(row['xG_Opponent']):
            wynik[grupa_docelowa]['xGA'] += row['xG_Opponent']

    # Wylicz średnie
    for grupa in wynik:
        mecze = wynik[grupa]['Mecze']
        if mecze > 0:
            wynik[grupa]['Śr. Pkt'] = round(wynik[grupa]['Pkt'] / mecze, 2)
            wynik[grupa]['Śr. xG'] = round(wynik[grupa]['xG'] / mecze, 2)
            wynik[grupa]['Śr. xGA'] = round(wynik[grupa]['xGA'] / mecze, 2)

    return wynik

# === KROK 3: ANALIZA MECZU ===

def analyze_match(gospodarz, gosc, kolejka):
    home_form = get_recent_form(gospodarz, 'home', kolejka)
    away_form = get_recent_form(gosc, 'away', kolejka)
    power_home = calculate_power_rating(home_form)
    power_away = calculate_power_rating(away_form)

    home_stats = get_home_stats(gospodarz, kolejka)
    away_stats = get_latest_away_stats(gosc, kolejka)
    df_match = df_matches[(df_matches['Round'] == kolejka) &
                          (df_matches['Home'] == gospodarz) &
                          (df_matches['Away'] == gosc)]
    xg_home = float(df_match['xG'].values[0]) if not df_match.empty and 'xG' in df_match.columns else None
    xg_away = float(df_match['xG_Opponent'].values[0]) if not df_match.empty and 'xG_Opponent' in df_match.columns else None
    wynik_meczu = str(df_match['Score'].values[0]) if not df_match.empty else None

    return {
        'Gospodarz': gospodarz,
        'Gość': gosc,
        'Round': kolejka,
        'Forma Gospodarza': home_form,
        'Forma Gościa': away_form,
        'xG Gospodarz': xg_home,
        'xG Gość': xg_away,
        'Wynik meczu': wynik_meczu,
        'Statystyki Gospodarza': home_stats,
        'Statystyki Gościa': away_stats,
        'PowerRating Gospodarza': power_home,
        'PowerRating Gościa': power_away

    }

# === KROK Analiza przyszłościowej kolejki ===
def analyze_round(kolejka):
    """
    Analizuje wszystkie mecze z danej kolejki i porównuje formę gospodarza vs gościa
    na podstawie średnich punktów z ostatnich 5 meczów (home/away).
    """
    print(f"\n\033[95m📊 ANALIZA KOLEJKI {kolejka}\033[0m\n")
    df_kolejka = df_matches[df_matches['Round'] == kolejka]

    if df_kolejka.empty:
        print("⚠️ Brak meczów w tej kolejce.")
        return

    for _, row in df_kolejka.iterrows():
        home = row['Home']
        away = row['Away']

        form_home = get_recent_form(home, 'home', kolejka)
        form_away = get_recent_form(away, 'away', kolejka)
        power_home = calculate_power_rating(form_home)
        power_away = calculate_power_rating(form_away)

        if power_home is not None and power_away is not None:
            diff = round(power_home - power_away, 2)
            if diff >= 1.0:
                symbol = "🟢" if diff > 0 else "🔴"
                label = f"{home if diff > 0 else away} clearly stronger ({diff:+})"
            elif abs(diff) >= 0.5:
                symbol = "🏠" if diff > 0 else "🛫"
                label = f"{home if diff > 0 else away} slightly stronger ({diff:+})"
            else:
                symbol = "⚖️"
                label = "Equal Power Rating"
        else:
            symbol = "🔍"
            label = "Not enough data"

        print(f"{home} vs {away} → {symbol} {label}")




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

    # Wybór trybu działania
    print("\nCo chcesz zrobić?")
    print("1. Przeanalizować jeden mecz ręcznie")
    print("2. Przeanalizować całą kolejkę")
    wybor = input("Wybierz opcję (1 lub 2): ").strip()

    if wybor == "1":
        gospodarz = wybierz_druzyne("Podaj nazwę drużyny gospodarzy: ")
        gosc = wybierz_druzyne("Podaj nazwę drużyny gości: ")
        kolejka = wybierz_kolejke("Podaj numer kolejki: ")
        wynik = analyze_match(gospodarz, gosc, kolejka)
        # W tym miejscu pozostaje Twoja dalsza analiza pojedynczego meczu

    elif wybor == "2":
        kolejka = wybierz_kolejke("Podaj numer kolejki do zbiorczej analizy: ")
        analyze_round(kolejka)

    else:
        print("Niepoprawny wybór.")

    
    fg = wynik['Forma Gospodarza']
    fgosc = wynik['Forma Gościa']
    power_home = wynik.get('PowerRating Gospodarza')
    power_away = wynik.get('PowerRating Gościa')

    # POBIERZ POZYCJE Z TABEL
    home_stats = get_home_stats(gospodarz, kolejka)
    away_stats = get_latest_away_stats(gosc, kolejka)

    

    # Statystyka vs poziom przeciwnika
    stats_home = get_season_form_vs_opponent_tiers(gospodarz, 'home', kolejka)
    stats_away = get_season_form_vs_opponent_tiers(gosc, 'away', kolejka) 

      # FORMA GOSPODARZA
    print("\n\033[94m----------------------------------------")
    print("FORMA GOSPODARZA:")
    pozycja_gospodarza = home_stats.get('Position') if home_stats else 'brak danych'
    print(f"\033[94mFORMA GOSPODARZA: {gospodarz} ({pozycja_gospodarza}. miejsce w tabeli domowej)\033[0m")
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
    pozycja_goscia = away_stats.get('Position') if away_stats else 'brak danych'
    print(f"{gosc} ({pozycja_goscia}. miejsce w tabeli wyjazdowej)")
  
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

print(f"Śr. punkty Gospodarza (dom - ostatnie 5 meczy): {sr_pkt_home if sr_pkt_home is not None else 'brak danych'}")
print(f"Śr. punkty Gościa (wyjazd - ostatnie 5 meczy): {sr_pkt_away if sr_pkt_away is not None else 'brak danych'}\n")
print(f"\033[1m{faworyt_text}\033[0m\n")

# === POWER RATING ===
power_home = calculate_power_rating(fg)
power_away = calculate_power_rating(fgosc)
print(f"📈 Power Rating Gospodarza: {power_home if power_home is not None else 'brak danych'}")
print(f"📈 Power Rating Gościa: {power_away if power_away is not None else 'brak danych'}\n")

# === xG i pozycje przeciwników ===
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

# === INTERPRETACJA POWER RATING ===
def interpretuj_power_rating(wartosc):
    """
    Interpretuje wartość Power Rating zgodnie z ustalonymi przedziałami.
    """
    if wartosc is None:
        return "brak danych"
    elif wartosc >= 2.5:
        return f"🟢 **ELITA** ({wartosc:+}) — znakomita forma, częste wysokie wygrane."
    elif wartosc >= 1.5:
        return f"🟢 **Bardzo mocna forma** ({wartosc:+}) — regularne zwycięstwa, dominacja nad słabszymi."
    elif wartosc >= 0.5:
        return f"🟡 **Dobra forma** ({wartosc:+}) — stabilna, przewaga nad przeciętnymi zespołami."
    elif wartosc > -0.5:
        return f"⚪ **Średnia forma** ({wartosc:+}) — wyrównany poziom, mecz może pójść w obie strony."
    elif wartosc > -1.5:
        return f"🟠 **Słaba forma** ({wartosc:+}) — drużyna traci punkty, podatna na porażki."
    elif wartosc > -2.5:
        return f"🔴 **Bardzo słaba forma** ({wartosc:+}) — częste porażki, brak skuteczności."
    else:
        return f"🔴 **Kryzys / bardzo słaby zespół** ({wartosc:+}) — fatalna forma, niezdolna do rywalizacji."


print("\n\033[95m========== INTERPRETACJA POWER RATING ==========\033[0m")
power_home = calculate_power_rating(fg)
power_away = calculate_power_rating(fgosc)

print(f"Power Rating Gospodarza: {power_home if power_home is not None else 'brak danych'} → {interpretuj_power_rating(power_home)}")
print(f"Power Rating Gościa: {power_away if power_away is not None else 'brak danych'} → {interpretuj_power_rating(power_away)}")



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

print(f"Forma Gospodarza: {ocena_formy(sr_pkt_home)}")
print(f"Forma Gościa: {ocena_formy(sr_pkt_away)}\n")

xg_home = fg.get('Śr. xG (5m)')
xg_away = fgosc.get('Śr. xG (5m)')
xg_przeciwnicy_home = fg.get('Śr. xG Przeciwników (5m)')
xg_przeciwnicy_away = fgosc.get('Śr. xG Przeciwników (5m)')

if xg_home is not None and xg_away is not None:
    diff = round(xg_home - xg_away, 2)
    if diff > 0:
        print(f"Gospodarz generuje więcej xG (średnio {diff} więcej).")
    elif diff < 0:
        print(f"Gość generuje więcej xG (średnio {abs(diff)} więcej).")
    else:
        print("Obie drużyny generują podobne xG.")

if xg_przeciwnicy_home is not None and xg_przeciwnicy_away is not None:
    diff = round(xg_przeciwnicy_away - xg_przeciwnicy_home, 2)
    if diff > 0:
        print(f"Gospodarz lepiej ogranicza przeciwników pod względem xG (średnio {diff} mniej straconego xG).")
    elif diff < 0:
        print(f"Gość lepiej ogranicza przeciwników pod względem xG (średnio {abs(diff)} mniej straconego xG).")
    else:
        print("Obie drużyny równie dobrze ograniczają przeciwników pod względem xG.")

# === STATYSTYKA VS. POZIOM PRZECIWNIKA ===
print("\n\033[96m========== STATYSTYKA VS. POZIOM PRZECIWNIKA ==========\033[0m")

print("\n\033[92mStatystyka Gospodarza: vs \033[0m")
for grupa, dane in stats_home.items():
    print(f"{grupa:<10} | Mecze: {dane['Mecze']:<2} | Śr. Pkt: {dane.get('Śr. Pkt', '-'):>4} | W:{dane['W']} R:{dane['R']} P:{dane['P']}")

print("\n\033[92mStatystyka Gościa: vs \033[0m")
for grupa, dane in stats_away.items():
    print(f"{grupa:<10} | Mecze: {dane['Mecze']:<2} | Śr. Pkt: {dane.get('Śr. Pkt', '-'):>4} | W:{dane['W']} R:{dane['R']} P:{dane['P']}")

