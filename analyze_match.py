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
           xg_val = opp_match.iloc[0].get('xG')


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

def get_efficiency_vs_opponent_tier(matches):
    tiers = {
        'top_6': (1, 6),
        'mid_7_10': (7, 10),
        'mid_11_14': (11, 14),
        'bottom_15_20': (15, 20)
    }

    tier_points = {'top_6': [], 'mid_7_10': [], 'mid_11_14': [], 'bottom_15_20': []}

    for m in matches:
        opp_pos = m.get('OpponentPosition')
        pts = m.get('Points')
        if opp_pos is None or pts is None:
            continue
        for tier, (low, high) in tiers.items():
            if low <= opp_pos <= high:
                tier_points[tier].append(pts)
                break

    weighted_sum = 0
    total_weight = 0
    weights = {'top_6': 1.3, 'mid_7_10': 1.1, 'mid_11_14': 0.9, 'bottom_15_20': 0.7}
    for tier, pts_list in tier_points.items():
        if pts_list:
            avg = sum(pts_list) / len(pts_list)
            weighted_sum += avg * weights[tier]
            total_weight += weights[tier]

    return weighted_sum / total_weight if total_weight > 0 else 0

def calculate_power_score(form_data):
    xpts = form_data.get('Śr. xPTS (5m)', 0) or 0
    xg = form_data.get('Śr. xG (5m)', 0) or 0
    xga = form_data.get('Śr. xG Przeciwników (5m)', 0) or 0
    pkt = form_data.get('Śr. Punkty (5m)', 0) or 0
    poz = form_data.get('Śr. Pozycja Przeciwników (5m)', 20) or 20
    dominacje = form_data.get('Domination Count', 0) or 0
    momentum = form_data.get('Momentum', 0) or 0
    efficiency = form_data.get('Efficiency_vs_Tier', 0) or 0

    diff_xg = xg - xga
    dominance_ratio = dominacje / 5  # na 5 ostatnich meczów

    score = (
        0.27 * xpts +
        0.18 * diff_xg +
        0.13 * pkt +
        0.10 * dominance_ratio +
        0.12 * (1 - poz / 20) +
        0.10 * momentum +
        0.10 * efficiency
    )

    score = max(0, min(score * 40, 100))  # skalowanie do 0–100
    return round(score, 2)

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

def ocena_sygnału_z_power_rating(power_home, power_away):
    roznica = power_home - power_away

    if power_home >= 20 and roznica >= 10:
        return "✅ Warto zagrać na GOSPODARZA – duża przewaga i dobra forma"
    elif power_away >= 20 and roznica <= -10:
        return "✅ Warto zagrać na GOŚCIA – duża przewaga i dobra forma"
    elif roznica >= 15:
        return "✅ Silny sygnał na GOSPODARZA"
    elif roznica <= -15:
        return "✅ Silny sygnał na GOŚCIA"
    elif 7 <= roznica < 15:
        return "🟡 Umiarkowana przewaga gospodarza"
    elif -15 < roznica <= -7:
        return "🟡 Umiarkowana przewaga gościa"
    else:
        return "🚫 Brak wyraźnej przewagi – lepiej nie grać"

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
    power_home = calculate_power_score(home_form)
    power_away = calculate_power_score(away_form)

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
        power_home = calculate_power_score(form_home)
        power_away = calculate_power_score(form_away)

        if power_home is None or power_away is None:
            print(f"{home} vs {away:<30} → 🔍 Brak danych")
            continue

        sr_pkt_home = form_home.get('Śr. Punkty (5m)', 0) or 0
        sr_pkt_away = form_away.get('Śr. Punkty (5m)', 0) or 0
        diff = round(sr_pkt_home - sr_pkt_away, 2)

        if diff > 0:
            faworyt = home
            roznica_txt = f"RÓŻNICA NA KORZYŚĆ GOSPODARZA WYNOSI {diff}"
        elif diff < 0:
            faworyt = away
            roznica_txt = f"RÓŻNICA NA KORZYŚĆ GOŚCIA WYNOSI {abs(diff)}"
        else:
            faworyt = "Brak przewagi"
            roznica_txt = "BRAK RÓŻNICY MIĘDZY DRUŻYNAMI"

        sygnal = ocena_sygnalu(power_home, power_away)  # np. 🟢 Przewaga gościa – możliwa okazja (4/5)

        # Jeśli chcesz pogrubić różnice przy dużych wartościach
        if abs(diff) >= 1.0:
            roznica_txt = f"\033[1m{roznica_txt}\033[0m"

        matchup = f"{home} vs {away}"
        line = (
        f"{matchup:<35} → {faworyt:<20} "
        f"|| Sygnał: {sygnal:<55} "
        f"|| {roznica_txt:<50}"
        )
        print(line)


# === INTERPRETACJA POWER RATING ===
def interpretuj_power_score(wartosc):
    if wartosc is None:
        return "brak danych"
    elif wartosc >= 20:
        return f"🟢 **ELITA** ({wartosc}) — znakomita forma, częste wysokie wygrane i dominacja."
    elif wartosc >= 15:
        return f"🟢 **Silna drużyna** ({wartosc}) — dobra forma, przewaga nad większością rywali."
    elif wartosc >= 10:
        return f"🟡 **Stabilna jakość** ({wartosc}) — umiarkowana forma, zdolność wygrywania z przeciętnymi."
    elif wartosc >= 5:
        return f"🟠 **Forma przeciętna / słabsza** ({wartosc}) — drużyna niestabilna lub w lekkim dołku."
    else:
        return f"🔴 **Kryzys / bardzo słaba forma** ({wartosc}) — częste porażki, duże problemy w grze."


def generate_power_ranking(kolejka):
    print(f"\n\033[96m📊 POWER RANKING PRZED KOLEJKĄ {kolejka}\033[0m\n")
    teams = sorted(set(df_long['TEAM'].dropna().unique()))
    ranking = []

    for team in teams:
        home_form = get_recent_form(team, 'home', kolejka)
        away_form = get_recent_form(team, 'away', kolejka)
        # 👇 DODAJ TUTAJ DEBUG
        print(f"\n=== {team.upper()} ===")
        print("🏠 HOME FORM:")
        for k, v in home_form.items():
            print(f"{k}: {v}")
        print("\n🛫 AWAY FORM:")
        for k, v in away_form.items():
            print(f"{k}: {v}")
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

def ocena_sygnalu(p1, p2):
    diff = p1 - p2
    if diff > 15:
        return "\033[1;32m✅ WARTO ZAGRAĆ na gospodarza (duża przewaga) (5/5)\033[0m"
    elif diff > 8:
        return "\033[1;36m🟢 Przewaga gospodarza – możliwa okazja (4/5)\033[0m"
    elif diff > 5:
        return "\033[93m🟡 Lekka przewaga gospodarza (3/5)\033[0m"
    elif diff > 2:
        return "\033[33m🟠 Bardzo lekka przewaga gospodarza (2/5)\033[0m"
    elif diff < -15:
        return "\033[1;32m✅ WARTO ZAGRAĆ na gościa (duża przewaga) (5/5)\033[0m"
    elif diff < -8:
        return "\033[1;36m🟢 Przewaga gościa – możliwa okazja (4/5)\033[0m"
    elif diff < -5:
        return "\033[93m🟡 Lekka przewaga gościa (3/5)\033[0m"
    elif diff < -2:
        return "\033[33m🟠 Bardzo lekka przewaga gościa (2/5)\033[0m"
    else:
        return "\033[91m🚫 Brak wyraźnej przewagi – lepiej odpuścić (1/5)\033[0m"

def is_prediction_correct(pred_team, home, away, home_goals, away_goals):
    if home_goals > away_goals:
        actual_winner = home
    elif home_goals < away_goals:
        actual_winner = away
    else:
        actual_winner = "Remis"
    return pred_team == actual_winner


from scipy.stats import poisson
import numpy as np

def calculate_xpts(xg_for, xg_against, max_goals=6):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, xg_for) * poisson.pmf(j, xg_against)
    p_win = np.sum(np.tril(prob_matrix, -1))
    p_draw = np.sum(np.diag(prob_matrix))
    xpts = 3 * p_win + 1 * p_draw
    return round(xpts, 3)

def get_recent_form_with_xpts(druzyna, typ, kolejka):
    norm_name = normalize_team_name(druzyna)
    filt = (df_long['TEAM'].apply(normalize_team_name) == norm_name) & \
           (df_long['Type'].str.lower() == typ.lower()) & \
           (df_long['Round'] < kolejka)
    ostatnie = df_long[filt].sort_values(by='Round', ascending=False).head(5)

    sr_pkt = round(float(ostatnie['Points'].mean()), 2) if not ostatnie.empty else 0
    sr_xg = round(ostatnie['xG'].mean(), 2) if not ostatnie.empty and 'xG' in ostatnie.columns else 0
    sr_xga = round(ostatnie['xG_Opponent'].mean(), 2) if not ostatnie.empty and 'xG_Opponent' in ostatnie.columns else 0
    dom_count = int(ostatnie['Domination'].sum()) if 'Domination' in ostatnie.columns else 0
    momentum = 1 if ostatnie.shape[0] >= 2 and ostatnie.iloc[0]['Points'] > ostatnie.iloc[-1]['Points'] else 0
    xpts_list = [
        calculate_xpts(row['xG'], row['xG_Opponent'])
        for _, row in ostatnie.iterrows()
        if pd.notna(row['xG']) and pd.notna(row['xG_Opponent'])
    ]
    xpts = round(sum(xpts_list) / len(xpts_list), 3) if xpts_list else 0

    return {
        'Śr. Punkty (5m)': sr_pkt,
        'Śr. xG (5m)': sr_xg,
        'Śr. xGA (5m)': sr_xga,
        'Domination Count': dom_count,
        'Momentum': momentum,
        'Śr. xPTS (5m)': xpts
    }

def analyze_round_with_results_xpts(kolejka):
    df_kolejka = df_matches[df_matches['Round'] == kolejka]
    results = []
    for _, row in df_kolejka.iterrows():
        home = row['Home']
        away = row['Away']
        goals_home = row['Goals_For']
        goals_away = row['Goals_Against']
        wynik_str = f"{goals_home}–{goals_away}"

        # Pobierz formę z ostatnich 5 meczów
        form_home = get_recent_form_with_xpts(home, 'home', kolejka)
        form_away = get_recent_form_with_xpts(away, 'away', kolejka)

        # Średnie punkty
        avg_pts_home = form_home.get('Śr. Punkty (5m)', 0)
        avg_pts_away = form_away.get('Śr. Punkty (5m)', 0)
        pts_diff = round(avg_pts_home - avg_pts_away, 2)
        

        # Dodatkowo możemy dalej liczyć PowerRating (jeśli chcesz używać sygnałów)
        pr_home = calculate_power_score(form_home)
        pr_away = calculate_power_score(form_away)
        signal = ocena_sygnalu(pr_home, pr_away)

        prediction = home if pts_diff > 0 else away if pts_diff < 0 else "Remis"
        tip_correct = is_prediction_correct(prediction, home, away, goals_home, goals_away)

        # Wypisz w konsoli
        status = "✅" if tip_correct else "❌"

        # Sformatuj wynik jako liczby całkowite + myślnik
        wynik_str = f"{int(goals_home)}–{int(goals_away)}"

        # Wyjustowana etykieta i liczba różnicy
        roznica_label = "RÓŻNICA:"
        roznica_text = f"{roznica_label:<10} {pts_diff:>4.1f}"

        # Koloruj całość (label + liczba), jeśli różnica ≥ 1
        if abs(pts_diff) >= 1:
            highlighted_roznica = f"\033[1;37m{roznica_text:<17}\033[0m"
        else:
            highlighted_roznica = f"{roznica_text:<17}"

        # Wyrównany i przejrzysty print
        print(
            f"{home:<20} vs {away:<20} → {prediction:<15} || "
            f"Sygnał: {signal:<50} || "
            f"{highlighted_roznica}|| Wynik: {wynik_str:<5} {status}"
        )




        # Zapisz do tabeli
        results.append({
            "Match": f"{home} vs {away}",
            "Predicted": prediction,
            "Signal": signal,
            "PointDiff (5m avg)": pts_diff,
            "Result": wynik_str,
            "TipCorrect": "✅" if tip_correct else "❌"
        })

    return pd.DataFrame(results)

def analyze_all_played_rounds_with_results_xpts():
    all_rounds = sorted(df_matches['Round'].unique())
    full_results = []

    print("\n📊 ANALIZA WSZYSTKICH KOLEJEK (tylko mecze z różnicą punktów ≥ 1)\n")

    for kolejka in all_rounds:
        df_kolejka = df_matches[df_matches['Round'] == kolejka]
        any_printed = False  # flaga: czy pokazano mecz z tej kolejki

        for _, row in df_kolejka.iterrows():
            home = row['Home']
            away = row['Away']
            goals_home = row['Goals_For']
            goals_away = row['Goals_Against']
            
            if pd.isna(goals_home) or pd.isna(goals_away):
                continue

            wynik_str = f"{int(goals_home)}–{int(goals_away)}"

            form_home = get_recent_form_with_xpts(home, 'home', kolejka)
            form_away = get_recent_form_with_xpts(away, 'away', kolejka)

            avg_pts_home = form_home.get('Śr. Punkty (5m)', 0)
            avg_pts_away = form_away.get('Śr. Punkty (5m)', 0)
            pts_diff = round(avg_pts_home - avg_pts_away, 2)

            # ⛔️ Pomiń, jeśli nie ma różnicy ≥ 1
            if abs(pts_diff) < 1:
                continue

            # tylko jeśli coś pokazujemy — pokaż nagłówek kolejki
            if not any_printed:
                print(f"\n📍 Kolejka {kolejka}\n")
                any_printed = True

            pr_home = calculate_power_score(form_home)
            pr_away = calculate_power_score(form_away)
            signal = ocena_sygnalu(pr_home, pr_away)

            prediction = home if pts_diff > 0 else away if pts_diff < 0 else "Remis"
            tip_correct = is_prediction_correct(prediction, home, away, goals_home, goals_away)
            status = "✅" if tip_correct else "❌"

            roznica_label = "RÓŻNICA:"
            roznica_text = f"{roznica_label:<10} {pts_diff:>4.1f}"

            highlighted_roznica = f"\033[1;37m{roznica_text:<17}\033[0m"

            print(
                f"{home:<20} vs {away:<20} → {prediction:<15} || "
                f"Sygnał: {signal:<50} || "
                f"{highlighted_roznica}|| Wynik: {wynik_str:<5} {status}"
            )

            full_results.append({
                "Round": kolejka,
                "Match": f"{home} vs {away}",
                "Predicted": prediction,
                "Signal": signal,
                "PointDiff (5m avg)": pts_diff,
                "Result": wynik_str,
                "TipCorrect": status
            })
    # PODSUMOWANIE trafień
    correct = sum(1 for r in full_results if r["TipCorrect"] == "✅")
    incorrect = sum(1 for r in full_results if r["TipCorrect"] == "❌")
    total = correct + incorrect
    accuracy = round(100 * correct / total, 1) if total > 0 else 0.0

    print("\n📈 PODSUMOWANIE TYPÓW")
    print(f"🎯 Trafione typy:     {correct}")
    print(f"❌ Nietrafione typy:  {incorrect}")
    print(f"📊 Skuteczność:       {accuracy}%")


    return pd.DataFrame(full_results)





def format_summary_output(fg, fgosc, wynik, power_home, power_away,
                          sr_pkt_home, sr_pkt_away,
                          xg_home, xg_away,
                          xg_przeciwnicy_home, xg_przeciwnicy_away,
                          stats_home, stats_away):
    output = []

    # === PODSUMOWANIE ===
    output.append("\n\033[93m========== PODSUMOWANIE ==========\033[0m")

    roznica_pkt = None
    if sr_pkt_home is not None and sr_pkt_away is not None:
        roznica_pkt = round(sr_pkt_home - sr_pkt_away, 2)
    if roznica_pkt >= 1.0:
        output.append(f"\n\033[97;1mRÓŻNICA NA KORZYŚĆ GOSPODARZA WYNOSI {roznica_pkt}\033[0m")
    elif roznica_pkt <= -1.0:
        output.append(f"\n\033[97;1mRÓŻNICA NA KORZYŚĆ GOŚCIA WYNOSI {abs(roznica_pkt)}\033[0m")
    else:
        output.append("\n\033[1mBRAK WYRAŹNEJ RÓŻNICY W ŚR. PUNKTACH.\033[0m")


    output.append(f"\n🎯 Sygnał: {ocena_sygnalu(power_home, power_away)}\n")

    output.append(f"Śr. punkty Gospodarza (dom - ostatnie 5 meczy): {sr_pkt_home if sr_pkt_home is not None else 'brak danych'}")
    output.append(f"Śr. punkty Gościa (wyjazd - ostatnie 5 meczy): {sr_pkt_away if sr_pkt_away is not None else 'brak danych'}\n")

    output.append(f"📈 Power Rating Gospodarza: {power_home} → {interpretuj_power_score(power_home)}")
    output.append(f"📈 Power Rating Gościa: {power_away} → {interpretuj_power_score(power_away)}\n")

    output.append(f"Średnie xG Gospodarza: {xg_home}")
    output.append(f"Średnie xG przeciwników Gospodarza: {xg_przeciwnicy_home}\n")
    output.append(f"Średnie xG Gościa: {xg_away}")
    output.append(f"Średnie xG przeciwników Gościa: {xg_przeciwnicy_away}\n")

    output.append(f"Śr. pozycja przeciwników Gospodarza: {fg.get('Śr. Pozycja Przeciwników (5m)', '-')}")
    output.append(f"Śr. pozycja przeciwników Gościa: {fgosc.get('Śr. Pozycja Przeciwników (5m)', '-')}\n")

    if isinstance(wynik, dict):
        output.append(f"Wynik meczu: {wynik.get('Wynik meczu', 'brak danych')}")

    # === DODATKOWE WNIOSKI ===
    output.append("\n\n\033[92m========== DODATKOWE WNIOSKI ==========\033[0m")

    def ocena_formy(s):
        if s is None:
            return "Brak danych"
        elif s > 2.0:
            return "Świetna forma"
        elif s > 1.5:
            return "Dobra forma"
        elif s > 1.0:
            return "Średnia forma"
        else:
            return "Słaba forma"

    output.append(f"\nForma Gospodarza: {ocena_formy(sr_pkt_home)}")
    output.append(f"Forma Gościa: {ocena_formy(sr_pkt_away)}")

    if xg_home is not None and xg_away is not None:
        diff = round(xg_home - xg_away, 2)
        if diff > 0:
            output.append(f"Gospodarz generuje więcej xG (średnio {diff} więcej).")
        elif diff < 0:
            output.append(f"Gość generuje więcej xG (średnio {abs(diff)} więcej).")
        else:
            output.append("Obie drużyny generują podobne xG.")

    if xg_przeciwnicy_home is not None and xg_przeciwnicy_away is not None:
        diff = round(xg_przeciwnicy_away - xg_przeciwnicy_home, 2)
        if diff > 0:
            output.append(f"Gospodarz lepiej ogranicza przeciwników pod względem xG (średnio {diff} mniej straconego xG).")
        elif diff < 0:
            output.append(f"Gość lepiej ogranicza przeciwników pod względem xG (średnio {abs(diff)} mniej straconego xG).")
        else:
            output.append("Obie drużyny równie dobrze ograniczają przeciwników pod względem xG.")

    if 'Efficiency_vs_Tier' in fg:
        output.append(f"\nEfektywność Gospodarza vs Tier: {round(fg['Efficiency_vs_Tier'], 2)}")
    if 'Efficiency_vs_Tier' in fgosc:
        output.append(f"Efektywność Gościa vs Tier: {round(fgosc['Efficiency_vs_Tier'], 2)}")

    # === STATYSTYKA VS. TIER ===
    output.append("\n\n\033[96m========== STATYSTYKA VS. POZIOM PRZECIWNIKA ==========\033[0m")
    output.append("\n\033[92mStatystyka Gospodarza: vs \033[0m")
    for grupa, dane in stats_home.items():
        output.append(f"{grupa:<10} | Mecze: {dane['Mecze']:<2} | Śr. Pkt: {dane.get('Śr. Pkt', '-') :>4} | W:{dane['W']} R:{dane['R']} P:{dane['P']}")

    output.append("\n\033[92mStatystyka Gościa: vs \033[0m")
    for grupa, dane in stats_away.items():
        output.append(f"{grupa:<10} | Mecze: {dane['Mecze']:<2} | Śr. Pkt: {dane.get('Śr. Pkt', '-') :>4} | W:{dane['W']} R:{dane['R']} P:{dane['P']}")

    return "\n".join(output)




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

    # Wybór trybu działania (z pętlą)

while True:
    print("\nCo chcesz zrobić?")
    print("1. Przeanalizować jeden mecz ręcznie")
    print("2. Przeanalizować całą kolejkę")
    print("3. Wygenerować Power Ranking przed daną kolejką")
    print("4. Przeanalizuj kolejkę z wynikami i xPTS")
    print("5. Przeanalizuj wszystkie kolejki z wynikami i xPTS")
    print("0. Wyjście")


    wybor = input("Wybierz opcję (0, 1, 2, 3, 4 lub 5): ").strip()

    if wybor == "1":
        gospodarz = wybierz_druzyne("Podaj nazwę drużyny gospodarzy: ")
        gosc = wybierz_druzyne("Podaj nazwę drużyny gości: ")
        kolejka = wybierz_kolejke("Podaj numer kolejki: ")
        wynik = analyze_match(gospodarz, gosc, kolejka)
        # Możesz tu dalej analizować mecz lub wyświetlić podsumowanie

    elif wybor == "2":
        kolejka = wybierz_kolejke("Podaj numer kolejki do zbiorczej analizy: ")
        analyze_round(kolejka)

    elif wybor == "3":
        kolejka = wybierz_kolejke("Podaj numer kolejki (Power Ranking będzie liczony na podstawie wcześniejszych danych): ")
        generate_power_ranking(kolejka)
    
    elif wybor == "4":
        kolejka = wybierz_kolejke("Podaj numer kolejki do analizy z wynikami: ")
        analyze_round_with_results_xpts(kolejka)

    elif wybor == "5":
        analyze_all_played_rounds_with_results_xpts()



    elif wybor == "0":
        print("Zakończono działanie programu. 👋")
        break

    else:
        print("Niepoprawny wybór. Spróbuj ponownie.")




