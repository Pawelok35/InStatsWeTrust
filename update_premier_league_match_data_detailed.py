import pandas as pd

# Wczytaj Twoją istniejącą bazę
baza = pd.read_excel("data/premier_league_match_data_detailed.xlsx")

# Wczytaj nowy plik anglia2.xlsx (surowe dane)
anglia = pd.read_excel("data/premier_league_raw_match_data.xlsx")

def normalize(name):
    return str(name).lower().strip().replace(" fc", "").replace("\u00a0", "")

# Poprawne kolumny
baza['TEAM_norm'] = baza['TEAM'].apply(normalize)
baza['Opponent_norm'] = baza['Opponent'].apply(normalize)
anglia['Home_norm'] = anglia['Home'].apply(normalize)
anglia['Away_norm'] = anglia['Away'].apply(normalize)

# Tworzymy klucz identyfikujący mecz
baza['Klucz'] = baza['Round'].astype(str) + "_" + baza['TEAM_norm'] + "_" + baza['Opponent_norm'] + "_" + baza['Type']
anglia['Klucz_home'] = anglia['Round'].astype(str) + "_" + anglia['Home_norm'] + "_" + anglia['Away_norm'] + "_home"
anglia['Klucz_away'] = anglia['Round'].astype(str) + "_" + anglia['Away_norm'] + "_" + anglia['Home_norm'] + "_away"

# Słowniki xG i Score
xg_home_dict = dict(zip(anglia['Klucz_home'], anglia['xG']))
xg_away_dict = dict(zip(anglia['Klucz_away'], anglia['xG.1']))
score_home_dict = dict(zip(anglia['Klucz_home'], anglia['Score']))
score_away_dict = dict(zip(anglia['Klucz_away'], anglia['Score']))

# Funkcje przypisujące xG i Score
def znajdz_xg(row):
    if row['Type'] == 'home':
        return xg_home_dict.get(row['Klucz'], None)
    else:
        return xg_away_dict.get(row['Klucz'], None)

def znajdz_score(row):
    if row['Type'] == 'home':
        return score_home_dict.get(row['Klucz'], None)
    else:
        return score_away_dict.get(row['Klucz'], None)

# Uzupełniamy dane
baza['xG'] = baza.apply(znajdz_xg, axis=1)
baza['Score'] = baza.apply(znajdz_score, axis=1)

# Czyścimy
baza = baza.drop(columns=['TEAM_norm', 'Opponent_norm', 'Klucz'])

# Zapisujemy
baza.to_excel("data/premier_league_match_data_detailed.xlsx", index=False)

print("✅ Baza została zaktualizowana i nadpisana: data/premier_league_match_data_detailed.xlsx")
