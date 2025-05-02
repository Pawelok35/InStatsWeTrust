import pandas as pd
import os
print("👉 Working directory:", os.getcwd())

def update_match_data(raw_path, detailed_path):
    # Wczytaj dane
    raw_data = pd.read_excel(raw_path)
    detailed_data = pd.read_excel(detailed_path)

    # Oczyść nagłówki
    raw_data.columns = raw_data.columns.str.strip()
    detailed_data.columns = detailed_data.columns.str.strip()

    # Diagnostyka kolumn
    print("📋 Kolumny w raw_data:", raw_data.columns.tolist())

    # Tworzenie kolumn Goals_For i Goals_Against, jeśli ich nie ma
    if 'Goals_For' not in raw_data.columns or 'Goals_Against' not in raw_data.columns:
        print("⚠️ Brakuje kolumn Goals_For/Goals_Against – tworzymy je automatycznie z Score.")
        def split_score(score):
            for sep in ['–', '-', ':']:
                if sep in score:
                    parts = score.strip().split(sep)
                    return int(parts[0]), int(parts[1])
            return 0, 0  # fallback
        raw_data['Goals_For'], raw_data['Goals_Against'] = zip(*raw_data['Score'].apply(split_score))
    else:
        print("✅ Kolumny Goals_For i Goals_Against są obecne.")

    # Normalizacja drużyn
    for col in ['Home', 'Away']:
        raw_data[col] = raw_data[col].astype(str).str.strip()
    for col in ['TEAM', 'Opponent', 'Type']:
        if col in detailed_data.columns:
            detailed_data[col] = detailed_data[col].astype(str).str.strip()

    # Przygotowanie danych
    raw_data['Round'] = raw_data['Round'].astype(int)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])

    # Wiersze dla gospodarzy
    home_df = pd.DataFrame({
        'Round': raw_data['Round'],
        'TEAM': raw_data['Home'],
        'Opponent': raw_data['Away'],
        'Type': 'Home',
        'Points': raw_data.apply(lambda r: 3 if r['Goals_For'] > r['Goals_Against'] else 1 if r['Goals_For'] == r['Goals_Against'] else 0, axis=1),
        'Domination': raw_data.apply(lambda r: 1 if r['Goals_For'] - r['Goals_Against'] > 2 else 0, axis=1),
        'xG': raw_data['xG'],
        'xG_Opponent': raw_data['xG.1'],
        'Goals_For': raw_data['Goals_For'],
        'Goals_Against': raw_data['Goals_Against'],
        'Score': raw_data['Score']
    })

    # Wiersze dla gości
    away_df = pd.DataFrame({
        'Round': raw_data['Round'],
        'TEAM': raw_data['Away'],
        'Opponent': raw_data['Home'],
        'Type': 'Away',
        'Points': raw_data.apply(lambda r: 3 if r['Goals_Against'] > r['Goals_For'] else 1 if r['Goals_Against'] == r['Goals_For'] else 0, axis=1),
        'Domination': raw_data.apply(lambda r: 1 if r['Goals_Against'] - r['Goals_For'] > 2 else 0, axis=1),
        'xG': raw_data['xG.1'],
        'xG_Opponent': raw_data['xG'],
        'Goals_For': raw_data['Goals_Against'],
        'Goals_Against': raw_data['Goals_For'],
        'Score': raw_data['Score'].apply(lambda s: f"{s.split('–')[1]}–{s.split('–')[0]}" if '–' in s else s)
    })

    # Połącz dane
    transformed_raw = pd.concat([home_df, away_df], ignore_index=True)

    # Oczyść kluczowe kolumny
    for df in [transformed_raw, detailed_data]:
        df[['Round', 'TEAM', 'Opponent', 'Type']] = df[['Round', 'TEAM', 'Opponent', 'Type']].astype(str).apply(lambda x: x.str.strip())

    # Usuń z detailed_data istniejące mecze (będą nadpisane)
    detailed_data = detailed_data.merge(
        transformed_raw[['Round', 'TEAM', 'Opponent', 'Type']],
        on=['Round', 'TEAM', 'Opponent', 'Type'],
        how='left',
        indicator=True
    )
    detailed_data = detailed_data[detailed_data['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Połącz wszystko
    updated_detailed_data = pd.concat([detailed_data, transformed_raw], ignore_index=True)
    updated_detailed_data.drop_duplicates(subset=['Round', 'TEAM', 'Opponent', 'Type'], keep='first', inplace=True)

    # Sprawdź, czy zmieniono dane
    initial_len = len(detailed_data)
    final_len = len(updated_detailed_data)

    if final_len != initial_len:
        updated_detailed_data.to_excel(detailed_path, index=False)
        print(f"✅ Plik został zaktualizowany – nowa długość: {final_len} wierszy.")
    else:
        print("ℹ️ Brak zmian – plik nie został zmodyfikowany.")

# Uruchomienie
if __name__ == "__main__":
    update_match_data(
        raw_path="data/premier_league_raw_match_data.xlsx",
        detailed_path="data/premier_league_match_data_detailed.xlsx"
    )

