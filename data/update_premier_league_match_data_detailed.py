
import pandas as pd
import os

def update_match_data(raw_path, detailed_path):
    print("👉 Working directory:", os.getcwd())

    # Load data
    raw_data = pd.read_excel(raw_path)

    # Clean headers
    raw_data.columns = raw_data.columns.str.strip()

    # Fill missing goals columns if needed
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

    # Normalize teams
    for col in ['Home', 'Away']:
        raw_data[col] = raw_data[col].astype(str).str.strip()

    raw_data['Round'] = raw_data['Round'].astype(int)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])

    print(f"📊 Meczy w pliku RAW: {len(raw_data)}")
    print(f"🏟️ Unikalnych drużyn: {len(set(raw_data['Home']) | set(raw_data['Away']))}")
    print(f"📅 Kolejki: {raw_data['Round'].min()} - {raw_data['Round'].max()}")

    # Build home and away entries
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

    detailed_df = pd.concat([home_df, away_df], ignore_index=True)
    detailed_df.drop_duplicates(subset=['Round', 'TEAM', 'Opponent', 'Type'], keep='first', inplace=True)

    # Save to Excel
    detailed_df.to_excel(detailed_path, index=False)
    print(f"✅ Zapisano do pliku: {detailed_path}")
    print(f"🟢 Łącznie wierszy: {len(detailed_df)}")

# Run
if __name__ == "__main__":
    update_match_data(
        raw_path="data/premier_league_raw_match_data.xlsx",
        detailed_path="data/premier_league_match_data_detailed.xlsx"
    )
