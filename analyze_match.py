# analyze_match.py — główny plik z menu

from data_loader import load_data, wybierz_druzyne, wybierz_kolejke
from core import (
    analyze_match,
    analyze_round,
    generate_power_ranking,
    analyze_round_with_results_xpts
)

def get_all_played_rounds(df_matches):
    return sorted(df_matches['Round'].dropna().unique())

if __name__ == "__main__":
    df_matches, df_long, home_table, away_table, wszystkie_druzyny, min_kolejka, max_kolejka = load_data()

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

    while True:
        print("\nCo chcesz zrobić?")
        print("1. Przeanalizować jeden mecz ręcznie")
        print("2. Przeanalizować całą kolejkę")
        print("3. Wygenerować Power Ranking przed daną kolejką")
        print("4. Przeanalizuj kolejkę z wynikami i xPTS")
        print("5. Wyświetl analizę wszystkich rozegranych kolejek")
        print("6. Wyświetl tylko mecze z silnym sygnałem (4/5 lub 5/5)")
        print("7. Przetestuj silne sygnały z alternatywnym PowerRatingiem (bez Momentum)")

        print("0. Wyjście")

        wybor = input("Wybierz opcję (0–7): ").strip()

        if wybor == "1":
            gospodarz = wybierz_druzyne(wszystkie_druzyny, "Podaj nazwę drużyny gospodarzy: ")
            gosc = wybierz_druzyne(wszystkie_druzyny, "Podaj nazwę drużyny gości: ")
            kolejka = wybierz_kolejke(min_kolejka, max_kolejka, "Podaj numer kolejki: ")
            wynik = analyze_match(gospodarz, gosc, kolejka)

        elif wybor == "2":
            kolejka = wybierz_kolejke(min_kolejka, max_kolejka, "Podaj numer kolejki do zbiorczej analizy: ")
            analyze_round(kolejka)

        elif wybor == "3":
            kolejka = wybierz_kolejke(min_kolejka, max_kolejka, "Podaj numer kolejki (Power Ranking będzie liczony na podstawie wcześniejszych danych): ")
            generate_power_ranking(kolejka)

        elif wybor == "4":
            kolejka = wybierz_kolejke(min_kolejka, max_kolejka, "Podaj numer kolejki do analizy z wynikami: ")
            analyze_round_with_results_xpts(kolejka)

        elif wybor == "5":
            kolejki = get_all_played_rounds(df_matches)
            for kolejka in kolejki:
                print(f"\n\033[96m📅 KOLEJKA {kolejka}\033[0m")
                analyze_round_with_results_xpts(kolejka)

        elif wybor == "6":
            kolejki = get_all_played_rounds(df_matches)
            statystyki = {'typy': 0, 'trafione': 0}
            for kolejka in kolejki:
                print(f"\n\033[96m📅 KOLEJKA {kolejka}\033[0m")
                analyze_round_with_results_xpts(kolejka, filtruj_silne_sygnaly=True, statystyki=statystyki)

            print("\n📊 PODSUMOWANIE SILNYCH SYGNAŁÓW:")
            print(f"Liczba typów: {statystyki['typy']}")
            print(f"Trafionych:   {statystyki['trafione']}")
            if statystyki['typy'] > 0:
                skutecznosc = 100 * statystyki['trafione'] / statystyki['typy']
                print(f"Skuteczność:  {skutecznosc:.2f}%")
            else:
                print("Brak danych do wyliczenia skuteczności.")

        elif wybor == "7":
            from data_loader import load_data, get_all_played_rounds
            from core import analyze_round_with_results_xpts_v2


            df_matches, *_ = load_data()
            kolejki = get_all_played_rounds(df_matches)
            statystyki = {'typy': 0, 'trafione': 0}
            for kolejka in kolejki:
                print(f"\n\033[96m📅 KOLEJKA {kolejka}\033[0m")
                analyze_round_with_results_xpts_v2(kolejka, filtruj_silne_sygnaly=True, statystyki=statystyki)

            print("\n📊 PODSUMOWANIE SILNYCH SYGNAŁÓW (WARIANT V2):")
            print(f"Liczba typów: {statystyki['typy']}")
            print(f"Trafionych:   {statystyki['trafione']}")
            if statystyki['typy'] > 0:
                skutecznosc = 100 * statystyki['trafione'] / statystyki['typy']
                print(f"Skuteczność:  {skutecznosc:.2f}%")
            else:
                print("Brak danych do wyliczenia skuteczności.")



        elif wybor == "0":
            print("Zakończono działanie programu. 👋")
            break

        else:
            print("Niepoprawny wybór. Spróbuj ponownie.")