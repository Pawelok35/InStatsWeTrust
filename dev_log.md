# 📔 Development Journal – InStatsWeTrust

## 📅 2024-04-24
### ✅ Zrealizowane:
- Wprowadzenie koncepcji systematycznej analizy meczów.
- Ustalenie formatu wejściowego danych (mecze, tabele HOME/AWAY, dominacja, xG).
- Przygotowanie struktury funkcji `get_recent_form()` i `analyze_match()`.

### 💡 Pomysły:
- Analiza przeciwników wg pozycji w tabeli.
- Kolorystyczne wyróżnianie sekcji w terminalu.

---

## 📅 2024-04-25
### ✅ Zrealizowane:
- Dodanie wyświetlania xG gospodarza i gościa na podstawie 5 ostatnich meczów.
- Obliczanie średniej pozycji przeciwników.
- Formatowanie wyników przeciwników.

### 🐞 Naprawiono:
- Problem z `None` przy braku danych xG.
- Błąd w `get_recent_form()` gdy brakowało pozycji przeciwnika.

---

## 📅 2024-04-26
### ✅ Zrealizowane:
- Dodanie porównania średnich pozycji przeciwników gospodarza i gościa.
- Wyświetlanie nagłówków i podsumowań w kolorach.
- Komentarz: kto miał trudniejszych rywali.

### 💡 Pomysły:
- Wprowadzenie oceny formy (świetna/dobra/średnia/słaba).

---

## 📅 2024-04-27
### ✅ Zrealizowane:
- Dodanie średniego xG przeciwników.
- Segmentacja drużyn na Top 6, 7–10, 11–14, 15–20.
- Obliczenia bilansów W/R/P i średnich punktów w tych grupach.

### 🐞 Naprawiono:
- Błąd z `ValueError: cannot convert float NaN to integer`.
- Nieczytelne wyrównanie tekstu – dodano padding i kolumny.

---

## 📅 2024-04-28
### ✅ Zrealizowane:
- Wyświetlanie formy gospodarza/gościa z nazwą drużyny i aktualną pozycją w tabeli domowej/wyjazdowej.
- Ujednolicenie stylu: kolory, odstępy, sekcje.
- Dziennik aktywności przeniesiony do `dev_log.md`.

---

## 📅 2024-04-29
### ✅ Zrealizowane:
- Rozpoczęcie porządkowania repozytorium i plików `.py` oraz `.xlsx`.
- Dodano README i opis repozytorium na GitHubie.
- Wykluczenie `.venv` i dodanie `.gitignore`.

### 💡 Pomysły:
- Export analizy do pliku.
- Stworzenie wskaźnika siły drużyny 0–100.