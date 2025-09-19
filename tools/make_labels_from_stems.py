import pandas as pd, os, re, textwrap

STEMS_CSV = "data/processed/metric_stems_2024.csv"
OUT_TXT   = "data/processed/metric_labels_update.txt"

# 1) Heurystyki "adnych" nazw
REPL = {
    r"\bepa\b": "EPA",
    r"\bsr\b": "Success Rate",
    r"\byds\b": "Yards",
    r"\bpg\b": "(pg)",
    r"\brz\b": "Red Zone",
    r"\b3d\b": "3rd Down",
    r"\b2d\b": "2nd Down",
    r"\b4d\b": "4th Down",
    r"\bto\b": "Turnover",
    r"\bxp\b": "XP",
    r"\bfg\b": "FG",
    r"\bko\b": "Kickoff",
    r"\batt\b": "Attempts",
    r"\bconv\b": "Conversions",
    r"\bfaced\b": "Faced",
    r"\ballowed\b": "Allowed",
    r"\bmedian\b": "Median",
    r"\bearly_down\b": "Early Down",
    r"\blate_down\b": "Late Down",
    r"\bexplosive\b": "Explosive",
    r"\bchunk20\b": "Chunk 20+",
    r"\bavg\b": "Avg",
    r"_per_play": " per play",
    r"_per_game": " per game",
    r"_rate": " Rate",
    r"_roll3": " (last 3)",
    r"_delta3": " Δ (last 3 vs season)",
    r"_h1\b": " 1H",
    r"_h2\b": " 2H",
}

PREFIX = [
    (r"^off_", "Offense · "),
    (r"^def_", "Defense · "),
    (r"^net_", "Net · "),
    (r"^team_", ""),   # raczej nie mamy tutaj, ale na wszelki
]

# 2) Rczne, lepsze tumaczenia (overrides)
OVERRIDE = {
    "avg_start_yardline_100": "Avg Starting Yards (O)",
    "avg_start_yardline_100_faced": "Avg Starting Yards Faced",
    "field_pos_advantage": "Field Position Edge",
    "st_score": "Special Teams Score",
    "pot_per_game": "Points off Turnovers (pg)",
    "third_fourth_pen_pg": "3rd/4th Down Penalties (pg)",
    "net_red_zone_epa": "Net Red Zone EPA",
    "net_late_down_epa": "Net Late Down EPA",
    "net_third_down_sr": "Net 3rd Down SR",
    "pass_rush_delta_off": "Pass Rush Delta (Off)",
}

def prettify(stem: str) -> str:
    if stem in OVERRIDE:
        return OVERRIDE[stem]

    label = stem

    # rozwin prefiksy
    for pat, rep in PREFIX:
        if re.search(pat, label):
            label = re.sub(pat, rep, label)
            break

    # zamieni podkrelniki na spacje (po prefiksach)
    label = label.replace("_", " ")

    # specjalne skróty/liczby
    label = re.sub(r"\bep a\b", "EPA", label, flags=re.I)  # safeguard
    # reguy z REPL
    for pat, rep in REPL.items():
        label = re.sub(pat, rep, label)

    # dopieszczanie: due litery na pocztku sów (zostaw skróty jak EPA/XP/FG)
    def title_keep_acronyms(w):
        return w if w.isupper() and len(w) <= 4 else (w.capitalize() if w.isalpha() else w)
    label = " ".join(title_keep_acronyms(w) for w in label.split())

    # kosmetyka kropek i separatorów
    label = label.replace("·  ", "· ").replace("  ", " ")
    return label.strip()

def load_stems(path):
    df = pd.read_csv(path)
    if "stem" not in df.columns:
        raise RuntimeError("CSV must have a 'stem' column")
    return sorted(df["stem"].dropna().unique())

def main():
    stems = load_stems(STEMS_CSV)
    pairs = []
    for s in stems:
        pairs.append((s, prettify(s)))

    # posortuj po kluczu (oryginalny stem)
    pairs.sort(key=lambda t: t[0])

    # zbuduj blok update
    lines = ["METRIC_LABELS.update({"]
    for k, v in pairs:
        # escape cudzysowów w labelach
        v = v.replace('"', '\\"')
        lines.append(f'    "{k}": "{v}",')
    lines.append("})\n")

    out_text = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(out_text)

    # poka podgld pierwszych 25
    preview = "\n".join(lines[:27])
    print("✅ Generated metric labels update. Preview:\n")
    print(preview)
    print(f"\n💾 Full block saved to: {OUT_TXT}")

if __name__ == "__main__":
    main()
