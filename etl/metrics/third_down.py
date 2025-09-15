# etl/metrics/third_down.py
import pandas as pd

EXCLUDE_FLAGS = ["qb_spike", "qb_kneel"]  # 1 == wykluczamy

def _safe_col(df: pd.DataFrame, name: str, default=0):
    """Zwraca istniejącą kolumnę albo Series z domyślną wartością (dopasowaną do indeksu df)."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)

def _is_attempt(df: pd.DataFrame) -> pd.Series:
    """Próba 3rd down: down == 3 i nie spike/kneel. 'no_play' zostawiamy (może być penalty-FD)."""
    mask = (df["down"] == 3)
    for f in EXCLUDE_FLAGS:
        if f in df.columns:
            mask &= (_safe_col(df, f).fillna(0).astype(int) != 1)
    return mask

def _is_conversion(df: pd.DataFrame) -> pd.Series:
    """Konwersja: gain FD lub TD lub penalty dające FD."""
    got_fd_gain = _safe_col(df, "first_down").fillna(0).astype(int).eq(1)
    td          = _safe_col(df, "touchdown").fillna(0).astype(int).eq(1)
    fd_pen      = _safe_col(df, "first_down_penalty").fillna(0).astype(int).eq(1)
    return got_fd_gain | td | fd_pen

def _agg_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    side: 'offense' (group by posteam) lub 'defense' (group by defteam)
    Zwraca agregaty po (season, week, team).
    """
    team_col = "posteam" if side == "offense" else "defteam"
    grp_cols = ["season", "week", team_col]

    df = df.copy()

    # maski
    att_mask  = _is_attempt(df)
    conv_mask = _is_conversion(df)

    # kolumny pomocnicze (na poziomie akcji)
    df["att_int"]      = att_mask.astype(int)
    df["conv_int"]     = (att_mask & conv_mask).astype(int)
    df["succ_int"]     = df["conv_int"]  # sukces == konwersja na 3rd down
    dropback           = _safe_col(df, "qb_dropback").fillna(0).astype(int)
    df["dropback_att"] = (att_mask & dropback.eq(1)).astype(int)

    epa = _safe_col(df, "epa").fillna(0.0).astype(float)
    ytg = _safe_col(df, "ydstogo").fillna(0.0).astype(float)
    df["epa_on_att"] = epa.where(att_mask, 0.0)
    df["ytg_on_att"] = ytg.where(att_mask, 0.0)

    grouped = df.groupby(grp_cols, dropna=False).agg(
        att=("att_int", "sum"),
        conv=("conv_int", "sum"),
        succ=("succ_int", "sum"),
        drop_att=("dropback_att", "sum"),
        epa_sum=("epa_on_att", "sum"),
        epa_cnt=("att_int", "sum"),           # liczba prób 3rd
        ytg_sum=("ytg_on_att", "sum"),
    ).reset_index()

    grouped.rename(columns={team_col: "team"}, inplace=True)

    def _div(a, b):
        return (a / b).where(b != 0, 0.0)

    if side == "offense":
        grouped["off_3d_att"]             = grouped["att"]
        grouped["off_3d_conv"]            = grouped["conv"]
        grouped["off_3d_rate"]            = _div(grouped["conv"], grouped["att"])
        grouped["off_3d_sr"]              = _div(grouped["succ"], grouped["att"])
        grouped["off_3d_pass_rate"]       = _div(grouped["drop_att"], grouped["att"])
        grouped["off_3d_avg_togo"]        = _div(grouped["ytg_sum"], grouped["att"])
        grouped["off_3d_epa_per_play"]    = _div(grouped["epa_sum"], grouped["epa_cnt"])

        keep = ["season","week","team",
                "off_3d_att","off_3d_conv","off_3d_rate","off_3d_sr",
                "off_3d_pass_rate","off_3d_avg_togo","off_3d_epa_per_play"]
    else:
        grouped["def_3d_att_faced"]            = grouped["att"]
        grouped["def_3d_allowed_conv"]         = grouped["conv"]
        grouped["def_3d_allowed_rate"]         = _div(grouped["conv"], grouped["att"])
        grouped["def_3d_sr_allowed"]           = _div(grouped["succ"], grouped["att"])
        grouped["def_3d_pass_rate_faced"]      = _div(grouped["drop_att"], grouped["att"])
        grouped["def_3d_avg_togo_faced"]       = _div(grouped["ytg_sum"], grouped["att"])
        grouped["def_3d_epa_allowed_per_play"] = _div(grouped["epa_sum"], grouped["epa_cnt"])

        keep = ["season","week","team",
                "def_3d_att_faced","def_3d_allowed_conv","def_3d_allowed_rate","def_3d_sr_allowed",
                "def_3d_pass_rate_faced","def_3d_avg_togo_faced","def_3d_epa_allowed_per_play"]

    return grouped[keep]

def third_down_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    """Zwraca weekly O/D 3rd down metrics per team."""
    off  = _agg_side(pbp, "offense")
    deff = _agg_side(pbp, "defense")

    weekly = pd.merge(off, deff, on=["season","week","team"], how="outer")

    # <-- kluczowa zmiana: fillna tylko dla kolumn liczbowych
    num_cols = weekly.select_dtypes(include="number").columns
    weekly[num_cols] = weekly[num_cols].fillna(0)

    return weekly.sort_values(["season","week","team"])


def third_down_season(weekly: pd.DataFrame) -> pd.DataFrame:
    """Agregacja sezonowa: sumy + wskaźniki z sum (stabilniejsze)."""
    grp = weekly.groupby(["season","team"], as_index=False).agg({
        "off_3d_att": "sum",
        "off_3d_conv": "sum",
        "off_3d_epa_per_play": "mean",
        "off_3d_sr": "mean",
        "off_3d_rate": "mean",
        "off_3d_pass_rate": "mean",
        "off_3d_avg_togo": "mean",

        "def_3d_att_faced": "sum",
        "def_3d_allowed_conv": "sum",
        "def_3d_epa_allowed_per_play": "mean",
        "def_3d_sr_allowed": "mean",
        "def_3d_allowed_rate": "mean",
        "def_3d_pass_rate_faced": "mean",
        "def_3d_avg_togo_faced": "mean",
    })

    def _div(a, b):
        return (a / b).where(b != 0, 0.0)

    # przeliczamy rate z sum (bardziej stabilne)
    grp["off_3d_rate"]        = _div(grp["off_3d_conv"], grp["off_3d_att"])
    grp["def_3d_allowed_rate"]= _div(grp["def_3d_allowed_conv"], grp["def_3d_att_faced"])
    return grp
