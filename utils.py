# utils.py — funkcje pomocnicze do analizy Power Ratingu

import math
from scipy.stats import poisson
import pandas as pd

def safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def calculate_expected_points(xg, xga, max_goals=5):
    """
    Szacowanie xPTS na podstawie xG i xGA z wykorzystaniem rozkładu Poissona.
    """
    xg = safe_float(xg)
    xga = safe_float(xga)
    win_prob = draw_prob = loss_prob = 0.0

    for goals_for in range(0, max_goals + 1):
        for goals_against in range(0, max_goals + 1):
            p_for = poisson.pmf(goals_for, xg)
            p_against = poisson.pmf(goals_against, xga)
            prob = p_for * p_against

            if goals_for > goals_against:
                win_prob += prob
            elif goals_for == goals_against:
                draw_prob += prob
            else:
                loss_prob += prob

    xpts = 3 * win_prob + 1 * draw_prob
    return round(xpts, 3)


def calculate_power_score(form_data):
    xpts = safe_float(form_data.get('Śr. xPTS (5m)', 0))
    xg = safe_float(form_data.get('Śr. xG (5m)', 0))
    xga = safe_float(form_data.get('Śr. xG Przeciwników (5m)', 0))
    eff = safe_float(form_data.get('Efficiency_vs_Tier', 0))  # opcjonalne
    momentum = safe_float(form_data.get('Momentum', 0))
    sos = safe_float(form_data.get('Śr. Pozycja Przeciwników (5m)', 10))
    goal_diff = safe_float(form_data.get('Avg_Goal_Diff_5m', 0))

    xg_diff = xg - xga
    sos_correction = 1 - min(sos / 20, 1)

    score = (
        0.30 * xpts +
        0.20 * xg_diff +
        0.15 * eff +
        0.15 * momentum +
        0.10 * sos_correction +
        0.10 * goal_diff
    )
    return round(score, 3)


def interpretuj_power_score(score):
    if score is None:
        return "Brak danych"
    elif score >= 1.5:
        return "🔥 Top form"
    elif score >= 1.0:
        return "📈 Silna drużyna"
    elif score >= 0.5:
        return "⚖️ Średnia forma"
    elif score >= 0.0:
        return "🟡 Słaba forma"
    else:
        return "🔻 Kryzys"


def ocena_sygnalu(power_home, power_away):
    if power_home is None or power_away is None:
        return "🚫 Brak danych — brak decyzji"

    diff = power_home - power_away
    if diff > 1.5:
        return "✅ WARTO ZAGRAĆ na gospodarza (duża przewaga) (5/5)"
    elif diff > 1.0:
        return "🟢 Przewaga gospodarza (4/5)"
    elif diff > 0.5:
        return "🟡 Lekka przewaga gospodarza (3/5)"
    elif diff < -1.5:
        return "✅ WARTO ZAGRAĆ na gościa (duża przewaga) (5/5)"
    elif diff < -1.0:
        return "🟢 Przewaga gościa (4/5)"
    elif diff < -0.5:
        return "🟡 Lekka przewaga gościa (3/5)"
    else:
        return "🚫 Brak wyraźnej przewagi – lepiej odpuścić (1/5)"