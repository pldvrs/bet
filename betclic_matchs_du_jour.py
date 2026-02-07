#!/usr/bin/env python3
"""
Terminal de Décision — V25 Sniper Edition (Spécialisation Ligue + Spread + Absences)
====================================================================================
- V25 : League Profiler (BASELINE_PACE, HOME_ADVANTAGE, DEFENSIVE_FACTOR par ligue).
  Calculateur Handicap (Fair Spread), Star Impact Toggle (-10% Off_Rating), Money Buckets.
- V24 : Agrégation manuelle 100% (get_team_stats_weighted use_game_logs=True par défaut).
- V23 : SOS, Bayesian Shrinkage, Volatilité, Tale Adj. Off Rtg.
"""

import re
import time
from urllib.parse import quote_plus
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm, truncnorm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_KEY: str = "78179dc577a0de09af11266e4bddbdac"
BASE_URL: str = "https://v1.basketball.api-sports.io"
HEADER_KEY: str = "x-apisports-key"

# Source de vérité — Pro B FR = 8 ; Greek Basket League = 45 ; Italy Lega A = 52
LEAGUES: Dict[str, int] = {
    "Betclic Élite (FR)": 2,
    "Pro B (FR)": 8,
    "EuroLeague": 120,
    "LBA Italie": 4,
    "Lega A (Italie)": 52,
    "Liga ACB (ESP)": 5,
    "Turquie BSL": 194,
    "Grèce HEBA": 198,
    "Greek Basket League": 45,
    "ABA League (ADR)": 206,
}

# Filtre géographique strict
LEAGUE_COUNTRY: Dict[int, str] = {
    2: "France", 8: "France",
    5: "Spain", 4: "Italy", 52: "Italy",
    194: "Turkey", 198: "Greece", 45: "Greece",
    120: "Europe", 206: "Europe",
}

# V25 Sniper — League Profiler : spécialisation par ligue (NBA 98 poss, Euro ~70)
# BASELINE_PACE : possessions/match référence (fallback si pas de data)
# HOME_ADVANTAGE : pts avantage domicile (EuroLeague/Turquie > NBA)
# DEFENSIVE_FACTOR : poids défense (1.0 = neutre ; >1 = ligue défensive)
LEAGUE_CONFIGS: Dict[int, Dict[str, float]] = {
    2: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.05},   # Betclic Élite FR
    8: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.02},   # Pro B FR
    120: {"baseline_pace": 72.0, "home_advantage": 4.0, "defensive_factor": 1.08}, # EuroLeague
    4: {"baseline_pace": 70.0, "home_advantage": 3.5, "defensive_factor": 1.05},   # LBA Italie
    52: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.04},  # Lega A Italie
    5: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.03},   # Liga ACB
    194: {"baseline_pace": 74.0, "home_advantage": 4.5, "defensive_factor": 1.02}, # Turquie BSL
    198: {"baseline_pace": 70.0, "home_advantage": 4.0, "defensive_factor": 1.06}, # Grèce HEBA
    45: {"baseline_pace": 70.0, "home_advantage": 4.0, "defensive_factor": 1.06},  # Greek Basket League
    206: {"baseline_pace": 72.0, "home_advantage": 3.0, "defensive_factor": 1.02}, # ABA League
}
DEFAULT_LEAGUE_CONFIG: Dict[str, float] = {
    "baseline_pace": 72.0, "home_advantage": 3.0, "defensive_factor": 1.0,
}


def _get_league_config(league_id: Optional[int]) -> Dict[str, float]:
    """V25 — Retourne la config ligue ou défaut."""
    if league_id is not None and league_id in LEAGUE_CONFIGS:
        return LEAGUE_CONFIGS[league_id].copy()
    return DEFAULT_LEAGUE_CONFIG.copy()


# V24 Deep Data : agrégation manuelle Box Scores (batching) ; fallback API si échec
GAME_LOGS_MAX: int = 10  # Suffisant pour tendance, rapide à charger (1–2 appels batch)
MIN_GAME_LOGS_EXPLOITABLE: int = 3  # Si moins de 3 matchs exploitables → calcul manuel échoué
TIME_MACHINE_LAST_N: int = 5  # Derniers N matchs avant target_date pour backtest
# V22 — Saison actuelle API (Janvier 2026 = saison 2025-2026 → "2025")
CURRENT_SEASON_API: str = "2025"
# Saisons à tenter pour Time Machine : courante puis précédente (forcer box-scores saison en cours)
TIME_MACHINE_SEASONS: List[str] = [CURRENT_SEASON_API, "2024"]
TIME_MACHINE_MIN_GAMES: int = 5  # Si moins, on fusionne avec saison précédente
# Diagnostic Time Machine (rempli pendant run_backtest, affiché dans l'UI)
TIME_MACHINE_DEBUG_LOGS: List[str] = []
EDGE_MAX_BET_THRESHOLD: float = 10.0  # Edge > 10% → Conseil "🔥 MAX BET"
FIABILITE_MAE_GREEN: float = 6.0
FIABILITE_MAE_YELLOW: float = 12.0

# Saisons à tenter (Fix Pro B) : ordre de tir 2025-2026 en premier — ne pas rester bloqué sur "2025" si l'API attend "2025-2026"
SEASONS_TO_TRY: List[str] = ["2025-2026", "2025", "2024-2025", "2024"]
WINDOW_DAYS: int = 3
WEIGHT_LAST5: float = 0.6
WEIGHT_SEASON: float = 0.4
PYTHAGOREAN_EXP: float = 10.2  # FIBA / basket européen (11.5 trop agressif → probas extrêmes)
PROB_SMOOTH_FACTOR: float = 0.85  # Lissage anti-overconfidence : prob_lissée = raw * 0.85 + 0.075
PROB_SMOOTH_BIAS: float = 0.075
HOME_ADVANTAGE_PTS: float = 3.0
LAST_N_GAMES: int = 5

# Moteur physique FIBA : Pace = possessions par match (norme ~70)
PACE_DEFAULT_LIGUE: float = 72.0  # si stats Poss indisponibles
PACE_MIN: float = 65.0
PACE_MAX: float = 85.0
SCORE_PROJ_MIN: float = 60.0
SCORE_PROJ_MAX: float = 120.0
# Reality Check / Fiabilité V15
REALITY_MAE_FIABLE: float = 5.0
REALITY_MAE_INADAPTE: float = 15.0
REALITY_N_GAMES: int = 3
WEIGHT_FORM: float = 0.6
WEIGHT_SEASON_ADJ: float = 0.4
ORB_BONUS_THRESHOLD_PCT: float = 5.0
ORB_BONUS_SCORE_PCT: float = 0.03
TOV_MALUS_THRESHOLD_PCT: float = 3.0
TOV_MALUS_SCORE_PCT: float = 0.03

# Bookmakers majeurs: Betclic (17), Unibet/10Bet (7), Bwin (1). Over/Under Games = ID 4 ou 5.
BOOKMAKER_IDS: List[int] = [17, 7, 1]
BOOKMAKER_IDS_OU: List[int] = [17, 7, 1]
BET_HOME_AWAY_ID: int = 2
BET_OVER_UNDER_IDS: List[int] = [4, 5, 8, 16, 18]  # Over/Under, Over/Under Games, etc.
MIN_ODD_COMBINE: float = 1.30
MAX_ODD_SINGLE: float = 10.0   # V22 : ignorer cotes individuelles > 10 (Small Markets / hallucinations)
MAX_ODD_COMBINE_TOTAL: float = 25.0  # V22 : ignorer cote totale combiné > 25
PROBA_MIN_MAIN: float = 30.0  # Paris Moneyline avec Notre Proba < 30% → Paris Risqués / Longshots
KELLY_FRACTION: float = 4.0   # Fractional Kelly : mise = kelly_full / KELLY_FRACTION
# V23 — Strength of Schedule (SRS) et Bayesian
LEAGUE_AVG_DEF_RTG: float = 112.0  # pts/100 poss moyenne ligue (référence SOS)
BAYESIAN_SHRINKAGE_GAMES: float = 15.0  # Alpha = min(1, n_games / 15) pour régression à la moyenne
LEAGUE_AVG_EFG: float = 0.50
LEAGUE_AVG_TOV: float = 0.15
LEAGUE_AVG_ORB: float = 0.25
LEAGUE_AVG_FT_RATE: float = 0.25
# V23 — Volatilité (écart-type pts marqués sur 5 derniers)
VOLATILITY_STD_THRESHOLD: float = 15.0  # std > 15 → pénalité proba + avertissement
VOLATILITY_PROB_PENALTY_PCT: float = 5.0
WINDOW_PAST_BACKTEST: int = 2  # J-1 et J-2 pour le backtest P&L
EDGE_MIN_BET: float = 5.0  # Edge min pour compter un pari dans le backtest
TOTAL_PTS_SIGMA: float = 11.0  # écart-type pour P(Over) / P(Under)
EV_GREEN_THRESHOLD: float = 5.0
N_SIM: int = 5000
PTS_PALIERS: List[int] = [10, 12, 15, 18, 20, 25]
REB_AST_PALIERS: List[int] = [3, 4, 5, 6, 8, 10]
API_RETRIES: int = 3
API_RETRY_DELAY: float = 1.0

# ==============================================================================
# API
# ==============================================================================


def _api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{BASE_URL}/{endpoint}"
    headers = {HEADER_KEY: API_KEY.strip()}
    last_err: Optional[str] = None
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params or {}, timeout=15)
            data = r.json() if r.text else {}
            if r.status_code != 200:
                last_err = data.get("errors") or f"HTTP {r.status_code}"
                if r.status_code == 429:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                continue
            errs = data.get("errors")
            if errs:
                msg = errs[0] if isinstance(errs, list) else str(errs)
                if isinstance(msg, dict):
                    msg = "; ".join(f"{k}: {v}" for k, v in msg.items())
                last_err = msg
                continue
            return data, None
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(API_RETRY_DELAY * (attempt + 1))
    return None, last_err


def fetch_games(league_id: int, season: str, date_str: str) -> List[dict]:
    data, err = _api_get("games", {"date": date_str, "league": league_id, "season": season, "timezone": "Europe/Paris"})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


LEAGUE_PACE_HISTORY_DAYS: int = 30  # Jours en arrière pour estimer le pace ligue
LEAGUE_PACE_MIN_GAMES: int = 3  # Min de matchs avec box score pour moyenner le pace


def _estimate_pace_from_league_history(league_id: int, season: str) -> Optional[float]:
    """
    Calcule le pace moyen réel de la ligue sur les derniers matchs terminés (box scores).
    Ne plus utiliser 72.0 comme valeur fixe : on estime depuis l'historique.
    Retourne None si impossible (pas assez de box scores).
    """
    today = date.today()
    collected: List[dict] = []
    for d in range(1, LEAGUE_PACE_HISTORY_DAYS + 1):
        day_date = today - timedelta(days=d)
        date_str = day_date.strftime("%Y-%m-%d")
        games = fetch_games(league_id, season, date_str)
        for g in games:
            gid = g.get("id")
            if not gid:
                continue
            scores = g.get("scores") or {}
            sh = scores.get("home")
            sa = scores.get("away")
            pts_h = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
            pts_a = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)
            if (pts_h or 0) > 0 or (pts_a or 0) > 0:
                collected.append({"game_id": gid})
            if len(collected) >= 10:
                break
        if len(collected) >= 10:
            break
    if len(collected) < LEAGUE_PACE_MIN_GAMES:
        return None
    game_ids = [x["game_id"] for x in collected[:10]]
    batch_items = fetch_game_statistics_teams_batch(game_ids, max_per_call=20)
    poss_list: List[float] = []
    seen_game: set = set()
    for item in batch_items:
        gid = item.get("game_id") or (item.get("game") or {}).get("id")
        if not gid or gid in seen_game:
            continue
        raw = _extract_raw_stats_from_team_game(item)
        if raw and raw.get("Possessions", 0) > 0:
            poss_list.append(float(raw["Possessions"]))
            seen_game.add(gid)
    if len(poss_list) < LEAGUE_PACE_MIN_GAMES:
        return None
    return float(np.clip(sum(poss_list) / len(poss_list), PACE_MIN, PACE_MAX))


def fetch_games_by_date_only(league_id: int, date_str: str) -> Tuple[List[dict], str]:
    """
    Fallback : récupère les matchs par date + ligue SANS saison (certaines API renvoient alors les matchs).
    Retourne (liste des matchs, saison à utiliser extraite du 1er match ou "2024-2025").
    """
    data, err = _api_get("games", {"date": date_str, "league": league_id, "timezone": "Europe/Paris"})
    if err or not data:
        return [], ""
    resp = data.get("response")
    games = resp if isinstance(resp, list) else []
    used_season = "2024-2025"
    if games:
        g0 = games[0]
        league = g0.get("league") if isinstance(g0.get("league"), dict) else {}
        if league and league.get("season") is not None:
            used_season = str(league.get("season", used_season))
    return games, used_season


def fetch_team_statistics(league_id: int, season: str, team_id: int) -> Optional[dict]:
    """API /statistics : agrégats saison (points, rebonds, etc.). Peut être vide pour certaines ligues/saisons."""
    data, err = _api_get("statistics", {"league": league_id, "season": season, "team": team_id})
    if err or not data:
        return None
    resp = data.get("response")
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, list) and len(resp) > 0:
        return resp[0]
    return None


def _stats_from_scores_only(league_id: int, season: str, team_id: int) -> Optional[dict]:
    """
    V24 : Fallback quand Box Scores et API /statistics échouent.
    Construit les stats à partir des scores des derniers matchs (games avec pts_for/pts_against).
    Essaie la saison passée puis SEASONS_TO_TRY pour maximiser les données (cadors, Monaco, etc.).
    Retourne None si moins de 2 matchs avec scores.
    """
    for try_season in [season] + [s for s in SEASONS_TO_TRY if s != season]:
        last5 = fetch_team_games(league_id, try_season, team_id)
        last5 = [x for x in last5 if (x.get("pts_for") or 0) > 0 or (x.get("pts_against") or 0) > 0]
        if len(last5) < 2:
            continue
        ppg = sum(x.get("pts_for", 0) for x in last5) / len(last5)
        oppg = sum(x.get("pts_against", 0) for x in last5) / len(last5)
        if ppg <= 0:
            ppg = 75.0
        if oppg <= 0:
            oppg = 75.0
        pace = _estimate_pace_from_league_history(league_id, try_season) or PACE_DEFAULT_LIGUE
        off_rating = 100.0 * ppg / pace if pace > 0 else 100.0
        def_rating = 100.0 * oppg / pace if pace > 0 else 100.0
        return {
            "ppg": ppg, "oppg": oppg, "pace": pace, "off_rating": off_rating, "def_rating": def_rating,
            "efg_pct": LEAGUE_AVG_EFG, "tov_pct": LEAGUE_AVG_TOV, "orb_pct": LEAGUE_AVG_ORB, "ft_rate": LEAGUE_AVG_FT_RATE,
            "n_games": len(last5), "_from_scores_only": True, "_data_source": "Qualité Médium (Score Only)",
        }
    return None


def fetch_team_games(league_id: int, season: str, team_id: int) -> List[dict]:
    data, err = _api_get("games", {"league": league_id, "season": season, "team": team_id})
    if err or not data:
        return []
    resp = data.get("response")
    if not isinstance(resp, list):
        return []
    out: List[dict] = []
    for g in resp:
        teams_g = g.get("teams") or {}
        home = teams_g.get("home") or {}
        away = teams_g.get("away") or {}
        scores = g.get("scores") or {}
        pts_h = _safe_int(scores.get("home"))
        pts_a = _safe_int(scores.get("away"))
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        if home_id == team_id:
            out.append({"date": g.get("date"), "pts_for": pts_h, "pts_against": pts_a})
        elif away_id == team_id:
            out.append({"date": g.get("date"), "pts_for": pts_a, "pts_against": pts_h})
    out.sort(key=lambda x: x.get("date") or "", reverse=True)
    return out[:LAST_N_GAMES]


def fetch_team_games_with_opponents(league_id: int, season: str, team_id: int, n: int = LAST_N_GAMES) -> List[dict]:
    """V23 — Derniers n matchs avec opponent_id pour Strength of Schedule."""
    data, err = _api_get("games", {"league": league_id, "season": season, "team": team_id})
    if err or not data:
        return []
    resp = data.get("response")
    if not isinstance(resp, list):
        return []
    out: List[dict] = []
    for g in resp:
        teams_g = g.get("teams") or {}
        home = teams_g.get("home") or {}
        away = teams_g.get("away") or {}
        scores = g.get("scores") or {}
        pts_h = _safe_int(scores.get("home"))
        pts_a = _safe_int(scores.get("away"))
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        if home_id == team_id:
            out.append({"date": g.get("date"), "pts_for": pts_h, "pts_against": pts_a, "opponent_id": away_id})
        elif away_id == team_id:
            out.append({"date": g.get("date"), "pts_for": pts_a, "pts_against": pts_h, "opponent_id": home_id})
    out.sort(key=lambda x: x.get("date") or "", reverse=True)
    return out[:n]


def fetch_team_games_full(league_id: int, season: str, team_id: int, max_games: int = 20) -> List[dict]:
    """Game logs avec game_id pour récupérer les stats par match (raw box scores)."""
    data, err = _api_get("games", {"league": league_id, "season": season, "team": team_id})
    if err or not data:
        return []
    resp = data.get("response")
    if not isinstance(resp, list):
        return []
    out: List[dict] = []
    for g in resp:
        gid = g.get("id")
        teams_g = g.get("teams") or {}
        home = teams_g.get("home") or {}
        away = teams_g.get("away") or {}
        scores = g.get("scores") or {}
        sh = scores.get("home")
        sa = scores.get("away")
        pts_h = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
        pts_a = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        if home_id == team_id:
            out.append({"game_id": gid, "date": g.get("date"), "pts_for": pts_h, "pts_against": pts_a})
        elif away_id == team_id:
            out.append({"game_id": gid, "date": g.get("date"), "pts_for": pts_a, "pts_against": pts_h})
    out.sort(key=lambda x: x.get("date") or "", reverse=True)
    return out[:max_games]


def fetch_game_statistics_teams(game_id: int) -> List[dict]:
    """Box score par équipe pour un match. API: games/statistics/teams avec id=game_id."""
    data, err = _api_get("games/statistics/teams", {"id": game_id})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def fetch_game_statistics_teams_batch(game_ids: List[int], max_per_call: int = 20) -> List[dict]:
    """
    Box scores pour plusieurs matchs en un appel (doc API : param ids, max 20).
    Utile pour EuroLeague / ligues où un seul appel par match renvoie parfois vide.
    Retourne une liste plate d'items (chaque item : game.id, team.id ou team_id + stats).
    """
    if not game_ids:
        return []
    ids_str = "-".join(str(gid) for gid in game_ids[:max_per_call])
    data, err = _api_get("games/statistics/teams", {"ids": ids_str})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def _safe_get(obj: Any, *keys: Any, default: int = 0) -> int:
    """Accès sécurisé imbriqué : obj[k1][k2]... avec fallback default."""
    val = obj
    for k in keys:
        if val is None:
            return default
        val = val.get(k) if isinstance(val, dict) else None
    return _safe_int(val) if val is not None else default


def brute_force_extract(data: Any, keys_to_try: List[str]) -> Optional[int]:
    """
    Parsing Guerrier (Sniffer) : cherche récursivement total/all/made/attempted/etc.
    n'importe où dans le sous-dictionnaire. Gère les strings sales (ex: "45%" → 45).
    Retourne la première valeur entière trouvée, sinon None.
    """
    if data is None:
        return None
    for key in keys_to_try:
        if not isinstance(data, dict):
            break
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return max(0, int(val))
        if isinstance(val, str):
            out = _safe_int(val, -1)
            return out if out >= 0 else None
        if isinstance(val, dict):
            inner = brute_force_extract(val, ["total", "all", "made", "attempted", "attempts", "offensive", "offence"])
            if inner is not None:
                return inner
    if isinstance(data, dict) and "for" in data:
        inner = brute_force_extract(data["for"], keys_to_try)
        if inner is not None:
            return inner
    # Descente récursive dans tous les sous-dict
    if isinstance(data, dict):
        for _k, v in data.items():
            if isinstance(v, dict):
                res = brute_force_extract(v, keys_to_try)
                if res is not None:
                    return res
    return None


def _extract_raw_stats_from_team_game(team_stats_item: dict) -> Optional[dict]:
    """
    Extraction Guerrière (Sniffer) : brute_force_extract sur chaque section.
    Ne dépend pas de ['for']['total']. Cherche total/all/made n'importe où dans le sous-dict.
    Gère strings sales (ex: "45%"). Retourne None si stats invalides.
    """
    raw = team_stats_item or {}
    game_id = raw.get("game_id") or (raw.get("game") or {}).get("id") or "?"

    def _log_missing(section_name: str) -> None:
        print(f"[DEBUG] Section manquante pour Match ID {game_id} : {section_name}")

    fg = raw.get("field_goals") or raw.get("field_goals_goals")
    reb = raw.get("rebounds") or raw.get("rebounds_goals")
    th = raw.get("three_points") or raw.get("threepoint_goals") or raw.get("three_points_goals")
    ft = raw.get("free_throws") or raw.get("freethrows_goals") or raw.get("free_throws_goals")
    tov = raw.get("turnovers") or raw.get("turnovers_goals")
    pts = raw.get("points") or raw.get("points_goals")

    fga = brute_force_extract(fg, ["attempted", "attempts"]) if isinstance(fg, dict) else None
    fgm = brute_force_extract(fg, ["made", "total", "all"]) if isinstance(fg, dict) else None
    fta = brute_force_extract(ft, ["attempted", "attempts"]) if isinstance(ft, dict) else None
    ftm = brute_force_extract(ft, ["made", "total", "all"]) if isinstance(ft, dict) else None
    thm = brute_force_extract(th, ["made", "total", "all"]) if isinstance(th, dict) else None
    orb = brute_force_extract(reb, ["offensive", "offence"]) if isinstance(reb, dict) else None
    trb = brute_force_extract(reb, ["total", "all"]) if isinstance(reb, dict) else None
    tov_val = brute_force_extract(tov, ["total", "all"]) if isinstance(tov, dict) else _safe_int(tov) if tov is not None else 0
    pts_for = brute_force_extract(pts, ["total", "all"]) if isinstance(pts, dict) else None

    if pts_for is None and (fgm or thm or ftm):
        pts_for = 2 * (fgm or 0) + (thm or 0) + (ftm or 0)

    fga = fga or 0
    fgm = fgm or 0
    fta = fta or 0
    ftm = ftm or 0
    thm = thm or 0
    orb = orb or 0
    trb = trb or 0
    tov_val = tov_val or 0
    pts_for = pts_for or 0

    if fga <= 0 and fgm <= 0 and fta <= 0:
        top_keys = list(raw.keys()) if isinstance(raw, dict) else []
        _log_missing("field_goals")
        try:
            st.error(f"Box Score vide Match ID {game_id}. Clés: {top_keys[:20]}")
        except Exception:
            pass
        return None
    if fga <= 0 and fta <= 0:
        return None

    poss = fga + int(0.44 * fta) + tov_val - orb
    if poss <= 0:
        poss = max(1, fga + fta)
    return {
        "FGA": fga, "FGM": fgm, "3PM": thm, "FTA": fta, "FTM": ftm,
        "ORB": orb, "TRB": trb, "TOV": tov_val, "Possessions": poss, "pts_for": pts_for,
    }


def calculate_four_factors_manual(games_list: List[dict]) -> Optional[dict]:
    """
    Agrège les stats brutes : n'accepte un match QUE s'il a le score de l'équipe ET le score de l'adversaire.
    Interdiction stricte : si pts_against n'est pas récupéré (objet game['scores']), le calcul est interdit —
    un oppg à zéro ou par défaut fausse toute la courbe de Gauss. Si < 3 matchs complets → None → _incomplete.
    """
    if not games_list:
        pace = PACE_DEFAULT_LIGUE
        return {"ppg": 75.0, "oppg": 75.0, "pace": pace, "off_rating": 100.0 * 75.0 / pace, "def_rating": 100.0 * 75.0 / pace, "efg_pct": 0.5, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25}
    # Vérification croisée : score adversaire obligatoire (doit venir de l'objet game['scores'] dans le caller)
    games_complete = [g for g in games_list if g.get("pts_for") is not None and g.get("pts_against") is not None]
    if len(games_complete) < MIN_GAME_LOGS_EXPLOITABLE:
        return None
    # Interdire tout calcul si oppg serait faux (matchs sans pts_against exclus ci-dessus)
    pts_for_list = [g.get("pts_for") for g in games_complete]
    pts_against_list = [g.get("pts_against", 0) for g in games_complete]
    ppg = sum(pts_for_list) / len(pts_for_list)
    oppg = sum(pts_against_list) / len(pts_against_list)
    n = len(games_list)
    tot_fga = sum(g.get("FGA", 0) for g in games_list)
    tot_fgm = sum(g.get("FGM", 0) for g in games_list)
    tot_3pm = sum(g.get("3PM", 0) for g in games_list)
    tot_fta = sum(g.get("FTA", 0) for g in games_list)
    tot_orb = sum(g.get("ORB", 0) for g in games_list)
    tot_tov = sum(g.get("TOV", 0) for g in games_list)
    tot_poss = sum(g.get("Possessions", 0) for g in games_list)
    pace_avg = tot_poss / n if n and tot_poss > 0 else PACE_DEFAULT_LIGUE
    pace = float(np.clip(pace_avg, PACE_MIN, PACE_MAX))
    efg_pct = (tot_fgm + 0.5 * tot_3pm) / tot_fga if tot_fga > 0 else 0.5
    tov_pct = tot_tov / tot_poss if tot_poss > 0 else 0.15
    orb_pct = tot_orb / (tot_orb + 30.0 * n) if (tot_orb + 30.0 * n) > 0 else 0.25
    ft_rate = tot_fta / tot_fga if tot_fga > 0 else 0.25
    off_rating = 100.0 * ppg / pace if pace > 0 else 100.0
    def_rating = 100.0 * oppg / pace if pace > 0 else 100.0
    return {
        "ppg": ppg, "oppg": oppg, "pace": pace, "off_rating": off_rating, "def_rating": def_rating,
        "efg_pct": efg_pct, "tov_pct": tov_pct, "orb_pct": orb_pct, "ft_rate": ft_rate,
    }


def _date_strictly_before(game_date: str, target_date: str) -> bool:
    """Compare uniquement la partie date (YYYY-MM-DD) pour éviter TZ/format. Garde si match AVANT target."""
    g = (game_date or "")[:10]
    t = (target_date or "")[:10]
    return g < t and len(g) == 10 and len(t) == 10


def _build_degraded_stats_from_scores(last_n: List[dict], league_id: Optional[int] = None, season: Optional[str] = None, opponent_pace: Optional[float] = None) -> dict:
    """
    Construit un objet stats dégradé (Score Only) : ppg, oppg depuis scores, facteurs = moyennes ligue.
    Pace : estimation ligue (10 derniers matchs) si league_id/season fournis, sinon pace adverse si dispo, sinon défaut.
    """
    pts_for_list = [g.get("pts_for") for g in last_n if g.get("pts_for") is not None]
    pts_against_list = [g.get("pts_against") for g in last_n if g.get("pts_against") is not None]
    n = len(pts_for_list) or 1
    ppg = sum(pts_for_list) / n if pts_for_list else 75.0
    oppg = sum(pts_against_list) / n if pts_against_list else 75.0
    pace = PACE_DEFAULT_LIGUE
    if league_id and season:
        pace = _estimate_pace_from_league_history(league_id, season) or pace
    if (pace <= 0 or pace == PACE_DEFAULT_LIGUE) and opponent_pace is not None and opponent_pace > 0:
        pace = opponent_pace
    off_rating = 100.0 * ppg / pace if pace > 0 else 100.0
    def_rating = 100.0 * oppg / pace if pace > 0 else 100.0
    out = {
        "ppg": ppg, "oppg": oppg, "pace": pace, "off_rating": off_rating, "def_rating": def_rating,
        "efg_pct": 0.50, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25,
        "quality": "LOW", "_data_source": "Qualité Médium (Score Only)",
    }
    return out


def get_clean_stats_at_date(league_id: int, season: str, team_id: int, target_date: str) -> Optional[dict]:
    """
    TIME MACHINE : stats connues AVANT target_date (pas de lookahead).
    - Multi-saison : commence par la saison demandée ; si < 5 matchs avant target_date, fusionne avec saison(s) précédente(s).
    - Filtre date : YYYY-MM-DD uniquement.
    - Box Score : tente stats détaillées par match.
    - Fallback "Score Only" : si Box Score vide pour un match, utilise quand même pts_for/pts_against du game ; facteurs = moyennes ligue (0.50, 0.25, 0.15) ; quality='LOW'.
    - Ne renvoie None que si aucun match ni /statistics dispo.
    """
    target_norm = (target_date or "")[:10]
    used_season = season
    merged_before: List[dict] = []
    seen_dates: set = set()

    for try_season in [season] + [s for s in TIME_MACHINE_SEASONS if s != season]:
        all_games = fetch_team_games_full(league_id, try_season, team_id, max_games=50)
        before_date = [g for g in all_games if _date_strictly_before(g.get("date") or "", target_date)]
        for g in before_date:
            d = (g.get("date") or "")[:10]
            if d and d not in seen_dates:
                seen_dates.add(d)
                merged_before.append(g)
        merged_before.sort(key=lambda x: x.get("date") or "", reverse=True)
        if len(merged_before) >= TIME_MACHINE_MIN_GAMES:
            used_season = try_season
            break
        used_season = try_season

    last_n = merged_before[:TIME_MACHINE_LAST_N]

    raw_list: List[dict] = []
    game_ids = [g.get("game_id") for g in last_n if g.get("game_id")]

    # 1) Essai appel groupé (ids=id1-id2-...) — meilleur taux de réponse EuroLeague / certaines ligues
    if len(game_ids) >= 2:
        batch_items = fetch_game_statistics_teams_batch(game_ids)
        time.sleep(0.2)
        # Map (game_id, team_id) -> team_stats_item (chaque item a game.id ou game_id, team.id ou team_id)
        batch_by_game_team: Dict[Tuple[int, int], dict] = {}
        for item in batch_items:
            gid = item.get("game_id") or (item.get("game") or {}).get("id")
            tid = item.get("team_id") or (item.get("team") or {}).get("id")
            if gid is not None and tid is not None:
                batch_by_game_team[(int(gid), int(tid))] = item
        for g in last_n:
            gid = g.get("game_id")
            if not gid:
                continue
            team_entry = batch_by_game_team.get((int(gid), int(team_id)))
            if not team_entry:
                continue
            raw = _extract_raw_stats_from_team_game(team_entry)
            if raw is not None:
                raw["pts_against"] = g.get("pts_against") or 0
                raw_list.append(raw)

    # 2) Si pas assez de box scores (ex. EuroLeague raw_list=1), compléter avec appels un par un
    if len(raw_list) < 2:
        raw_list = []
        for g in last_n:
            gid = g.get("game_id")
            if not gid:
                continue
            team_stats_list = fetch_game_statistics_teams(gid)
            time.sleep(0.2)
            team_entry = None
            for item in team_stats_list:
                tid = item.get("team_id")
                if tid is None and isinstance(item.get("team"), dict):
                    tid = item.get("team", {}).get("id")
                if tid is not None and int(tid) == team_id:
                    team_entry = item
                    break
            if not team_entry:
                continue
            raw = _extract_raw_stats_from_team_game(team_entry)
            if raw is not None:
                raw["pts_against"] = g.get("pts_against") or 0
                raw_list.append(raw)

    if len(raw_list) >= 2:
        out = calculate_four_factors_manual(raw_list)
        if out is not None:
            out["quality"] = "HIGH"
            season_label = "2025" if used_season == "2025" else "2024"
            msg = f"DEBUG Time Machine: team_id={team_id} saison={used_season} ({season_label}) target={target_norm} | matchs avant date={len(last_n)} | raw_list={len(raw_list)} → Box Score OK"
            TIME_MACHINE_DEBUG_LOGS.append(msg)
            print(msg)
            return out

    if last_n:
        out = _build_degraded_stats_from_scores(last_n, league_id=league_id, season=used_season)
        season_label = "2025" if used_season == "2025" else "2024"
        msg = f"DEBUG Time Machine: team_id={team_id} saison={used_season} ({season_label}) target={target_norm} | raw_list={len(raw_list)} → Score Only (quality=LOW)"
        TIME_MACHINE_DEBUG_LOGS.append(msg)
        print(msg)
        return out

    raw_season = fetch_team_statistics(league_id, used_season, team_id)
    if raw_season:
        out = _team_stats_from_response(raw_season)
        out["_from_season_fallback"] = True
        out["quality"] = "LOW"
        season_label = "2025" if used_season == "2025" else "2024"
        msg = f"DEBUG Time Machine: team_id={team_id} saison={used_season} ({season_label}) target={target_norm} | raw_list={len(raw_list)} → Fallback /statistics saison"
        TIME_MACHINE_DEBUG_LOGS.append(msg)
        print(msg)
        return out

    season_label = "2025" if used_season == "2025" else "2024"
    msg = f"DEBUG Time Machine: team_id={team_id} saison={used_season} ({season_label}) target={target_norm} | matchs avant date={len(last_n)} | raw_list={len(raw_list)} → Pas de Data"
    TIME_MACHINE_DEBUG_LOGS.append(msg)
    print(msg)
    return None


def fetch_odds(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float]]:
    data, err = _api_get("odds", {"game": game_id, "league": league_id, "season": season})
    if err or not data:
        return None, None
    resp = data.get("response")
    if not isinstance(resp, list) or len(resp) == 0:
        return None, None
    item = resp[0]
    for bm in item.get("bookmakers") or []:
        if bm.get("id") not in BOOKMAKER_IDS:
            continue
        for bet in bm.get("bets") or []:
            if int(bet.get("id", 0)) != BET_HOME_AWAY_ID:
                continue
            odd_home = odd_away = None
            for v in bet.get("values") or []:
                val = (v.get("value") or "").strip().lower()
                try:
                    odd_f = float(v.get("odd") or 0)
                    if val == "home":
                        odd_home = odd_f
                    elif val == "away":
                        odd_away = odd_f
                except (TypeError, ValueError):
                    pass
            if odd_home is not None and odd_away is not None:
                return odd_home, odd_away
    for bm in item.get("bookmakers") or []:
        for bet in bm.get("bets") or []:
            if int(bet.get("id", 0)) != BET_HOME_AWAY_ID:
                continue
            odd_home = odd_away = None
            for v in bet.get("values") or []:
                val = (v.get("value") or "").strip().lower()
                try:
                    odd_f = float(v.get("odd") or 0)
                    if val == "home":
                        odd_home = odd_f
                    elif val == "away":
                        odd_away = odd_f
                except (TypeError, ValueError):
                    pass
            if odd_home is not None and odd_away is not None:
                return odd_home, odd_away
    return None, None


def fetch_odds_totals(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Récupère la ligne principale et les cotes EXACTES Over/Under.
    Priorité: Bookmakers Betclic (17), Unibet/10Bet (7), Bwin (1).
    Cherche le label "Over/Under" ou "Over/Under Games" (ID 4 ou 5).
    Retourne (line, cote_over, cote_under).
    """
    data, err = _api_get("odds", {"game": game_id, "league": league_id, "season": season})
    if err or not data:
        return None, None, None
    resp = data.get("response")
    if not isinstance(resp, list) or len(resp) == 0:
        return None, None, None
    item = resp[0]
    for bm in item.get("bookmakers") or []:
        bm_id = bm.get("id")
        if bm_id not in BOOKMAKER_IDS_OU:
            continue
        for bet in bm.get("bets") or []:
            bid = int(bet.get("id", 0))
            name = (bet.get("name") or "").lower()
            if bid not in BET_OVER_UNDER_IDS and "over" not in name and "under" not in name and "total" not in name and "points" not in name:
                continue
            if "1st" in name or "half" in name or "1st half" in name:
                continue
            values = bet.get("values") or []
            line = odd_over = odd_under = None
            for v in values:
                val_raw = (v.get("value") or "").strip()
                val = val_raw.upper()
                try:
                    odd_f = float(v.get("odd") or 0) if v.get("odd") else None
                    if "OVER" in val or val == "O":
                        odd_over = odd_f
                    elif "UNDER" in val or val == "U":
                        odd_under = odd_f
                    num_match = re.search(r"[\d]+[.,]?\d*", val_raw)
                    if num_match and line is None:
                        line = float(num_match.group(0).replace(",", "."))
                except (TypeError, ValueError):
                    pass
            if (odd_over is not None or odd_under is not None) and (line is not None):
                return line, odd_over, odd_under
            if (odd_over or odd_under) and line is None and values:
                first_val = str(values[0].get("value") or "")
                num_match = re.search(r"[\d]+[.,]?\d*", first_val)
                if num_match:
                    line = float(num_match.group(0).replace(",", "."))
                    return line, odd_over, odd_under
    for bm in item.get("bookmakers") or []:
        for bet in bm.get("bets") or []:
            bid = int(bet.get("id", 0))
            name = (bet.get("name") or "").lower()
            if bid not in BET_OVER_UNDER_IDS and "over" not in name and "under" not in name:
                continue
            values = bet.get("values") or []
            line = odd_over = odd_under = None
            for v in values:
                val_raw = (v.get("value") or "").strip()
                val = val_raw.upper()
                try:
                    odd_f = float(v.get("odd") or 0) if v.get("odd") else None
                    if "OVER" in val or val == "O":
                        odd_over = odd_f
                    elif "UNDER" in val or val == "U":
                        odd_under = odd_f
                    num_match = re.search(r"[\d]+[.,]?\d*", val_raw)
                    if num_match and line is None:
                        line = float(num_match.group(0).replace(",", "."))
                except (TypeError, ValueError):
                    pass
            if (odd_over or odd_under) and line is not None:
                return line, odd_over, odd_under
    return None, None, None


def fetch_odds_spread(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    V25 — Tente de récupérer le handicap (spread) du bookmaker.
    Cherche les paris avec 'handicap' ou 'spread' dans le nom.
    Retourne (spread_home, cote_home, cote_away) — spread = écart côté Home (ex: -5.5 = Home -5.5).
    """
    data, err = _api_get("odds", {"game": game_id, "league": league_id, "season": season})
    if err or not data:
        return None, None, None
    resp = data.get("response")
    if not isinstance(resp, list) or len(resp) == 0:
        return None, None, None
    item = resp[0]
    for bm in item.get("bookmakers") or []:
        if bm.get("id") not in BOOKMAKER_IDS:
            continue
        for bet in bm.get("bets") or []:
            name = (bet.get("name") or "").lower()
            if "handicap" not in name and "spread" not in name and "écart" not in name and "hcap" not in name:
                continue
            if "1st" in name or "half" in name or "mi-temps" in name:
                continue
            values = bet.get("values") or []
            spread_val = odd_home = odd_away = None
            for v in values:
                val_raw = (v.get("value") or "").strip()
                try:
                    odd_f = float(v.get("odd") or 0) if v.get("odd") else None
                    num_match = re.search(r"-?[\d]+[.,]?\d*", val_raw)
                    if num_match and spread_val is None:
                        spread_val = float(num_match.group(0).replace(",", "."))
                    if "home" in val_raw.lower() or val_raw.startswith("-"):
                        odd_home = odd_f
                    elif "away" in val_raw.lower() or val_raw.startswith("+"):
                        odd_away = odd_f
                except (TypeError, ValueError):
                    pass
            if spread_val is not None and (odd_home or odd_away):
                return spread_val, odd_home, odd_away
    return None, None, None


def calculate_fair_spread(proj_home: float, proj_away: float, home_name: str, away_name: str) -> str:
    """
    V25 — Calcule le Fair Spread (écart prédit).
    Ex: Home 85, Away 80 → "Home -5.0" (Home favori de 5 pts).
    """
    spread = proj_home - proj_away
    if abs(spread) < 0.05:
        return "Égalité"
    if spread > 0:
        return f"{home_name} -{spread:.1f}"
    return f"{away_name} -{abs(spread):.1f}"


# ==============================================================================
# HELPERS
# ==============================================================================


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, float):
        return max(0, int(val))
    if isinstance(val, dict):
        return _safe_int(val.get("total") or val.get("all"), default)
    if isinstance(val, str):
        # Gestion strings sales : "45%", "12", " 7 "
        s = val.strip().replace("%", "").replace(",", ".")
        try:
            return max(0, int(float(s)))
        except (TypeError, ValueError):
            return default
    return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        return _safe_float(val.get("average") or val.get("total"), default)
    if isinstance(val, str):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    return default


def _possessions_per_game(raw: dict, played: int) -> Optional[float]:
    """
    Formule stricte FIBA : Poss = FGA + 0.44*FTA + TOV - ORB (par match).
    Retourne None si données indisponibles.
    """
    if played <= 0:
        return None
    try:
        fg = raw.get("field_goals") or {}
        fg_for = fg.get("for") or {}
        fga_tot = _safe_int(fg_for.get("attempted"))
        ft = raw.get("free_throws") or {}
        ft_for = ft.get("for") or {}
        fta_tot = _safe_int(ft_for.get("attempted"))
        tov = raw.get("turnovers") or {}
        tov_for = tov.get("for") or {}
        tov_tot = _safe_int(tov_for.get("total"))
        reb = raw.get("rebounds") or {}
        reb_for = reb.get("for") or {}
        orb_tot = _safe_int(reb_for.get("offensive"))
        if fga_tot <= 0 and fta_tot <= 0:
            return None
        poss_tot = fga_tot + 0.44 * fta_tot + tov_tot - orb_tot
        if poss_tot <= 0:
            return None
        return poss_tot / played
    except Exception:
        return None


def _team_stats_from_response(raw: dict) -> dict:
    games = raw.get("games") or {}
    played = max(1, _safe_int(games.get("played")) or _safe_int(games.get("all")) or 1)
    points = raw.get("points") or {}
    for_pts = points.get("for") or {}
    against_pts = points.get("against") or {}
    ppg = _safe_float(for_pts.get("average")) or (_safe_int(for_pts.get("total")) / played if played else 75.0)
    oppg = _safe_float(against_pts.get("average")) or (_safe_int(against_pts.get("total")) / played if played else 75.0)
    if ppg <= 0:
        ppg = 75.0
    if oppg <= 0:
        oppg = 75.0
    poss_pg = _possessions_per_game(raw, played)
    pace = float(np.clip(poss_pg if poss_pg is not None else PACE_DEFAULT_LIGUE, PACE_MIN, PACE_MAX))
    off_rating = 100.0 * ppg / pace if pace > 0 else 100.0
    def_rating = 100.0 * oppg / pace if pace > 0 else 100.0
    out: dict = {
        "ppg": ppg, "oppg": oppg, "pace": pace, "off_rating": off_rating, "def_rating": def_rating,
        "efg_pct": 0.5, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25,
    }
    try:
        fg = raw.get("field_goals") or {}
        fg_for = fg.get("for") or {}
        fgm = _safe_float(fg_for.get("made")) or _safe_int(fg_for.get("total")) / max(played, 1)
        fga = _safe_float(fg_for.get("attempted")) or max(1, fgm * 2)
        threes = raw.get("three_points") or {}
        th_for = threes.get("for") or {}
        thm = _safe_float(th_for.get("made")) or 0.0
        ft = raw.get("free_throws") or {}
        ft_for = ft.get("for") or {}
        ftm = _safe_float(ft_for.get("made")) or 0.0
        fta = _safe_float(ft_for.get("attempted")) or max(0.1, ftm)
        reb = raw.get("rebounds") or {}
        reb_for = reb.get("for") or {}
        orb = _safe_float(reb_for.get("offensive")) or _safe_int(reb_for.get("total")) / 4.0
        tov = raw.get("turnovers") or {}
        tov_for = tov.get("for") or {}
        tov_val = _safe_float(tov_for.get("total")) or 0.0
        if fga and fga > 0:
            out["efg_pct"] = (fgm + 0.5 * thm) / fga
        out["tov_pct"] = tov_val / max(played * 70, 1) if tov_val else 0.15
        out["orb_pct"] = orb / max(played * 10, 1) if orb else 0.25
        out["ft_rate"] = fta / max(fga, 1) if fga else 0.25
    except Exception:
        pass
    return out


def _stats_from_game_logs(league_id: int, season: str, team_id: int, max_games: int = None) -> Optional[dict]:
    """
    V24 Deep Data : calcule ppg, oppg, pace et Four Factors à partir des Box Scores.
    Ordre saisons (Fix Pro B) : ['2025-2026', '2025', '2024-2025', '2024']. Stop si matchs + stats non-vides.
    Fallback Joueurs (Fix ABA/Dubai) : si team_stats vide ou FGA/PTS=0, reconstruction via stats joueurs.
    """
    max_g = max_games or GAME_LOGS_MAX
    seasons_order: List[str] = list(SEASONS_TO_TRY)

    for try_season in seasons_order:
        all_games = fetch_team_games_full(league_id, try_season, team_id, max_games=max_g)
        game_ids = [g.get("game_id") for g in all_games if g.get("game_id")]
        if not game_ids:
            continue

        batch_items: List[dict] = []
        for i in range(0, len(game_ids), 20):
            chunk = game_ids[i : i + 20]
            batch_items.extend(fetch_game_statistics_teams_batch(chunk, max_per_call=20))

        batch_by_game_team: Dict[Tuple[int, int], dict] = {}
        for item in batch_items:
            gid = item.get("game_id") or (item.get("game") or {}).get("id")
            tid = item.get("team_id") or (item.get("team") or {}).get("id")
            if gid is not None and tid is not None:
                batch_by_game_team[(int(gid), int(tid))] = item

        raw_list: List[dict] = []
        for g in all_games:
            gid = g.get("game_id")
            if not gid:
                continue
            team_entry = batch_by_game_team.get((int(gid), int(team_id)))
            raw: Optional[dict] = None
            if team_entry:
                raw = _extract_raw_stats_from_team_game(team_entry)
            # Arme nucléaire : si team stats vide ou FGA/PTS à zéro → reconstruction via joueurs
            if raw is None or ((raw.get("FGA") or 0) == 0 and (raw.get("pts_for") or 0) == 0):
                raw = aggregate_team_stats_from_players(gid, team_id, try_season)
            if raw is not None:
                raw["pts_against"] = g.get("pts_against") if g.get("pts_against") is not None else 0
                raw_list.append(raw)

        if len(raw_list) < MIN_GAME_LOGS_EXPLOITABLE:
            continue
        out = calculate_four_factors_manual(raw_list)
        if out is None or out.get("efg_pct", 0) <= 0 or out.get("efg_pct", 0) >= 0.95:
            continue
        return out
    return None


def _is_aberrant_factor(key: str, value: float) -> bool:
    """True si le facteur est à 0 ou aberrant (à afficher N/A)."""
    if value is None:
        return True
    if key == "efg_pct":
        return value <= 0 or value >= 0.95
    if key == "tov_pct":
        return value < 0 or value > 0.5
    if key == "orb_pct":
        return value < 0 or value > 0.6
    if key == "ft_rate":
        return value < 0 or value > 1.0
    return False


@st.cache_data(ttl=180)
def get_team_stats_weighted(league_id: int, season: str, team_id: int, use_game_logs: bool = True) -> dict:
    """
    V24 Deep Data : D'ABORD agrégation manuelle Box Scores (_stats_from_game_logs, batching).
    Si et seulement si le calcul échoue (< 3 matchs exploitables), fallback fetch_team_statistics.
    use_game_logs=False : force l'API /statistics (roue de secours uniquement).
    Cache 3 min.
    """
    manual = _stats_from_game_logs(league_id, season, team_id) if use_game_logs else None
    if manual is not None:
        last5 = fetch_team_games(league_id, season, team_id)
        last5 = [x for x in last5 if (x.get("pts_for") or 0) > 0 or (x.get("pts_against") or 0) > 0][:LAST_N_GAMES]
        if len(last5) >= 2:
            ppg_l5 = sum(x.get("pts_for", 0) for x in last5) / len(last5)
            oppg_l5 = sum(x.get("pts_against", 0) for x in last5) / len(last5)
            ppg = WEIGHT_SEASON * manual["ppg"] + WEIGHT_LAST5 * ppg_l5
            oppg = WEIGHT_SEASON * manual["oppg"] + WEIGHT_LAST5 * oppg_l5
        else:
            ppg, oppg = manual["ppg"], manual["oppg"]
        if ppg <= 0:
            ppg = 75.0
        if oppg <= 0:
            oppg = 75.0
        pace = manual.get("pace", PACE_DEFAULT_LIGUE)
        off_rating = 100.0 * ppg / pace if pace > 0 else 100.0
        def_rating = 100.0 * oppg / pace if pace > 0 else 100.0
        n_games = len(last5) if last5 else 5
        return {
            "ppg": ppg, "oppg": oppg, "pace": pace, "off_rating": off_rating, "def_rating": def_rating,
            "efg_pct": manual["efg_pct"], "tov_pct": manual["tov_pct"], "orb_pct": manual["orb_pct"], "ft_rate": manual["ft_rate"],
            "n_games": n_games, "_data_source": "Qualité Haute",
        }

    # V24 Fallback : API /statistics (season stats) — essayer plusieurs saisons (cadors, Monaco, ligues 2024-2025)
    raw: Optional[dict] = None
    used_season = season
    for try_season in [season] + [s for s in SEASONS_TO_TRY if s != season]:
        raw = fetch_team_statistics(league_id, try_season, team_id)
        if raw:
            games = raw.get("games") or {}
            played = max(0, _safe_int(games.get("played")) or _safe_int(games.get("all")))
            if played >= 2:
                used_season = try_season
                break
        raw = None
    played = 0
    if raw:
        games = raw.get("games") or {}
        played = max(0, _safe_int(games.get("played")) or _safe_int(games.get("all")))

    if raw is None or played < 2:
        # V24 Dernier recours : stats depuis les scores des derniers matchs (pas de box score)
        score_only = _stats_from_scores_only(league_id, season, team_id)
        if score_only is not None:
            return score_only
        # Box Scores ET API /statistics absents → marquer INCOMPLETE (pas de proba bidon)
        pace = PACE_DEFAULT_LIGUE
        return {
            "ppg": 75.0, "oppg": 75.0, "pace": pace, "off_rating": 100.0 * 75.0 / pace, "def_rating": 100.0 * 75.0 / pace,
            "efg_pct": 0.5, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25, "n_games": 0,
            "_incomplete": True, "_data_source": "FLUX DATA INTERROMPU",
        }

    season_stats = _team_stats_from_response(raw)
    pace = float(np.clip(season_stats.get("pace", PACE_DEFAULT_LIGUE), PACE_MIN, PACE_MAX))
    last5 = fetch_team_games(league_id, used_season, team_id)
    last5 = [x for x in last5 if (x.get("pts_for") or 0) > 0 or (x.get("pts_against") or 0) > 0]
    if len(last5) < 2:
        ppg, oppg = season_stats["ppg"], season_stats["oppg"]
    else:
        ppg_l5 = sum(x.get("pts_for", 0) for x in last5) / len(last5)
        oppg_l5 = sum(x.get("pts_against", 0) for x in last5) / len(last5)
        ppg = WEIGHT_SEASON * season_stats["ppg"] + WEIGHT_LAST5 * ppg_l5
        oppg = WEIGHT_SEASON * season_stats["oppg"] + WEIGHT_LAST5 * oppg_l5
    if ppg <= 0:
        ppg = 75.0
    if oppg <= 0:
        oppg = 75.0
    off_rating = 100.0 * ppg / pace if pace > 0 else 100.0
    def_rating = 100.0 * oppg / pace if pace > 0 else 100.0
    out = {"ppg": ppg, "oppg": oppg, "pace": pace, "off_rating": off_rating, "def_rating": def_rating}
    for k in ("efg_pct", "tov_pct", "orb_pct", "ft_rate"):
        out[k] = season_stats.get(k, 0.25 if k == "orb_pct" else 0.5 if k == "efg_pct" else 0.15 if k == "tov_pct" else 0.25)
    out["n_games"] = played
    out["_data_source"] = "Qualité Médium (API)"
    return out


def get_pace_and_off_rating(league_id: int, season: str, team_id: int) -> Tuple[float, float]:
    """Retourne (pace en poss/match, off_rating pts/100 poss)."""
    stats = get_team_stats_weighted(league_id, season, team_id)
    return stats.get("pace", PACE_DEFAULT_LIGUE), stats.get("off_rating", 100.0)


def reality_check_mae(league_id: int, season: str, team_id: int) -> Tuple[float, str, str]:
    """
    Backtest live : 3 derniers matchs terminés, compare score réel vs projection modèle.
    Retourne (MAE, label_court, label_affichage) :
    - 🟢 Err < 6 pts : "Modèle Calibré"
    - 🟡 Err 6–12 pts : "Moyen"
    - 🔴 Err > 12 pts : "Volatile / Imprévisible"
    """
    stats = get_team_stats_weighted(league_id, season, team_id)
    pace = stats.get("pace", PACE_DEFAULT_LIGUE)
    off_rating = stats.get("off_rating", 100.0)
    pred_ppg = (pace / 100.0) * off_rating
    games = fetch_team_games(league_id, season, team_id)
    games = [x for x in games if (x.get("pts_for") or 0) > 0 or (x.get("pts_against") or 0) > 0][:REALITY_N_GAMES]
    if len(games) < 1:
        return 0.0, "🟢", "Modèle Calibré"
    errors = [abs(pred_ppg - (x.get("pts_for") or 0)) for x in games]
    mae = sum(errors) / len(errors)
    if mae < FIABILITE_MAE_GREEN:
        return mae, "🟢", "Modèle Calibré"
    if mae <= FIABILITE_MAE_YELLOW:
        return mae, "🟡", "Moyen"
    return mae, "🔴", "Volatile / Imprévisible"


def get_team_form(league_id: int, season: str, team_id: int) -> str:
    """Retourne la forme des 5 derniers matchs : 'W-L-W-L-L'."""
    last5 = fetch_team_games(league_id, season, team_id)
    last5 = [x for x in last5 if (x.get("pts_for") or 0) > 0 or (x.get("pts_against") or 0) > 0][:LAST_N_GAMES]
    if not last5:
        return "-"
    letters: List[str] = []
    for g in last5:
        pf, pa = g.get("pts_for", 0) or 0, g.get("pts_against", 0) or 0
        letters.append("W" if pf > pa else "L")
    return "-".join(letters)


def build_tale_of_the_tape(
    league_id: int, season: str,
    home_id: int, away_id: int, home_name: str, away_name: str,
    sh: Optional[dict] = None, sa: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    Tableau Head-to-Head : Metric | Domicile | Extérieur | Analyse.
    V23 : Off. Rtg (brut) et Adj. Off Rtg (SOS) pour comparer stats brutes vs ajustées.
    """
    if sh is None:
        sh = get_team_stats_weighted(league_id, season, home_id)
    if sa is None:
        sa = get_team_stats_weighted(league_id, season, away_id)
    pace_h = round(sh.get("pace", 150.0), 1)
    pace_a = round(sa.get("pace", 150.0), 1)
    off_raw_h = round(sh.get("off_rating_raw") or sh.get("off_rating", 100.0), 1)
    off_raw_a = round(sa.get("off_rating_raw") or sa.get("off_rating", 100.0), 1)
    off_adj_h = round(sh.get("off_rating", 100.0), 1)
    off_adj_a = round(sa.get("off_rating", 100.0), 1)
    def_h = round(sh.get("oppg", 75.0), 1)
    def_a = round(sa.get("oppg", 75.0), 1)
    form_h = get_team_form(league_id, season, home_id)
    form_a = get_team_form(league_id, season, away_id)

    def _analyse_pace(ph: float, pa: float) -> str:
        avg = (ph + pa) / 2.0  # possessions FIBA ~70
        if avg < 69:
            return "Lent = Under"
        if avg > 74:
            return "Rapide = Over"
        return "Rythme moyen"

    def _analyse_off(oh: float, oa: float) -> str:
        if oh < 95 and oa < 95:
            return "Attaque faible"
        if oh >= 110 or oa >= 110:
            return "Attaque forte"
        return "Offensive moyenne"

    def _analyse_def(dh: float, da: float) -> str:
        if dh > 82 or da > 82:
            return "Défense faible"
        if dh < 72 and da < 72:
            return "Défense solide"
        return "Défense moyenne"

    def _analyse_form(fh: str, fa: str) -> str:
        w_h = fh.count("W") if isinstance(fh, str) else 0
        w_a = fa.count("W") if isinstance(fa, str) else 0
        if w_h >= 4 or w_a >= 4:
            return "Dynamique"
        if w_h <= 1 and w_a <= 1:
            return "En difficulté"
        return "Forme variable"

    rows: List[Dict[str, Any]] = [
        {"Metric": "Pace (Vitesse)", "Domicile": f"{pace_h} poss", "Extérieur": f"{pace_a} poss", "Analyse": _analyse_pace(pace_h, pace_a)},
        {"Metric": "Off. Rtg (brut)", "Domicile": f"{off_raw_h} pts/100", "Extérieur": f"{off_raw_a} pts/100", "Analyse": _analyse_off(off_raw_h, off_raw_a)},
        {"Metric": "Adj. Off Rtg (SOS)", "Domicile": f"{off_adj_h} pts/100", "Extérieur": f"{off_adj_a} pts/100", "Analyse": _analyse_off(off_adj_h, off_adj_a)},
        {"Metric": "Def. Rating", "Domicile": str(def_h), "Extérieur": str(def_a), "Analyse": _analyse_def(def_h, def_a)},
        {"Metric": "Forme (5 derniers)", "Domicile": form_h, "Extérieur": form_a, "Analyse": _analyse_form(form_h, form_a)},
    ]
    return rows


# ==============================================================================
# FACTOR MODEL ++ — Score Ajusté (ORB% / TOV%)
# ==============================================================================


def calculate_adjusted_score(stats_team: dict, stats_opponent: dict, league_id: Optional[int] = None) -> float:
    """
    V25 — Base : ppg ou ppg_sos. Bonus ORB% / Malus TOV% avec DEFENSIVE_FACTOR ligue.
    Bonus Rebond Off (ORB%) : si Team_ORB > Opponent_ORB de 5%, Score +3%.
    Malus Turnover (TOV%) : si Team_TOV > Opponent_TOV de 3%, Score -3%.
    DEFENSIVE_FACTOR > 1 amplifie ces ajustements (ligue défensive).
    """
    cfg = _get_league_config(league_id)
    dfac = cfg.get("defensive_factor", 1.0)
    base = stats_team.get("ppg_sos") or stats_team.get("ppg", 75.0)
    orb_team = stats_team.get("orb_pct", 0.25) * 100.0
    orb_opp = stats_opponent.get("orb_pct", 0.25) * 100.0
    tov_team = stats_team.get("tov_pct", 0.15) * 100.0
    tov_opp = stats_opponent.get("tov_pct", 0.15) * 100.0
    mult = 1.0
    if (orb_team - orb_opp) >= ORB_BONUS_THRESHOLD_PCT:
        mult += ORB_BONUS_SCORE_PCT * dfac
    if (tov_team - tov_opp) >= TOV_MALUS_THRESHOLD_PCT:
        mult -= TOV_MALUS_SCORE_PCT * dfac
    return max(50.0, min(120.0, base * mult))


# ==============================================================================
# MODÈLES — Pythagore + Pace Projector + EV O/U (utilise Score Ajusté)
# ==============================================================================


def _apply_bayesian_shrinkage(stats: dict) -> dict:
    """V23 — Régression à la moyenne sur les four factors si n_games < 15."""
    n = float(stats.get("n_games") or 15)
    alpha = min(1.0, n / BAYESIAN_SHRINKAGE_GAMES)
    out = dict(stats)
    for key, league_avg in [
        ("efg_pct", LEAGUE_AVG_EFG),
        ("tov_pct", LEAGUE_AVG_TOV),
        ("orb_pct", LEAGUE_AVG_ORB),
        ("ft_rate", LEAGUE_AVG_FT_RATE),
    ]:
        raw_val = out.get(key, league_avg)
        out[key] = alpha * raw_val + (1.0 - alpha) * league_avg
    return out


# Floor/Ceiling proba : laisser place à l'aléa sportif (pas de 100% ni 0%)
PROBA_FLOOR: float = 0.08
PROBA_CEILING: float = 0.92


def pythagorean_win_prob_home(stats_home: dict, stats_away: dict, league_id: Optional[int] = None) -> float:
    """V25 — Score Ajusté + Bayesian + HOME_ADVANTAGE dynamique par ligue."""
    cfg = _get_league_config(league_id)
    home_adv = cfg.get("home_advantage", HOME_ADVANTAGE_PTS)
    sh = _apply_bayesian_shrinkage(stats_home)
    sa = _apply_bayesian_shrinkage(stats_away)
    adj_home = calculate_adjusted_score(sh, sa, league_id)
    adj_away = calculate_adjusted_score(sa, sh, league_id)
    proj_home = adj_home + home_adv
    proj_away = adj_away
    denom = (proj_home ** PYTHAGOREAN_EXP) + (proj_away ** PYTHAGOREAN_EXP)
    if denom <= 0:
        return 0.5
    raw_prob = float((proj_home ** PYTHAGOREAN_EXP) / denom)
    prob_smoothed = raw_prob * PROB_SMOOTH_FACTOR + PROB_SMOOTH_BIAS
    return float(max(PROBA_FLOOR, min(PROBA_CEILING, prob_smoothed)))


def pythagorean_projection(stats_home: dict, stats_away: dict, league_id: Optional[int] = None) -> Tuple[float, float, float, bool]:
    """
    V25 — Score Ajusté (SOS + Bayesian) avec League Profiler (HOME_ADVANTAGE dynamique).
    Retourne (proj_home, proj_away, total_proj, is_reliable).
    """
    cfg = _get_league_config(league_id)
    home_adv = cfg.get("home_advantage", HOME_ADVANTAGE_PTS)
    sh = _apply_bayesian_shrinkage(stats_home)
    sa = _apply_bayesian_shrinkage(stats_away)
    proj_home = calculate_adjusted_score(sh, sa, league_id) + home_adv
    proj_away = calculate_adjusted_score(sa, sh, league_id)
    total_proj = proj_home + proj_away
    is_reliable = (SCORE_PROJ_MIN <= proj_home <= SCORE_PROJ_MAX and SCORE_PROJ_MIN <= proj_away <= SCORE_PROJ_MAX)
    return proj_home, proj_away, total_proj, is_reliable


def pace_projector_total(league_id: int, season: str, home_id: int, away_id: int, stats_cache: Optional[Dict[Tuple[int, str, int], dict]] = None) -> Tuple[float, bool]:
    """
    V25 — Projection total = Score Ajusté (SOS + Bayesian) + League Profiler.
    Retourne (total_proj, is_reliable).
    """
    sh = _apply_bayesian_shrinkage(_get_stats_from_cache(stats_cache, league_id, season, home_id))
    sa = _apply_bayesian_shrinkage(_get_stats_from_cache(stats_cache, league_id, season, away_id))
    proj_home = calculate_adjusted_score(sh, sa, league_id) + _get_league_config(league_id).get("home_advantage", HOME_ADVANTAGE_PTS)
    proj_away = calculate_adjusted_score(sa, sh, league_id)
    total_proj = proj_home + proj_away
    is_reliable = (SCORE_PROJ_MIN <= proj_home <= SCORE_PROJ_MAX and SCORE_PROJ_MIN <= proj_away <= SCORE_PROJ_MAX)
    return round(float(np.clip(total_proj, 120.0, 200.0)), 1), is_reliable


def prob_over_from_projection(projection: float, line: float, sigma: float = TOTAL_PTS_SIGMA) -> float:
    """P(Total > line) sous hypothèse Total ~ N(projection, sigma)."""
    if sigma <= 0:
        return 0.5
    return float(1.0 - norm.cdf((line - projection) / sigma))


def ev_pct(prob: float, cote: float) -> float:
    """EV% = (NotreProba * CoteBookmaker) - 1, en %."""
    if cote is None or cote <= 0:
        return 0.0
    return ((prob * cote) - 1.0) * 100.0


def kelly_criterion(prob: float, odd: float, fraction: float = KELLY_FRACTION) -> float:
    """
    V22 — Critère de Kelly (Fractional) pour la mise optimale en %.
    f = (b*p - q) / b avec b = cote-1, p = proba, q = 1-p.
    Fractional Kelly : résultat divisé par fraction (4) pour plus de sécurité.
    Retourne 0 si EV négatif.
    """
    if odd is None or odd <= 0 or prob <= 0 or prob >= 1:
        return 0.0
    b = odd - 1.0
    q = 1.0 - prob
    ev = prob * odd - 1.0
    if ev <= 0:
        return 0.0
    kelly_full = (b * prob - q) / b
    if kelly_full <= 0:
        return 0.0
    fractional = kelly_full / fraction
    return min(1.0, max(0.0, fractional)) * 100.0


# ==============================================================================
# BACKTEST P&L (J-1 / J-2)
# ==============================================================================


def _game_total_scores(g: dict) -> Tuple[int, int]:
    """Extrait les scores totaux du match (API : scores.home / scores.away ou .total)."""
    scores = g.get("scores") or {}
    home = scores.get("home")
    away = scores.get("away")
    if isinstance(home, dict):
        pts_h = _safe_int(home.get("total") or home.get("points"))
    else:
        pts_h = _safe_int(home)
    if isinstance(away, dict):
        pts_a = _safe_int(away.get("total") or away.get("points"))
    else:
        pts_a = _safe_int(away)
    return pts_h, pts_a


@st.cache_data(ttl=300)
def run_backtest_yesterday() -> Dict[str, Any]:
    """
    BACKTEST HONNÊTE : Prédiction basée uniquement sur données connues AVANT le match (Time Machine).
    Pour chaque match J-1/J-2 : get_clean_stats_at_date(team_id, date_match) = stats à la veille du match.
    Compare prédiction (basée sur données J-1) vs réalité. Cote simulée 1.65. P&L en unités.
    """
    today = date.today()
    TIME_MACHINE_DEBUG_LOGS.clear()
    results: Dict[str, Any] = {"bets": 0, "wins": 0, "profit": 0.0, "details": [], "matches_seen": 0}
    sim_odd = 1.65
    for days_back in range(1, WINDOW_PAST_BACKTEST + 1):
        d_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for league_name, league_id in LEAGUES.items():
            games: List[dict] = []
            used_season = ""
            for season in SEASONS_TO_TRY:
                games = fetch_games(league_id, season, d_str)
                if games:
                    used_season = season
                    break
            if not games:
                games, used_season = fetch_games_by_date_only(league_id, d_str)
            if not games:
                continue
            for g in games:
                status = g.get("status") or {}
                status_short = status.get("short") if isinstance(status, dict) else str(status)
                pts_h, pts_a = _game_total_scores(g)
                finished = (status_short == "FT") or (pts_h > 0 or pts_a > 0)
                if not finished:
                    continue
                if pts_h == 0 and pts_a == 0:
                    continue
                winner_real = "HOME" if pts_h > pts_a else "AWAY"
                teams = g.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
                away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
                home_name = home.get("name", "Domicile") if isinstance(home, dict) else "Domicile"
                away_name = away.get("name", "Extérieur") if isinstance(away, dict) else "Extérieur"
                if not home_id or not away_id:
                    continue
                results["matches_seen"] += 1
                try:
                    sh = get_clean_stats_at_date(league_id, used_season, home_id, d_str)
                    sa = get_clean_stats_at_date(league_id, used_season, away_id, d_str)
                except Exception:
                    results["details"].append(f"[{league_name}] {home_name} vs {away_name} | Stats J-1 indisponibles (API)")
                    continue
                if sh is None or sa is None:
                    results["details"].append(f"[{league_name}] {home_name} vs {away_name} | Pas de Data (stats indisponibles)")
                    continue
                prob_h = pythagorean_win_prob_home(sh, sa, league_id)
                low_quality = (sh.get("quality") == "LOW" or sa.get("quality") == "LOW")
                warn = " ⚠️ Score Only" if low_quality else ""
                bet_side = None
                if prob_h >= 0.60:
                    bet_side = "HOME"
                elif prob_h <= 0.40:
                    bet_side = "AWAY"
                if not bet_side:
                    results["details"].append(f"[{league_name}] {home_name} vs {away_name} | Pas de pari (proba {prob_h:.0%} entre 40–60 %){warn}")
                    continue
                results["bets"] += 1
                is_win = bet_side == winner_real
                if is_win:
                    results["wins"] += 1
                    results["profit"] += sim_odd - 1.0
                else:
                    results["profit"] -= 1.0
                results["details"].append(
                    f"{home_name} vs {away_name} | Prédiction (données J-1): {bet_side} ({prob_h:.0%}) | Réalité: {winner_real} | {'✅' if is_win else '❌'}{warn}"
                )
    results["debug_logs"] = list(TIME_MACHINE_DEBUG_LOGS)
    return results


# ==============================================================================
# BULLDOZER
# ==============================================================================


# Codes pays API → nom attendu (au cas où l'API renvoie "FR" au lieu de "France")
COUNTRY_CODE_TO_NAME: Dict[str, str] = {
    "FR": "France", "France": "France",
    "ES": "Spain", "Spain": "Spain", "ESP": "Spain",
    "IT": "Italy", "Italy": "Italy", "ITA": "Italy",
    "TR": "Turkey", "Turkey": "Turkey", "TUR": "Turkey",
    "GR": "Greece", "Greece": "Greece", "GRE": "Greece",
    "AU": "Australia", "Australia": "Australia",
}


def _game_country(g: dict) -> str:
    """Extrait le pays de la ligue du match (API : league.country ou league.country.name ou code)."""
    league = g.get("league") or {}
    country = league.get("country")
    if country is None:
        return ""
    if isinstance(country, dict):
        name = (country.get("name") or "").strip()
        code = (country.get("code") or "").strip().upper()
        if name:
            return name
        return COUNTRY_CODE_TO_NAME.get(code, code)
    raw = str(country).strip()
    return COUNTRY_CODE_TO_NAME.get(raw.upper(), raw)


def _game_passes_country_filter(g: dict, league_id: int) -> bool:
    """
    Filtre géo : rejette seulement si pays explicite et mauvais (ex. Australia pour Pro B FR).
    Si l'API ne renvoie pas de pays → on accepte (comportement "avant" pour retrouver les matchs).
    """
    expected = LEAGUE_COUNTRY.get(league_id)
    if expected is None:
        return True
    country = _game_country(g)
    if not country:
        return True
    country_norm = COUNTRY_CODE_TO_NAME.get(country.upper(), country)
    if expected == "Europe":
        europe = {"France", "Spain", "Italy", "Turkey", "Greece", "Germany", "Russia", "Serbia", "Lithuania", "Israel", "Croatia", "Slovenia", "Montenegro", "Poland", "Czech Republic", "Belgium", "Europe"}
        return country_norm in europe
    if expected == "France" and country_norm == "Australia":
        return False
    return country_norm.lower() == expected.lower()


@st.cache_data(ttl=120)
def collect_all_games_72h() -> List[Tuple[dict, str, int, str, str]]:
    """Force brute + filtre géo : si Betclic/Pro B → pays DOIT être France, etc."""
    today = date.today()
    seen_gid: set = set()
    out: List[Tuple[dict, str, int, str, str]] = []
    for league_name, league_id in LEAGUES.items():
        for d in range(WINDOW_DAYS):
            day_date = today + timedelta(days=d)
            date_str = day_date.strftime("%Y-%m-%d")
            games_for_date: List[dict] = []
            used_season = ""
            for season in SEASONS_TO_TRY:
                try:
                    games_for_date = fetch_games(league_id, season, date_str)
                    if games_for_date:
                        used_season = season
                        break
                except Exception:
                    continue
            for g in games_for_date:
                if not _game_passes_country_filter(g, league_id):
                    continue
                gid = g.get("id")
                if gid is None or gid in seen_gid:
                    continue
                seen_gid.add(gid)
                out.append((g, league_name, league_id, used_season, date_str))
    return out


# ==============================================================================
# V21 TURBO — Préchargement parallèle des stats
# ==============================================================================

def preload_all_teams_stats(games_with_meta: List[Tuple[dict, str, int, str, str]]) -> Dict[Tuple[int, str, int], dict]:
    """
    Charge toutes les stats d'équipes en parallèle (ThreadPoolExecutor).
    Retourne un dict {(league_id, season, team_id): stats} pour alimenter les builders sans rappel API.
    """
    unique_keys: set = set()
    for g, _league_name, league_id, season, _date_str in games_with_meta:
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        if home_id:
            unique_keys.add((league_id, season, home_id))
        if away_id:
            unique_keys.add((league_id, season, away_id))
    results: Dict[Tuple[int, str, int], dict] = {}
    default_stats = {"ppg": 75.0, "oppg": 75.0, "pace": PACE_DEFAULT_LIGUE, "efg_pct": 0.5, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25, "n_games": 0, "_incomplete": True, "_data_source": "FLUX DATA INTERROMPU"}
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_key = {
            executor.submit(get_team_stats_weighted, lid, seas, tid, True): (lid, seas, tid)
            for (lid, seas, tid) in unique_keys
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = default_stats.copy()

        # V23 — Strength of Schedule + Volatilité
    for (lid, seas, tid) in list(results.keys()):
        st = results[(lid, seas, tid)]
        raw_off = st.get("off_rating", 100.0)
        st["off_rating_raw"] = raw_off
        last_with_opp: List[dict] = []
        try:
            last_with_opp = fetch_team_games_with_opponents(lid, seas, tid, n=LAST_N_GAMES)
            if not last_with_opp:
                st["volatility_std"] = 0.0
                continue
            def_ratings: List[float] = []
            for g in last_with_opp:
                opp_id = g.get("opponent_id")
                if opp_id is None:
                    continue
                opp_key = (lid, seas, int(opp_id))
                if opp_key in results:
                    def_ratings.append(results[opp_key].get("def_rating", LEAGUE_AVG_DEF_RTG))
            if def_ratings:
                opp_avg_def = sum(def_ratings) / len(def_ratings)
                # SOS uniquement si vraies données adverses (pas la valeur par défaut 112.0)
                if abs(opp_avg_def - LEAGUE_AVG_DEF_RTG) >= 0.01:
                    adj_off = raw_off + (LEAGUE_AVG_DEF_RTG - opp_avg_def)
                    st["off_rating"] = max(80.0, min(130.0, adj_off))
                    pace = st.get("pace", PACE_DEFAULT_LIGUE)
                    st["ppg_sos"] = st["off_rating"] / 100.0 * pace  # pour calculate_adjusted_score
            # V23 — Volatilité : écart-type des pts_for sur les 5 derniers matchs (même last_with_opp)
            pts_for_list = [g.get("pts_for") for g in last_with_opp if g.get("pts_for") is not None]
            if len(pts_for_list) >= 2:
                st["volatility_std"] = float(np.std(pts_for_list))
            else:
                st["volatility_std"] = 0.0
        except Exception:
            st["volatility_std"] = 0.0
    return results


def _get_stats_from_cache(stats_cache: Optional[Dict[Tuple[int, str, int], dict]], league_id: int, season: str, team_id: int) -> dict:
    """Utilise le cache TURBO si fourni, sinon appelle l'API."""
    if stats_cache is not None:
        key = (league_id, season, team_id)
        if key in stats_cache:
            return stats_cache[key]
    return get_team_stats_weighted(league_id, season, team_id)


# ==============================================================================
# BUILD ML ROWS (avec EV pour Combiné)
# ==============================================================================


def build_ml_rows(games_with_meta: List[Tuple[dict, str, int, str, str]], stats_cache: Optional[Dict[Tuple[int, str, int], dict]] = None) -> pd.DataFrame:
    rows: List[dict] = []
    for g, league_name, league_id, season, date_str in games_with_meta:
        gid = g.get("id")
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        home_name = home.get("name", "Domicile") if isinstance(home, dict) else "Domicile"
        away_name = away.get("name", "Extérieur") if isinstance(away, dict) else "Extérieur"
        if not home_id or not away_id:
            continue
        try:
            sh = _get_stats_from_cache(stats_cache, league_id, season, home_id)
            sa = _get_stats_from_cache(stats_cache, league_id, season, away_id)
        except Exception:
            sh = {"ppg": 75.0, "oppg": 75.0}
            sa = {"ppg": 75.0, "oppg": 75.0}
        prob_home = pythagorean_win_prob_home(sh, sa, league_id)
        prob_away = 1.0 - prob_home
        fair_odd_home = round(1.0 / prob_home, 2) if prob_home > 0 else 999.99
        fair_odd_away = round(1.0 / prob_away, 2) if prob_away > 0 else 999.99
        odd_home, odd_away = fetch_odds(gid, league_id, season)
        match_label = f"{home_name} vs {away_name}"
        ev_home = ev_pct(prob_home, odd_home) if odd_home else 0.0
        ev_away = ev_pct(prob_away, odd_away) if odd_away else 0.0
        for prob, fair_odd, market_odd, pari, ev in [
            (prob_home, fair_odd_home, odd_home, home_name, ev_home),
            (prob_away, fair_odd_away, odd_away, away_name, ev_away),
        ]:
            rows.append({
                "_game_id": gid, "_league_id": league_id, "_season": season, "_date": date_str,
                "_home_id": home_id, "_away_id": away_id, "_home_name": home_name, "_away_name": away_name,
                "_league_name": league_name, "Ligue": league_name, "Match": match_label, "Pari": pari,
                "Notre Proba": round(prob * 100, 2), "Fair Odd": fair_odd, "Market Odd": market_odd if market_odd else "-",
                "EDGE %": round(ev, 2), "_prob": prob, "_odd": market_odd, "_ev": ev, "_type": "ML",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("EDGE %", ascending=False, ignore_index=True)


def _format_game_time(g: dict) -> str:
    """Heure du match : date + time si dispo (ex. 31/01 20:00)."""
    d = g.get("date") or ""
    t = g.get("time") or ""
    if not d:
        return "-"
    try:
        dt_str = d
        if t and ":" in t:
            dt_str = f"{d}T{t}"
            dt = datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            dt = datetime.strptime(d[:10], "%Y-%m-%d")
        return dt.strftime("%d/%m %H:%M") if t else dt.strftime("%d/%m")
    except Exception:
        return d[:10] if len(d) >= 10 else "-"


def _money_bucket_label(edge: float, fiabilite: str) -> str:
    """V25 — Money Management Buckets : Safe / Standard / Agressif selon Edge et Fiabilité."""
    if edge < 3.0:
        return "Safe (1%)"
    if edge <= 7.0:
        return "Standard (2%)"
    if edge > 7.0 and "🟢" in fiabilite:
        return "Agressif (3-5%)"
    return "Standard (2%)"


def build_actionable_table(games_with_meta: List[Tuple[dict, str, int, str, str]], stats_cache: Optional[Dict[Tuple[int, str, int], dict]] = None) -> pd.DataFrame:
    """
    V25 — Tableau actionnable avec Fair Spread et Money Buckets.
    Conseil : NO BET si 🔴, 🔥 MAX BET si 🟢 et Edge > 10%, sinon Value / Bet modéré.
    """
    rows: List[dict] = []
    for g, league_name, league_id, season, date_str in games_with_meta:
        gid = g.get("id")
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        home_name = home.get("name", "Domicile") if isinstance(home, dict) else "Domicile"
        away_name = away.get("name", "Extérieur") if isinstance(away, dict) else "Extérieur"
        if not home_id or not away_id:
            continue
        try:
            sh = _get_stats_from_cache(stats_cache, league_id, season, home_id)
            sa = _get_stats_from_cache(stats_cache, league_id, season, away_id)
        except Exception:
            sh = {"ppg": 75.0, "oppg": 75.0}
            sa = {"ppg": 75.0, "oppg": 75.0}
        odd_home, odd_away = fetch_odds(gid, league_id, season)
        incomplete = sh.get("_incomplete") or sa.get("_incomplete")
        if incomplete:
            prob_home = prob_away = 0.5
            vainqueur = "—"
            proba_ajustee = "⚠️ DATA MANQUANTE"
            cote = odd_home if odd_home is not None else odd_away if odd_away is not None else "-"
            edge = 0.0
            fiabilite = "⚠️ DATA MANQUANTE"
            volatility_std = 0.0
            fair_spread = "—"
        else:
            prob_home = pythagorean_win_prob_home(sh, sa, league_id)
            prob_away = 1.0 - prob_home
            _, sym_h, label_h = reality_check_mae(league_id, season, home_id)
            _, sym_a, label_a = reality_check_mae(league_id, season, away_id)
            if sym_h == "🔴" or sym_a == "🔴":
                fiabilite = "🔴 Volatile / Imprévisible"
            elif sym_h == "🟡" or sym_a == "🟡":
                fiabilite = "🟡 Moyen"
            else:
                fiabilite = "🟢 Modèle Calibré"
            if prob_home >= prob_away:
                vainqueur = home_name
                proba_ajustee = round(prob_home * 100, 1)
                cote = odd_home if odd_home is not None else "-"
                edge = ev_pct(prob_home, odd_home) if odd_home else 0.0
            else:
                vainqueur = away_name
                proba_ajustee = round(prob_away * 100, 1)
                cote = odd_away if odd_away is not None else "-"
                edge = ev_pct(prob_away, odd_away) if odd_away else 0.0
            vol_h = sh.get("volatility_std") or 0.0
            vol_a = sa.get("volatility_std") or 0.0
            volatility_std = max(vol_h, vol_a)
            if volatility_std > VOLATILITY_STD_THRESHOLD:
                proba_ajustee = max(0.0, proba_ajustee - VOLATILITY_PROB_PENALTY_PCT)
                proba_ajustee = round(proba_ajustee, 1)
                fiabilite = fiabilite + " ⚠️"
            # V25 — Fair Spread (notre prédiction + bookmaker si dispo)
            proj_home, proj_away, _, _ = pythagorean_projection(sh, sa, league_id)
            fair_spread = calculate_fair_spread(proj_home, proj_away, home_name, away_name)
            bm_spread, _, _ = fetch_odds_spread(gid, league_id, season)
            if bm_spread is not None:
                fair_spread = f"{fair_spread} | Book: {bm_spread:+.1f}"

        data_source = "Qualité Haute"
        conseil = "Bet modéré"
        if incomplete:
            data_source = "FLUX DATA INTERROMPU"
            conseil = "FLUX DATA INTERROMPU"
        elif sh.get("_data_source") == "FLUX DATA INTERROMPU" or sa.get("_data_source") == "FLUX DATA INTERROMPU":
            data_source = "FLUX DATA INTERROMPU"
            conseil = "FLUX DATA INTERROMPU"
        elif sh.get("_data_source") == "Qualité Médium (Score Only)" or sa.get("_data_source") == "Qualité Médium (Score Only)":
            data_source = "Qualité Médium (Score Only)"
        elif sh.get("_data_source") == "Qualité Médium (API)" or sa.get("_data_source") == "Qualité Médium (API)":
            data_source = "Qualité Médium (API)"
        if not incomplete and conseil != "FLUX DATA INTERROMPU":
            if "🔴" in fiabilite:
                conseil = "NO BET (Trop volatile)"
            elif "🟢" in fiabilite and edge >= EDGE_MAX_BET_THRESHOLD:
                conseil = "🔥 MAX BET"
            elif edge > 0:
                conseil = "Value"
            else:
                conseil = "Bet modéré"
        # V25 — Money Buckets (Safe / Standard / Agressif) ; News Search = lien X/Twitter absences
        mise_label = _money_bucket_label(edge, fiabilite) if not incomplete else "—"
        if incomplete or "NO BET" in str(conseil):
            mise_label = "—"
        query = quote_plus(f"{home_name} {away_name} injury basketball").replace("%20", "+")
        news_search_url = f"https://x.com/search?q={query}&f=live"
        rows.append({
            "Heure": _format_game_time(g),
            "Match": f"{home_name} vs {away_name}",
            "Fair Spread": fair_spread,
            "Vainqueur Prédit": vainqueur,
            "Probabilité Ajustée %": proba_ajustee,
            "Volatilité": round(volatility_std, 1),
            "Cote": cote,
            "Mise (%)": mise_label,
            "Fiabilité Modèle": fiabilite,
            "Conseil": conseil,
            "_data_source": data_source,
            "News Search": news_search_url,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df


# ==============================================================================
# BUILD O/U TABLE — Match | Notre Total | Ligne | Cote Over | Cote Under | EV%
# ==============================================================================


def build_ou_table(games_with_meta: List[Tuple[dict, str, int, str, str]], stats_cache: Optional[Dict[Tuple[int, str, int], dict]] = None) -> pd.DataFrame:
    rows: List[dict] = []
    for g, league_name, league_id, season, date_str in games_with_meta:
        gid = g.get("id")
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        home_name = home.get("name", "Domicile") if isinstance(home, dict) else "Domicile"
        away_name = away.get("name", "Extérieur") if isinstance(away, dict) else "Extérieur"
        if not home_id or not away_id:
            continue
        notre_total, is_reliable = pace_projector_total(league_id, season, home_id, away_id, stats_cache)
        line, cote_over, cote_under = fetch_odds_totals(gid, league_id, season)
        match_label = f"{home_name} vs {away_name}"
        if not is_reliable:
            prob_over, prob_under = 0.5, 0.5
            ev_over = ev_under = 0.0
        else:
            prob_over = prob_over_from_projection(notre_total, line) if line is not None else 0.5
            prob_under = 1.0 - prob_over
            ev_over = ev_pct(prob_over, cote_over) if cote_over else 0.0
            ev_under = ev_pct(prob_under, cote_under) if cote_under else 0.0
        ev_best = max(ev_over, ev_under)
        display_total: Any = notre_total if is_reliable else "Pas de prédiction fiable"
        _, sym_h, label_h = reality_check_mae(league_id, season, home_id)
        _, sym_a, label_a = reality_check_mae(league_id, season, away_id)
        if sym_a == "🔴" or sym_h == "🔴":
            fiabilite = "🔴 Volatile / Imprévisible"
        elif sym_a == "🟡" or sym_h == "🟡":
            fiabilite = "🟡 Moyen"
        else:
            fiabilite = "🟢 Modèle Calibré"
        rows.append({
            "_game_id": gid, "_league_id": league_id, "_season": season, "_date": date_str,
            "_home_id": home_id, "_away_id": away_id, "_home_name": home_name, "_away_name": away_name,
            "_league_name": league_name,
            "Match": match_label,
            "Notre Total": display_total,
            "_notre_total_raw": notre_total,
            "_reliable": is_reliable,
            "Fiabilité Modèle": fiabilite,
            "Ligne Bookmaker": line if line is not None else "-",
            "Cote Over": cote_over if cote_over is not None else "-",
            "Cote Under": cote_under if cote_under is not None else "-",
            "EV% Over": round(ev_over, 2),
            "EV% Under": round(ev_under, 2),
            "EV%": round(ev_best, 2),
            "_prob_over": prob_over, "_prob_under": prob_under,
            "_cote_over": cote_over, "_cote_under": cote_under,
            "_ev_over": ev_over, "_ev_under": ev_under,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("EV%", ascending=False, ignore_index=True)


# ==============================================================================
# SMART COMBINÉ — Filtre cote ≥ 1.30, sélection par max EV
# ==============================================================================


def build_combine_candidates(df_ml: pd.DataFrame, df_ou: pd.DataFrame) -> List[dict]:
    candidates: List[dict] = []
    if not df_ml.empty:
        for _, r in df_ml.iterrows():
            odd = r.get("_odd")
            if odd is not None and odd >= MIN_ODD_COMBINE:
                candidates.append({
                    "type": "ML",
                    "label": f"{r['Match']} — {r['Pari']}",
                    "odd": odd,
                    "ev": r.get("_ev", 0),
                    "prob": r.get("_prob", 0),
                })
    if not df_ou.empty:
        for _, r in df_ou.iterrows():
            for side, key_odd, key_ev, key_prob in [
                ("Over", "_cote_over", "_ev_over", "_prob_over"),
                ("Under", "_cote_under", "_ev_under", "_prob_under"),
            ]:
                odd = r.get(key_odd)
                if odd is not None and odd >= MIN_ODD_COMBINE:
                    candidates.append({
                        "type": "O/U",
                        "label": f"{r['Match']} — {side}",
                        "odd": odd,
                        "ev": r.get(key_ev, 0),
                        "prob": r.get(key_prob, 0),
                    })
    return sorted(candidates, key=lambda x: x["ev"], reverse=True)


def select_smart_combine(candidates: List[dict]) -> Tuple[Optional[dict], Optional[dict], float]:
    """V22 : filtre Small Markets — cote individuelle ≤ 10, cote totale ≤ 25."""
    filtered = [c for c in candidates if (c.get("odd") or 0) <= MAX_ODD_SINGLE]
    if len(filtered) >= 2:
        c1, c2 = filtered[0], filtered[1]
        total = c1["odd"] * c2["odd"]
        if total > MAX_ODD_COMBINE_TOTAL:
            return None, None, 0.0
        return c1, c2, total
    if len(filtered) == 1:
        return filtered[0], None, filtered[0]["odd"]
    return None, None, 0.0


# ==============================================================================
# MATCH VISION — Simulation score (Gauss), Four Factors, Player Monte Carlo
# ==============================================================================


def plot_score_simulation(proj_home: float, proj_away: float, line_total: Optional[float], home_name: str, away_name: str) -> go.Figure:
    sigma = TOTAL_PTS_SIGMA
    x = np.linspace(50, 120, 200)
    y_h = norm.pdf(x, proj_home, sigma)
    y_a = norm.pdf(x, proj_away, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_h, mode="lines", name=home_name, line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x, y=y_a, mode="lines", name=away_name, line=dict(color="red")))
    if line_total is not None:
        fig.add_vline(x=line_total / 2, line_dash="dash", line_color="gray", annotation_text="Ligne/2 (ref)")
    fig.update_layout(title="Distribution des points (Gauss)", xaxis_title="Points", yaxis_title="Densité", showlegend=True, height=320)
    return fig


def _radar_safe(key: str, value: float, default: float) -> float:
    """Pour le radar : utilise default si facteur aberrant (évite échelle cassée)."""
    return default if _is_aberrant_factor(key, value) else value


def plot_four_factors_radar(stats_home: dict, stats_away: dict, home_name: str, away_name: str) -> go.Figure:
    """RADAR FIX V16 : deux équipes visibles. Propreté = (1 - TOV%). Facteurs aberrants → valeur par défaut."""
    categories = ["Efficacité (eFG%)", "Rebond Off (ORB%)", "Propreté (Inv. TOV%)", "Lancers (FT Rate)"]
    efg_h = _radar_safe("efg_pct", stats_home.get("efg_pct", 0.5), 0.5)
    efg_a = _radar_safe("efg_pct", stats_away.get("efg_pct", 0.5), 0.5)
    orb_h = _radar_safe("orb_pct", stats_home.get("orb_pct", 0.25), 0.25)
    orb_a = _radar_safe("orb_pct", stats_away.get("orb_pct", 0.25), 0.25)
    tov_h = _radar_safe("tov_pct", stats_home.get("tov_pct", 0.15), 0.15)
    tov_a = _radar_safe("tov_pct", stats_away.get("tov_pct", 0.15), 0.15)
    ft_h = _radar_safe("ft_rate", stats_home.get("ft_rate", 0.25), 0.25)
    ft_a = _radar_safe("ft_rate", stats_away.get("ft_rate", 0.25), 0.25)
    val_h = [efg_h, orb_h, 1.0 - tov_h, ft_h]
    val_a = [efg_a, orb_a, 1.0 - tov_a, ft_a]
    r_h = val_h + [val_h[0]]
    r_a = val_a + [val_a[0]]
    theta = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_h, theta=theta, fill="toself", name=home_name, line=dict(color="blue")))
    fig.add_trace(go.Scatterpolar(r=r_a, theta=theta, fill="toself", name=away_name, line=dict(color="red")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.6])),
        showlegend=True,
        title="Four Factors (les deux équipes)",
        height=400,
    )
    return fig


def _format_factor_pct(key: str, value: float) -> str:
    """Affiche le facteur en % ou 'N/A' si aberrant (V16)."""
    if _is_aberrant_factor(key, value):
        return "N/A"
    return f"{value * 100:.1f}%"


def build_factor_bars_and_verdict(stats_home: dict, stats_away: dict, home_name: str, away_name: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Comparatif Facteurs Clés (eFG%, ORB%, TOV%) et phrase Le Verdict en langage naturel.
    Retourne (verdict_sentence, list of {factor, home_pct, away_pct, home_na, away_na} pour barres).
    V16 : facteurs aberrants affichés N/A.
    """
    efg_h = stats_home.get("efg_pct", 0.5) * 100
    efg_a = stats_away.get("efg_pct", 0.5) * 100
    orb_h = stats_home.get("orb_pct", 0.25) * 100
    orb_a = stats_away.get("orb_pct", 0.25) * 100
    tov_h = stats_home.get("tov_pct", 0.15) * 100
    tov_a = stats_away.get("tov_pct", 0.15) * 100
    pace_h = stats_home.get("pace", PACE_DEFAULT_LIGUE)
    pace_a = stats_away.get("pace", PACE_DEFAULT_LIGUE)
    bars = [
        {"factor": "Efficacité Tir (eFG%)", "home_pct": efg_h, "away_pct": efg_a, "home_na": _is_aberrant_factor("efg_pct", stats_home.get("efg_pct", 0.5)), "away_na": _is_aberrant_factor("efg_pct", stats_away.get("efg_pct", 0.5))},
        {"factor": "Rebond Offensif %", "home_pct": orb_h, "away_pct": orb_a, "home_na": _is_aberrant_factor("orb_pct", stats_home.get("orb_pct", 0.25)), "away_na": _is_aberrant_factor("orb_pct", stats_away.get("orb_pct", 0.25))},
        {"factor": "Pertes de balle %", "home_pct": tov_h, "away_pct": tov_a, "home_na": _is_aberrant_factor("tov_pct", stats_home.get("tov_pct", 0.15)), "away_na": _is_aberrant_factor("tov_pct", stats_away.get("tov_pct", 0.15))},
    ]
    parts: List[str] = []
    if not _is_aberrant_factor("orb_pct", stats_home.get("orb_pct", 0.25)) and not _is_aberrant_factor("orb_pct", stats_away.get("orb_pct", 0.25)) and abs(orb_h - orb_a) >= 2:
        leader_orb = home_name if orb_h > orb_a else away_name
        diff = abs(orb_h - orb_a)
        parts.append(f"{leader_orb} domine le rebond (+{diff:.0f}%)")
    if not _is_aberrant_factor("efg_pct", stats_home.get("efg_pct", 0.5)) and not _is_aberrant_factor("efg_pct", stats_away.get("efg_pct", 0.5)) and abs(efg_h - efg_a) >= 2:
        leader_efg = home_name if efg_h > efg_a else away_name
        diff = abs(efg_h - efg_a)
        parts.append(f"{leader_efg} est plus efficace au tir (+{diff:.1f}%)")
    if not _is_aberrant_factor("tov_pct", stats_home.get("tov_pct", 0.15)) and not _is_aberrant_factor("tov_pct", stats_away.get("tov_pct", 0.15)) and abs(tov_h - tov_a) >= 1:
        better_tov = home_name if tov_h < tov_a else away_name
        parts.append(f"{better_tov} perd moins la balle")
    pace_leader = home_name if pace_h > pace_a else away_name
    parts.append(f"Avantage Pace : {pace_leader}.")
    verdict = " ".join(parts) if parts else f"Équilibre. Avantage Pace : {pace_leader}."
    return verdict, bars


def render_factor_bars(bars: List[Dict[str, Any]], home_name: str, away_name: str) -> None:
    """Affiche barres de progression A vs B pour eFG%, ORB%, TOV%. V16 : N/A si facteur aberrant."""
    for b in bars:
        f, h, a = b["factor"], b["home_pct"], b["away_pct"]
        home_na = b.get("home_na", False)
        away_na = b.get("away_na", False)
        st.caption(f"**{f}**")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.progress(min(1.0, h / 100.0) if not home_na else 0.0)
            st.caption(f"{home_name}: {'N/A' if home_na else f'{h:.1f}%'}")
        with col3:
            st.progress(min(1.0, a / 100.0) if not away_na else 0.0)
            st.caption(f"{away_name}: {'N/A' if away_na else f'{a:.1f}%'}")


# Players + Monte Carlo (fetch_players / fetch_player_stats utilisent SEASONS_TO_TRY pour ABA/Dubai)
def fetch_players(team_id: int, season: str) -> List[dict]:
    for s in [season] + [x for x in SEASONS_TO_TRY if x != season]:
        data, err = _api_get("players", {"team": team_id, "season": s})
        if err or not data:
            continue
        resp = data.get("response")
        if isinstance(resp, list) and len(resp) > 0:
            return resp
    return []


def fetch_player_stats(player_id: int, team_id: int, season: str) -> List[dict]:
    for s in [season] + [x for x in SEASONS_TO_TRY if x != season]:
        data, err = _api_get("games/statistics/players", {"player": player_id, "season": s, "team": team_id})
        if not err and data:
            resp = data.get("response")
            if isinstance(resp, list):
                return resp
        data2, err2 = _api_get("games/statistics/players", {"player": player_id, "season": s})
        if not err2 and data2:
            resp = data2.get("response")
            if isinstance(resp, list):
                return resp
    return []


def aggregate_team_stats_from_players(game_id: int, team_id: int, season: str) -> Optional[dict]:
    """
    Arme nucléaire (Fix ABA/Dubai) : si games/statistics/teams est vide, reconstruit les stats équipe
    en sommant les stats de chaque joueur pour ce match. Retourne {FGA, FGM, 3PM, FTA, FTM, ORB, TRB, TOV, Possessions, pts_for}.
    """
    players = fetch_players(team_id, season)
    if not players:
        return None
    tot_fga = tot_fgm = tot_3pm = tot_fta = tot_ftm = tot_orb = tot_trb = tot_tov = tot_pts = 0
    found_any = False
    for p in players:
        pid = p.get("id")
        if not pid:
            continue
        time.sleep(0.15)
        raw_list = fetch_player_stats(pid, team_id, season)
        for row in raw_list:
            g = row.get("game") or {}
            row_gid = g.get("id") if isinstance(g, dict) else row.get("game_id")
            if row_gid != game_id:
                continue
            found_any = True
            pts = _int_or_total(row.get("points"), 0)
            reb = row.get("rebounds")
            if isinstance(reb, dict):
                tot_trb += _int_or_total(reb.get("total"), 0)
                tot_orb += _int_or_total(reb.get("offensive"), 0)
            else:
                tot_trb += _int_or_total(reb, 0)
            tot_tov += _int_or_total(row.get("turnovers"), 0)
            tot_pts += pts
            fg = row.get("field_goals") or row.get("goals")
            if isinstance(fg, dict):
                tot_fga += _int_or_total(fg.get("attempted") or fg.get("attempts"), 0)
                tot_fgm += _int_or_total(fg.get("made") or fg.get("total"), 0)
            th = row.get("three_points") or row.get("threepoint_goals")
            if isinstance(th, dict):
                tot_3pm += _int_or_total(th.get("made") or th.get("total"), 0)
            ft = row.get("free_throws") or row.get("freethrows")
            if isinstance(ft, dict):
                tot_fta += _int_or_total(ft.get("attempted") or ft.get("attempts"), 0)
                tot_ftm += _int_or_total(ft.get("made") or ft.get("total"), 0)
    if not found_any:
        return None
    poss = tot_fga + int(0.44 * tot_fta) + tot_tov - tot_orb
    if poss <= 0:
        poss = max(1, tot_fga + tot_fta)
    return {
        "FGA": tot_fga, "FGM": tot_fgm, "3PM": tot_3pm, "FTA": tot_fta, "FTM": tot_ftm,
        "ORB": tot_orb, "TRB": tot_trb, "TOV": tot_tov, "Possessions": poss, "pts_for": tot_pts,
    }


def _minutes_to_float(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    if isinstance(val, str) and ":" in val:
        parts = val.strip().split(":")
        if len(parts) == 2:
            try:
                return int(parts[0]) + int(parts[1]) / 60.0
            except (ValueError, TypeError):
                pass
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return 0.0


def _int_or_total(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, dict):
        return max(0, int(val.get("total", default)))
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return default


def _extract_game_stats(raw_row: dict) -> Optional[dict]:
    min_f = _minutes_to_float(raw_row.get("minutes"))
    if min_f <= 0:
        return None
    pts = _int_or_total(raw_row.get("points"), 0)
    reb = _int_or_total(raw_row.get("rebounds"), 0)
    ast = _int_or_total(raw_row.get("assists"), 0)
    ppm = pts / min_f if min_f > 0 else 0.0
    rpm = reb / min_f if min_f > 0 else 0.0
    apm = ast / min_f if min_f > 0 else 0.0
    return {"MIN": min_f, "PTS": pts, "REB": reb, "AST": ast, "PPM": ppm, "RPM": rpm, "APM": apm}


def build_player_stats_df(raw_stats: List[dict]) -> pd.DataFrame:
    rows = []
    for raw_row in raw_stats:
        row = _extract_game_stats(raw_row)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["MIN", "PTS", "REB", "AST", "PPM", "RPM", "APM"])
    return pd.DataFrame(rows)


def run_monte_carlo_player(df: pd.DataFrame, n_sim: int = N_SIM, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if df.empty or len(df) < 2:
        return np.array([]), np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    n = len(df)
    min_min = float(df["MIN"].min())
    max_min = float(df["MIN"].max())
    mean_min = float(df["MIN"].mean())
    std_min = float(df["MIN"].std())
    if std_min <= 0 or np.isnan(std_min):
        std_min = max(1.0, (max_min - min_min) / 4.0)
    a = (min_min - mean_min) / std_min
    b = (max_min - mean_min) / std_min
    sim_min = truncnorm.rvs(a, b, loc=mean_min, scale=std_min, size=n_sim, random_state=rng)
    sim_min = np.clip(sim_min, min_min, max_min)
    idx = rng.integers(0, n, size=n_sim)
    sim_pts = sim_min * df["PPM"].values[idx]
    sim_reb = sim_min * df["RPM"].values[idx]
    sim_ast = sim_min * df["APM"].values[idx]
    return sim_pts, sim_reb, sim_ast


def build_fair_odds_matrix_player(sim_pts: np.ndarray, sim_reb: np.ndarray, sim_ast: np.ndarray, n_sim: int) -> pd.DataFrame:
    rows = []
    for th in PTS_PALIERS:
        prob_pct = float(np.sum(sim_pts >= th) / n_sim) * 100.0
        prob = prob_pct / 100.0
        fair_odd = round(1.0 / prob, 2) if prob > 0 else 999.99
        rows.append({"Stat": "PTS", "Palier": th, "Proba %": round(prob_pct, 2), "Cote juste": fair_odd})
    for th in REB_AST_PALIERS:
        prob_reb = float(np.sum(sim_reb >= th) / n_sim) * 100.0
        fair_reb = round(1.0 / (prob_reb / 100.0), 2) if prob_reb > 0 else 999.99
        rows.append({"Stat": "REB", "Palier": th, "Proba %": round(prob_reb, 2), "Cote juste": fair_reb})
        prob_ast = float(np.sum(sim_ast >= th) / n_sim) * 100.0
        fair_ast = round(1.0 / (prob_ast / 100.0), 2) if prob_ast > 0 else 999.99
        rows.append({"Stat": "AST", "Palier": th, "Proba %": round(prob_ast, 2), "Cote juste": fair_ast})
    return pd.DataFrame(rows)


# ==============================================================================
# STREAMLIT — Terminal
# ==============================================================================


def style_ev(val: float) -> str:
    if val is None or (isinstance(val, str) and val == "-"):
        return ""
    try:
        v = float(val)
        if v < 0:
            return "color: #f85149;"
        if v >= EV_GREEN_THRESHOLD:
            return "color: #3fb950;"
    except (TypeError, ValueError):
        pass
    return ""


def main() -> None:
    st.set_page_config(page_title="Terminal V22", layout="wide", page_icon="🏀")
    st.title("🏀 Terminal V24 — Deep Data (Box Scores + Season Stats)")
    st.caption("Chargement parallèle des stats · Backtest on demand · Time Machine J-1 · Four Factors")

    st.markdown("---")
    with st.expander("📊 Backtest & Vérification (J-1)", expanded=False):
        st.caption("Analyse des matchs terminés (J-1/J-2) avec stats Time Machine. Lourd en API — lancer uniquement si besoin.")
        if st.button("Lancer l'analyse du passé"):
            with st.spinner("Exécution du Backtest sur matchs terminés…"):
                bt_res = run_backtest_yesterday()
            st.session_state["backtest_result"] = bt_res
        bt_res = st.session_state.get("backtest_result")
        if bt_res:
            col_bt1, col_bt2, col_bt3 = st.columns(3)
            matches_seen = bt_res.get("matches_seen", 0)
            col_bt1.metric("Paris simulés (48h)", bt_res["bets"], help=f"Matchs terminés détectés : {matches_seen}")
            wins, bets = bt_res["wins"], bt_res["bets"]
            taux = f"{wins}/{bets}" if bets > 0 else "0/0"
            pct = (wins / bets * 100) if bets > 0 else 0.0
            col_bt2.metric("Taux de réussite", taux, f"{pct:.1f}%")
            profit = bt_res["profit"]
            col_bt3.metric("P&L (unité fixe)", f"{profit:.2f} U", delta_color="off" if profit >= 0 else "normal")
            if matches_seen > 0 and bets == 0:
                st.caption(f"ℹ️ {matches_seen} match(s) terminé(s) détecté(s) — aucun pari (proba 40–60 % ou stats J-1 indisponibles). Voir détail ci-dessous.")
            with st.expander("Voir le détail du Backtest"):
                if bt_res["details"]:
                    for d in bt_res["details"]:
                        st.text(d)
                elif matches_seen == 0:
                    st.info("Aucun match terminé trouvé sur les ligues configurées (Pro B, Euroleague, etc.) pour J-1/J-2.")
                else:
                    st.info("Matchs détectés mais aucun détail enregistré.")
                if bt_res.get("debug_logs"):
                    with st.expander("🔧 Diagnostic Time Machine (DEBUG)", expanded=False):
                        for line in bt_res["debug_logs"]:
                            st.code(line, language=None)
        else:
            st.info("Cliquez sur « Lancer l'analyse du passé » pour exécuter le backtest (consommation API).")
    st.markdown("---")

    with st.spinner("Récupération des matchs (Bulldozer)…"):
        games_with_meta = collect_all_games_72h()
    if not games_with_meta:
        st.warning("Aucun match trouvé sur 72h.")
        return
    st.success(f"{len(games_with_meta)} match(s) chargé(s).")
    n_matches = len(games_with_meta)
    st.caption("Les prédictions et tableaux s'affichent ci-dessous — faites défiler si besoin. Le backtest (J-1) est optionnel : ouvrez l'expander « 📊 Backtest & Vérification » ci-dessus et cliquez sur le bouton pour lancer.")
    st.caption("V25 Sniper : League Profiler · Fair Spread · Money Buckets · Star Impact. Saison 2025 · TURBO (~3–5 s).")

    with st.spinner("🚀 Chargement des stats en parallèle (TURBO)…"):
        stats_cache = preload_all_teams_stats(games_with_meta)
    with st.spinner("Construction des tableaux (ML, O/U, conseils)…"):
        df_ml = build_ml_rows(games_with_meta, stats_cache)
        df_ou = build_ou_table(games_with_meta, stats_cache)
        df_actionable = build_actionable_table(games_with_meta, stats_cache)
        candidates = build_combine_candidates(df_ml, df_ou)
        p1, p2, cote_totale = select_smart_combine(candidates)

    tab_scanner, tab_match = st.tabs(["Scanner Global", "Analyse Match (Deep Dive)"])

    with tab_scanner:
        st.subheader("📋 Tableau actionnable — Heure | Match | Fair Spread | Vainqueur | Proba | Volatilité | Cote | Mise | Fiabilité | Conseil | News Search")
        if not df_actionable.empty:
            st.dataframe(
                df_actionable,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Heure": st.column_config.TextColumn("Heure", width="small"),
                    "Match": st.column_config.TextColumn("Match", width="large"),
                    "Fair Spread": st.column_config.TextColumn("Fair Spread", width="medium", help="Notre prédiction d'écart ; Book = ligne bookmaker si dispo"),
                    "Vainqueur Prédit": st.column_config.TextColumn("Vainqueur Prédit", width="medium"),
                    "Probabilité Ajustée %": st.column_config.NumberColumn("Proba %", format="%.1f"),
                    "Volatilité": st.column_config.NumberColumn("Volatilité", format="%.1f", help="Écart-type pts (5 derniers) ; >15 → −5% proba + ⚠️"),
                    "Cote": st.column_config.NumberColumn("Cote", format="%.2f"),
                    "Mise (%)": st.column_config.TextColumn("Mise", width="small", help="V25 Money Buckets : Safe 1% / Standard 2% / Agressif 3-5%"),
                    "Fiabilité Modèle": st.column_config.TextColumn("Fiabilité", width="medium"),
                    "Conseil": st.column_config.TextColumn("Conseil", width="medium"),
                    "News Search": st.column_config.LinkColumn("News Search", display_text="Rechercher absences X", help="Lien X/Twitter : injury basketball"),
                },
            )
            st.caption("V25 : Fair Spread = écart prédit. Money Buckets : Edge <3% Safe 1% · 3-7% Standard 2% · >7% et 🟢 Agressif 3-5%.")
        else:
            st.info("Aucun match pour le tableau actionnable.")

        st.subheader("⚡ Smart Combiné (algo EV)")
        with st.expander("LE COMBINÉ DU JOUR", expanded=True):
            if p1 and p2 and cote_totale >= MIN_ODD_COMBINE:
                st.markdown(f"**Pari 1:** {p1['label']} — Cote {p1['odd']:.2f} (EV {p1['ev']:.1f}%)")
                st.markdown(f"**Pari 2:** {p2['label']} — Cote {p2['odd']:.2f} (EV {p2['ev']:.1f}%)")
                st.markdown(f"**Cote Totale:** {cote_totale:.2f}")
                gain_100 = 100.0 * (cote_totale - 1.0)
                risque = "Moyen" if cote_totale > 2.0 else "Faible"
                st.markdown(f"**Risque :** {risque} | **Gain Potentiel :** {'Élevé' if cote_totale >= 2.0 else 'Modéré'} (Cote {cote_totale:.2f}) — Pour 100€ : **{gain_100:.2f}€**")
            elif p1:
                st.markdown(f"Un seul pari ≥ {MIN_ODD_COMBINE}: **{p1['label']}** — Cote {p1['odd']:.2f}. Ajoutez un second pari pour le combiné.")
            else:
                st.info(f"Aucun pari avec cote ≥ {MIN_ODD_COMBINE} pour former le combiné.")
            st.caption(f"V22 : Cotes individuelles > {MAX_ODD_SINGLE:.0f} et combinés > {MAX_ODD_COMBINE_TOTAL:.0f} exclus (Small Markets).")

        st.subheader("Tableau Over/Under — Match | Notre Total | Ligne | Cote Over | Cote Under | Fiabilité Modèle | EV%")
        if not df_ou.empty:
            display_ou = df_ou[["Match", "Notre Total", "Ligne Bookmaker", "Cote Over", "Cote Under", "Fiabilité Modèle", "EV%"]].copy()
            st.dataframe(
                display_ou.style.apply(lambda row: [style_ev(row["EV%"])] * len(row), axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Match": st.column_config.TextColumn("Match", width="large"),
                    "Notre Total": st.column_config.TextColumn("Notre Total"),
                    "Ligne Bookmaker": st.column_config.NumberColumn("Ligne", format="%.1f"),
                    "Cote Over": st.column_config.NumberColumn("Cote Over", format="%.2f"),
                    "Cote Under": st.column_config.NumberColumn("Cote Under", format="%.2f"),
                    "Fiabilité Modèle": st.column_config.TextColumn("Fiabilité Modèle"),
                    "EV%": st.column_config.NumberColumn("EV%", format="%.2f"),
                },
            )
            st.caption("Rouge : EV négatif · Vert : EV > 5% · Fiabilité : backtest 3 derniers matchs.")
        else:
            st.info("Aucune donnée Over/Under.")

        st.subheader("Moneyline (Vainqueur)")
        if not df_ml.empty:
            df_ml_main = df_ml[df_ml["Notre Proba"] >= PROBA_MIN_MAIN]
            df_ml_longshots = df_ml[df_ml["Notre Proba"] < PROBA_MIN_MAIN]
            if not df_ml_main.empty:
                st.dataframe(
                    df_ml_main[["Ligue", "Match", "Pari", "Notre Proba", "Fair Odd", "Market Odd", "EDGE %"]].style.apply(
                        lambda row: [style_ev(row["EDGE %"])] * len(row), axis=1
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Paris avec probabilité victoire ≥ 30%. Rouge : EDGE négatif · Vert : EDGE > 5%")
            else:
                st.info("Aucun pari Moneyline avec probabilité ≥ 30%.")
            if not df_ml_longshots.empty:
                with st.expander("Paris Risqués / Longshots (Proba < 30%)", expanded=False):
                    st.dataframe(
                        df_ml_longshots[["Ligue", "Match", "Pari", "Notre Proba", "Fair Odd", "Market Odd", "EDGE %"]].style.apply(
                            lambda row: [style_ev(row["EDGE %"])] * len(row), axis=1
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption("Paris à faible probabilité — à considérer avec prudence.")

    with tab_match:
        st.subheader("Match Vision — Sélectionnez un match")
        seen_gid = set()
        match_options: List[Tuple[str, int, int, str, int, int, str, str]] = []
        if not df_ml.empty:
            for _, r in df_ml.iterrows():
                gid = r.get("_game_id")
                if gid is not None and gid not in seen_gid:
                    seen_gid.add(gid)
                    match_options.append((r["Match"], gid, r["_league_id"], r["_season"], r["_home_id"], r["_away_id"], r["_home_name"], r["_away_name"]))
        if not match_options:
            st.info("Aucun match à analyser.")
        else:
            options_labels = [m[0] for m in match_options]
            sel = st.selectbox("Choisir un match", options_labels, key="match_vision_select")
            if sel:
                idx = next(i for i, m in enumerate(match_options) if m[0] == sel)
                _, game_id, league_id, season, home_id, away_id, home_name, away_name = match_options[idx]
                sh = _get_stats_from_cache(stats_cache, league_id, season, home_id)
                sa = _get_stats_from_cache(stats_cache, league_id, season, away_id)
                proj_h, proj_a, total_proj, is_reliable = pythagorean_projection(sh, sa)
                line_val, _, _ = fetch_odds_totals(game_id, league_id, season)

                st.markdown("**Comparatif Facteurs Clés**")
                verdict, bars = build_factor_bars_and_verdict(sh, sa, home_name, away_name)
                render_factor_bars(bars, home_name, away_name)
                st.markdown("**Le Verdict**")
                st.info(verdict)

                st.markdown("**Tale of the Tape — Head-to-Head (V24 : brut vs Adj. SOS)**")
                tale = build_tale_of_the_tape(league_id, season, home_id, away_id, home_name, away_name, sh=sh, sa=sa)
                df_tale = pd.DataFrame(tale)
                st.dataframe(df_tale, use_container_width=True, hide_index=True, column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Domicile": st.column_config.TextColumn(f"Domicile ({home_name})", width="medium"),
                    "Extérieur": st.column_config.TextColumn(f"Extérieur ({away_name})", width="medium"),
                    "Analyse": st.column_config.TextColumn("Analyse", width="medium"),
                })
                st.caption("Pace en possessions (FIBA ~70). Off. Rtg = pts/100 poss ; Adj. SOS = ajusté au Strength of Schedule.")

                st.markdown("**Simulation du score**")
                if not is_reliable:
                    st.warning("Pas de prédiction fiable (score projeté hors plage 60–120 pts).")
                else:
                    st.caption(f"Projection : (Pace/100)×OffRtg → {home_name} {proj_h:.1f} pts, {away_name} {proj_a:.1f} pts, Total {total_proj:.1f} pts.")
                fig_score = plot_score_simulation(proj_h, proj_a, line_val, home_name, away_name)
                st.plotly_chart(fig_score, use_container_width=True)

                st.markdown("**Four Factors (radar)**")
                fig_radar = plot_four_factors_radar(sh, sa, home_name, away_name)
                st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown("**Zone 3 — Player Props (Monte Carlo)**")
                season_player = season if season in ("2025", "2025-2026") else "2025-2026"
                players_home = fetch_players(home_id, season_player)
                players_away = fetch_players(away_id, season_player)
                team_choice = st.radio("Équipe", [home_name, away_name], key="team_players")
                team_id = home_id if team_choice == home_name else away_id
                players_list = players_home if team_choice == home_name else players_away
                if not players_list:
                    st.caption("Aucun joueur récupéré pour cette équipe (saison 2025 / 2025-2026).")
                else:
                    player_options = [f"{p.get('name', '?')} (ID {p.get('id')})" for p in players_list]
                    player_sel = st.selectbox("Joueur", player_options, key="player_select")
                    if player_sel:
                        pid = int(players_list[player_options.index(player_sel)].get("id", 0))
                        raw_stats = fetch_player_stats(pid, team_id, season_player)
                        if not raw_stats:
                            st.caption("Aucune stat joueur pour cette saison.")
                        else:
                            df_player = build_player_stats_df(raw_stats)
                            if df_player.empty or len(df_player) < 2:
                                st.caption("Pas assez de matchs pour la simulation.")
                            else:
                                sim_pts, sim_reb, sim_ast = run_monte_carlo_player(df_player)
                                if len(sim_pts) > 0:
                                    df_fair = build_fair_odds_matrix_player(sim_pts, sim_reb, sim_ast, len(sim_pts))
                                    st.dataframe(df_fair, use_container_width=True, hide_index=True)
                                    st.caption(f"Saison {season_player} · {len(df_player)} matchs · Monte Carlo {N_SIM} tirages.")


if __name__ == "__main__":
    main()
