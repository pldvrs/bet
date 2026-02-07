"""
Scanner d'Anomalies de Masse — 90% Edge Finder (API Basketball)
================================================================
Ligues et saisons chargés dynamiquement depuis l'API (Betclic, EuroLeague, etc.).
Form 60% last 5 · Pace · Signaux uniquement.
"""

import re
import time
import numpy as np
from datetime import date, timedelta
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import truncnorm

# ==============================================================================
# CONFIGURATION — Source de vérité
# ==============================================================================

LEAGUES: Dict[str, int] = {
    "Betclic Élite (FR)": 2,
    "Pro B (FR)": 3,
    "EuroLeague": 120,
    "LBA Italie": 4,
    "Liga ACB (ESP)": 5,
    "LBL Lettonie": 46,
    "LKL Lituanie": 45,
    "Liban": 134,
    "Islande": 169,
    "Finlande": 165,
    "Suède": 163,
}

# Saison en cours (31 jan 2026 = saison 2025-2026). Doc API : season = YYYY ou YYYY-YYYY
SEASON_API: str = "2025-2026"
LEAGUE_SEASON: Dict[int, str] = {}
BASE_URL: str = "https://v1.basketball.api-sports.io"

BOOKMAKER_IDS: List[int] = [17, 7, 1]
BET_HOME_AWAY_ID: int = 2
BET_TOTALS_IDS: List[int] = [8, 16, 18]  # Over/Under selon fournisseur

PYTHAGOREAN_EXP: float = 13.91
HOME_ADVANTAGE_PTS: float = 3.0
WEIGHT_LAST5: float = 0.6
WEIGHT_SEASON: float = 0.4
LAST_N_GAMES: int = 5
N_SIM: int = 5000
VALUE_BET_THRESHOLD_PCT: float = 65.0
PTS_PALIERS: List[int] = [10, 12, 15, 18, 20, 25]
REB_AST_PALIERS: List[int] = [3, 4, 5, 6, 8, 10]

API_RETRIES: int = 3
API_RETRY_DELAY: float = 1.0
CACHE_TTL: int = 600  # 10 min cache agressif
NEXT_DAYS: int = 8  # Fenêtre pour matchs à venir (si utilisé)
WINDOW_72H_DAYS: int = 3  # Aujourd'hui, Demain, Après-demain — War Room

st.set_page_config(
    page_title="Scanner d'Anomalies de Masse — 90% Edge Finder",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .main .block-container { padding-top: 0.75rem; max-width: 1600px; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #e6edf3 !important; font-weight: 600; }
    .value-detected { color: #3fb950 !important; font-weight: 700; }
    .edge-flash { background-color: #00ff41 !important; color: #0d1117 !important; }
    .edge-dark { background-color: #238636 !important; color: #e6edf3 !important; }
    .triptyque { background: linear-gradient(135deg, #1a472a 0%, #162d20 100%); border: 1px solid #3fb950; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# API — Retry + Cache agressif
# ==============================================================================

def _api_get(
    api_key: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{BASE_URL}/{endpoint}"
    headers = {"x-apisports-key": api_key.strip()}
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


@st.cache_data(ttl=600)
def fetch_seasons(api_key: str) -> List[str]:
    """Liste des saisons disponibles (API). Ex: ['2019-2020', '2024-2025', '2025-2026']."""
    data, err = _api_get(api_key, "seasons", None)
    if err or not data:
        return []
    resp = data.get("response")
    if not isinstance(resp, list):
        return []
    out: List[str] = []
    for s in resp:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, (int, float)):
            out.append(str(int(s)))
    out.sort(key=_season_sort_key)
    return out


def _season_sort_key(s: str) -> tuple:
    """Tri saisons: 2025-2026 avant 2024-2025."""
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a) if a.isdigit() else 0, int(b) if b.isdigit() else 0)
    return (int(s) if s.isdigit() else 0, 0)


@st.cache_data(ttl=600)
def fetch_leagues_for_season(api_key: str, season: str) -> List[dict]:
    """Ligues disponibles pour une saison donnée (API). Chaque item: id, name, country (dict)."""
    data, err = _api_get(api_key, "leagues", {"season": season})
    if err or not data:
        return []
    resp = data.get("response")
    if not isinstance(resp, list):
        return []
    out: List[dict] = []
    for L in resp:
        if not isinstance(L, dict):
            continue
        lid = L.get("id")
        name = L.get("name") or ""
        country = L.get("country") or {}
        cname = country.get("name", "") if isinstance(country, dict) else ""
        if lid is not None and name:
            out.append({"id": int(lid), "name": name, "country": cname})
    return out


def build_league_options_from_hardcoded(api_key: str) -> Tuple[List[dict], str, str]:
    """
    Liste = anciennes ligues (LEAGUES en dur). Saison par ligue = API (current/prev)
    pour que Betclic etc. utilisent la bonne saison (ex. 2024-2025 si pas encore en 2025-2026).
    """
    seasons_list = fetch_seasons(api_key)
    if seasons_list:
        current = seasons_list[-1] if seasons_list else SEASON_API
        prev = seasons_list[-2] if len(seasons_list) >= 2 else current
    else:
        current = SEASON_API
        prev = "2024-2025"
    leagues_curr = fetch_leagues_for_season(api_key, current)
    leagues_prev = fetch_leagues_for_season(api_key, prev) if prev != current else []
    ids_curr = {L["id"] for L in leagues_curr}
    options: List[dict] = []
    for name, lid in LEAGUES.items():
        season = current if lid in ids_curr else prev
        options.append({
            "label": name,
            "id": lid,
            "season": season,
            "name": name,
            "country": "",
        })
    return options, current, prev


@st.cache_data(ttl=120)  # 2 min pour éviter de figer un [] en cache
def fetch_games_by_date(
    api_key: str,
    league_id: int,
    season: str,
    date_str: str,
) -> List[dict]:
    data, err = _api_get(
        api_key, "games",
        {"date": date_str, "league": league_id, "season": season, "timezone": "Europe/Paris"},
    )
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


@st.cache_data(ttl=120)
def fetch_games_by_date_all(
    api_key: str,
    date_str: str,
    season: str,
) -> List[dict]:
    """
    Une seule requête GET : tous les matchs du jour pour une saison (sans filtre ligue).
    Chaque game a league.id, league.name, league.season. On filtre côté client par nos ligues.
    """
    data, err = _api_get(
        api_key, "games",
        {"date": date_str, "season": season, "timezone": "Europe/Paris"},
    )
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def fetch_games_next_n_days(
    api_key: str,
    league_id: int,
    season: str,
    n_days: int = NEXT_DAYS,
) -> List[dict]:
    """Matchs des n prochains jours (date >= aujourd'hui)."""
    today = date.today()
    all_games: List[dict] = []
    seen: set = set()
    for d in range(n_days):
        dte = today + timedelta(days=d)
        date_str = dte.strftime("%Y-%m-%d")
        games = fetch_games_by_date(api_key, league_id, season, date_str)
        for g in games:
            gid = g.get("id")
            if gid is not None and gid not in seen:
                seen.add(gid)
                all_games.append(g)
    return all_games


@st.cache_data(ttl=CACHE_TTL)
def _fetch_odds_raw(
    api_key: str,
    game_id: int,
    league_id: int,
    season: str,
) -> Optional[dict]:
    """Une seule requête odds par match (cache)."""
    data, err = _api_get(
        api_key, "odds",
        {"game": game_id, "league": league_id, "season": season},
    )
    if err or not data:
        return None
    resp = data.get("response")
    if not isinstance(resp, list) or len(resp) == 0:
        return None
    return resp[0]


def fetch_odds(
    api_key: str,
    game_id: int,
    league_id: int,
    season: str,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Récupère les cotes Home/Away. Retourne (odd_home, odd_away, bookmaker_name) ou (None, None, None) si en attente."""
    item = _fetch_odds_raw(api_key, game_id, league_id, season)
    if not item:
        return None, None, None
    bookmakers = item.get("bookmakers") or []
    for bm in bookmakers:
        if bm.get("id") not in BOOKMAKER_IDS:
            continue
        bets = bm.get("bets") or []
        for bet in bets:
            if int(bet.get("id", 0)) != BET_HOME_AWAY_ID:
                continue
            values = bet.get("values") or []
            odd_home = odd_away = None
            for v in values:
                val = (v.get("value") or "").strip().lower()
                odd_str = v.get("odd")
                if odd_str is not None:
                    try:
                        odd_f = float(odd_str)
                        if val == "home":
                            odd_home = odd_f
                        elif val == "away":
                            odd_away = odd_f
                    except (TypeError, ValueError):
                        pass
            if odd_home is not None and odd_away is not None:
                return odd_home, odd_away, (bm.get("name") or f"ID {bm.get('id')}")
    for bm in bookmakers:
        bets = bm.get("bets") or []
        for bet in bets:
            if int(bet.get("id", 0)) != BET_HOME_AWAY_ID:
                continue
            values = bet.get("values") or []
            odd_home = odd_away = None
            for v in values:
                val = (v.get("value") or "").strip().lower()
                odd_str = v.get("odd")
                if odd_str is not None:
                    try:
                        odd_f = float(odd_str)
                        if val == "home":
                            odd_home = odd_f
                        elif val == "away":
                            odd_away = odd_f
                    except (TypeError, ValueError):
                        pass
            if odd_home is not None and odd_away is not None:
                return odd_home, odd_away, (bm.get("name") or "Marché")
    return None, None, None


def fetch_odds_totals(
    api_key: str,
    game_id: int,
    league_id: int,
    season: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """(line, odd_over, odd_under, bookmaker) ou (None, None, None, None) si en attente de cotes."""
    item = _fetch_odds_raw(api_key, game_id, league_id, season)
    if not item:
        return None, None, None, None
    bookmakers = item.get("bookmakers") or []
    for bm in bookmakers:
        bets = bm.get("bets") or []
        for bet in bets:
            bid = int(bet.get("id", 0))
            name = (bet.get("name") or "").lower()
            if bid not in BET_TOTALS_IDS and "total" not in name and "over" not in name and "points" not in name:
                continue
            values = bet.get("values") or []
            line = odd_over = odd_under = None
            for v in values:
                val_raw = (v.get("value") or "").strip()
                val = val_raw.upper()
                odd_str = v.get("odd")
                try:
                    if "OVER" in val or val == "O":
                        odd_over = float(odd_str) if odd_str else None
                    elif "UNDER" in val or val == "U":
                        odd_under = float(odd_str) if odd_str else None
                    num_match = re.search(r"[\d]+[.,]?\d*", val_raw)
                    if num_match and line is None:
                        line = float(num_match.group(0).replace(",", "."))
                except (TypeError, ValueError):
                    pass
            if (odd_over is not None or odd_under is not None) and (odd_over or odd_under):
                if line is None and values:
                    first_val = str(values[0].get("value") or "")
                    num_match = re.search(r"[\d]+[.,]?\d*", first_val)
                    if num_match:
                        line = float(num_match.group(0).replace(",", "."))
                return line or 0.0, odd_over, odd_under, (bm.get("name") or "Marché")
    return None, None, None, None


@st.cache_data(ttl=CACHE_TTL)
def fetch_team_games(
    api_key: str,
    league_id: int,
    season: str,
    team_id: int,
) -> List[dict]:
    """Derniers matchs de l'équipe (pour forme last 5)."""
    data, err = _api_get(
        api_key, "games",
        {"league": league_id, "season": season, "team": team_id},
    )
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
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        pts_h = _safe_int(scores.get("home"))
        pts_a = _safe_int(scores.get("away"))
        if home_id == team_id:
            out.append({"date": g.get("date"), "pts_for": pts_h, "pts_against": pts_a})
        elif away_id == team_id:
            out.append({"date": g.get("date"), "pts_for": pts_a, "pts_against": pts_h})
    out.sort(key=lambda x: x.get("date") or "", reverse=True)
    return out[:LAST_N_GAMES]


@st.cache_data(ttl=CACHE_TTL)
def fetch_team_statistics(
    api_key: str,
    league_id: int,
    season: str,
    team_id: int,
) -> Tuple[Optional[dict], Optional[str]]:
    data, err = _api_get(
        api_key, "statistics",
        {"league": league_id, "season": season, "team": team_id},
    )
    if err or not data:
        return None, err
    resp = data.get("response")
    if isinstance(resp, dict):
        return resp, None
    if isinstance(resp, list) and len(resp) > 0:
        return resp[0], None
    return None, "Pas de stats"


@st.cache_data(ttl=300)
def fetch_players(api_key: str, team_id: int, season: str) -> List[dict]:
    data, err = _api_get(api_key, "players", {"team": team_id, "season": season})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def fetch_player_stats(
    api_key: str,
    player_id: int,
    team_id: int,
    season: str,
) -> Tuple[List[dict], Optional[str]]:
    data, err = _api_get(
        api_key, "games/statistics/players",
        {"player": player_id, "season": season, "team": team_id},
    )
    if err:
        if "team" in (err or "").lower():
            data2, err2 = _api_get(
                api_key, "games/statistics/players",
                {"player": player_id, "season": season},
            )
            if not err2 and data2:
                resp = data2.get("response")
                return (resp if isinstance(resp, list) else []), None
        return [], err
    resp = data.get("response") if data else []
    return (resp if isinstance(resp, list) else []), None


# ==============================================================================
# PARSING JOUEUR — Minutes "MM:SS" → float, points, rebounds.total, assists
# ==============================================================================

def _minutes_mmss_to_float(minutes_val: Any) -> float:
    if minutes_val is None:
        return 0.0
    if isinstance(minutes_val, (int, float)):
        return max(0.0, float(minutes_val))
    if not isinstance(minutes_val, str) or not minutes_val.strip():
        return 0.0
    s = minutes_val.strip()
    if s in ("0", "0:00", "00:00"):
        return 0.0
    m = re.match(r"^(\d+):(\d+)$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    try:
        return max(0.0, float(s))
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
    min_f = _minutes_mmss_to_float(raw_row.get("minutes"))
    if min_f <= 0:
        return None
    pts = _int_or_total(raw_row.get("points"), 0)
    reb = _int_or_total(raw_row.get("rebounds"), 0)
    ast = _int_or_total(raw_row.get("assists"), 0)
    ppm = pts / min_f if min_f > 0 else 0.0
    rpm = reb / min_f if min_f > 0 else 0.0
    apm = ast / min_f if min_f > 0 else 0.0
    return {"MIN": min_f, "PTS": pts, "REB": reb, "AST": ast, "PPM": ppm, "RPM": rpm, "APM": apm}


def build_player_stats_dataframe(raw_stats: List[dict]) -> pd.DataFrame:
    rows = []
    for raw_row in raw_stats:
        row = _extract_game_stats(raw_row)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["MIN", "PTS", "REB", "AST", "PPM", "RPM", "APM"])
    return pd.DataFrame(rows)


def run_monte_carlo_player(
    df: pd.DataFrame,
    n_sim: int = N_SIM,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_fair_odds_matrix_player(
    sim_pts: np.ndarray,
    sim_reb: np.ndarray,
    sim_ast: np.ndarray,
    n_sim: int,
) -> pd.DataFrame:
    rows = []
    for th in PTS_PALIERS:
        prob_pct = float(np.sum(sim_pts >= th) / n_sim) * 100.0
        prob = prob_pct / 100.0
        fair_odd = round(1.0 / prob, 2) if prob > 0 else 999.99
        rows.append({"Stat": "PTS", "Palier": th, "Proba %": round(prob_pct, 2), "Cote juste": fair_odd, "Value Bet": prob_pct > VALUE_BET_THRESHOLD_PCT})
    for th in REB_AST_PALIERS:
        prob_reb = float(np.sum(sim_reb >= th) / n_sim) * 100.0
        fair_reb = round(1.0 / (prob_reb / 100.0), 2) if prob_reb > 0 else 999.99
        rows.append({"Stat": "REB", "Palier": th, "Proba %": round(prob_reb, 2), "Cote juste": fair_reb, "Value Bet": prob_reb > VALUE_BET_THRESHOLD_PCT})
        prob_ast = float(np.sum(sim_ast >= th) / n_sim) * 100.0
        fair_ast = round(1.0 / (prob_ast / 100.0), 2) if prob_ast > 0 else 999.99
        rows.append({"Stat": "AST", "Palier": th, "Proba %": round(prob_ast, 2), "Cote juste": fair_ast, "Value Bet": prob_ast > VALUE_BET_THRESHOLD_PCT})
    return pd.DataFrame(rows)


# ==============================================================================
# TEAM STATS
# ==============================================================================

def _safe_int(val: Any, default: int = 0) -> int:
    """Extrait un int même si l'API renvoie un dict (ex. games.played)."""
    if val is None:
        return default
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, float):
        return max(0, int(val))
    if isinstance(val, dict):
        return _safe_int(val.get("total") or val.get("all") or val.get("played"), default)
    if isinstance(val, str):
        try:
            return max(0, int(float(val)))
        except (TypeError, ValueError):
            return default
    return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Extrait un float même si l'API renvoie un dict."""
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


def _team_stats_from_response(raw: dict) -> dict:
    """PPG, OPPG, Pace + Four Factors (eFG%, TOV%, ORB%, FT Rate) si dispo dans l'API."""
    games = raw.get("games") or {}
    played = max(1, _safe_int(games.get("played")) or _safe_int(games.get("all")) or 1)
    points = raw.get("points") or {}
    for_pts = points.get("for") or {}
    against_pts = points.get("against") or {}
    ppg = _safe_float(for_pts.get("average")) or (_safe_int(for_pts.get("total")) / played if played else 0.0)
    oppg = _safe_float(against_pts.get("average")) or (_safe_int(against_pts.get("total")) / played if played else 0.0)
    if ppg <= 0:
        ppg = 75.0
    if oppg <= 0:
        oppg = 75.0
    out: dict = {"ppg": ppg, "oppg": oppg, "pace": ppg + oppg, "efg_pct": 0.5, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25}
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


def get_team_stats_form_weighted(
    api_key: str,
    league_id: int,
    season: str,
    team_id: int,
) -> dict:
    """Stats saison + 60% last 5. Retourne ppg, oppg, pace + Four Factors (efg_pct, orb_pct, etc.)."""
    raw, _ = fetch_team_statistics(api_key, league_id, season, team_id)
    if raw is None:
        return {"ppg": 75.0, "oppg": 75.0, "pace": 150.0, "efg_pct": 0.5, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.25}
    season_stats = _team_stats_from_response(raw)
    last5 = fetch_team_games(api_key, league_id, season, team_id)
    last5 = [x for x in last5 if (x.get("pts_for") or 0) > 0 or (x.get("pts_against") or 0) > 0]
    if len(last5) < 2:
        return season_stats
    ppg_l5 = sum(x.get("pts_for", 0) for x in last5) / len(last5)
    oppg_l5 = sum(x.get("pts_against", 0) for x in last5) / len(last5)
    ppg = WEIGHT_SEASON * season_stats["ppg"] + WEIGHT_LAST5 * ppg_l5
    oppg = WEIGHT_SEASON * season_stats["oppg"] + WEIGHT_LAST5 * oppg_l5
    if ppg <= 0:
        ppg = 75.0
    if oppg <= 0:
        oppg = 75.0
    out = {"ppg": ppg, "oppg": oppg, "pace": ppg + oppg}
    for k in ("efg_pct", "tov_pct", "orb_pct", "ft_rate"):
        out[k] = season_stats.get(k, 0.25 if k == "orb_pct" else 0.5 if k == "efg_pct" else 0.15 if k == "tov_pct" else 0.25)
    return out


# ==============================================================================
# PRÉDICTION ÉQUIPE — Loi de Pythagore (exposant 13.91) + Score projeté
# ==============================================================================

def pythagorean_win_prob_and_projection(
    stats_home: dict,
    stats_away: dict,
    home_advantage_pts: float = HOME_ADVANTAGE_PTS,
    exp: float = PYTHAGOREAN_EXP,
) -> Tuple[float, float, float, float]:
    """Retourne (win_prob_home, proj_home, proj_away, total_proj)."""
    ppg_h = stats_home.get("ppg", 75.0)
    oppg_h = stats_home.get("oppg", 75.0)
    ppg_a = stats_away.get("ppg", 75.0)
    oppg_a = stats_away.get("oppg", 75.0)
    proj_home = (ppg_h + oppg_a) / 2.0 + home_advantage_pts
    proj_away = (ppg_a + oppg_h) / 2.0
    denom = (proj_home ** exp) + (proj_away ** exp)
    if denom <= 0:
        return 0.5, proj_home, proj_away, proj_home + proj_away
    win_prob_home = (proj_home ** exp) / denom
    return float(win_prob_home), float(proj_home), float(proj_away), float(proj_home + proj_away)


# ==============================================================================
# SCAN SIGNALS — Form 60% last 5, Pace, Score de confiance, Triptyque
# ==============================================================================

def _confidence_score(ev_pct: float, total_proj: float, line: Optional[float], pace_h: float, pace_a: float) -> float:
    """Score 0-100 : écart cote fair/marché + écart total projeté/ligne. Pace élevé renforce Over."""
    base = min(100, max(0, 50 + ev_pct * 0.8))
    if line and line > 0:
        edge_total = (total_proj - line) / line * 100
        base = min(100, max(0, 50 + ev_pct * 0.4 + edge_total * 0.5))
        if pace_h + pace_a > 320:  # deux équipes à haut rythme -> Over prioritaire
            base = min(100, base + 10)
    return round(base, 1)


def build_scan_signals(
    api_key: str,
    league_options: List[dict],
    date_str: str,
    current_season: Optional[str] = None,
    prev_season: Optional[str] = None,
) -> Tuple[List[dict], List[dict], List[dict], set]:
    """
    Retourne (signals, games_with_meta, triptyque top 3, leagues_no_match).
    1) Requête consolidée: GET tous les matchs du jour (2 saisons), filtre par nos ligues.
    2) Si rien: fallback requête par ligue en essayant current + prev season.
    """
    current_season = current_season or SEASON_API
    prev_season = prev_season or "2024-2025"
    selected_ids = {o["id"] for o in league_options}
    opt_by_id = {o["id"]: o for o in league_options}
    signals: List[dict] = []
    games_with_meta: List[dict] = []
    seen_gid: set = set()
    leagues_no_match: set = set()
    games_to_process: List[Tuple[dict, int, str, str]] = []  # (game, league_id, league_name, season)

    # 1) Consolidé: une requête par saison (sans filtre ligue), puis filtre côté client
    for season in (current_season, prev_season):
        try:
            for g in fetch_games_by_date_all(api_key, date_str, season):
                lid = (g.get("league") or {}).get("id")
                gid = g.get("id")
                if gid is None or lid is None or lid not in selected_ids or gid in seen_gid:
                    continue
                seen_gid.add(gid)
                opt = opt_by_id.get(lid)
                league_name = (opt["label"] if opt else (g.get("league") or {}).get("name") or str(lid))
                s = (opt["season"] if opt else (g.get("league") or {}).get("season") or season)
                games_to_process.append((g, int(lid), league_name, s))
        except Exception:
            pass

    # 2) Fallback: si rien en consolidé, requête par ligue (2 saisons par ligue)
    if not games_to_process:
        for opt in league_options:
            league_id = opt.get("id")
            league_name = opt.get("label") or opt.get("name") or str(league_id)
            season = opt.get("season") or SEASON_API
            if league_id is None:
                continue
            found = False
            for try_season in (season, prev_season):
                if try_season == season and season == prev_season:
                    pass  # once
                try:
                    games = fetch_games_by_date(api_key, league_id, try_season, date_str)
                    for g in games:
                        gid = g.get("id")
                        if gid is None or gid in seen_gid:
                            continue
                        seen_gid.add(gid)
                        games_to_process.append((g, league_id, league_name, try_season))
                        found = True
                    if games:
                        break
                except Exception:
                    pass
            if not found:
                leagues_no_match.add(league_name)

    # 3) Traiter chaque match (consolidé ou fallback)
    for g, league_id, league_name, season in games_to_process:
        gid = g.get("id")
        if gid is None:
            continue
        teams_g = g.get("teams") or {}
        home = teams_g.get("home") or {}
        away = teams_g.get("away") or {}
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        home_name = home.get("name", "Domicile") if isinstance(home, dict) else "Domicile"
        away_name = away.get("name", "Extérieur") if isinstance(away, dict) else "Extérieur"
        if not home_id or not away_id:
            continue
        sh = get_team_stats_form_weighted(api_key, league_id, season, home_id)
        sa = get_team_stats_form_weighted(api_key, league_id, season, away_id)
        win_h, proj_h, proj_a, total_proj = pythagorean_win_prob_and_projection(sh, sa)
        fair_home = round(1.0 / win_h, 2) if win_h > 0 else 999.99
        fair_away = round(1.0 / (1.0 - win_h), 2) if win_h < 1 else 999.99
        pace_h = sh.get("pace", sh["ppg"] + sh["oppg"])
        pace_a = sa.get("pace", sa["ppg"] + sa["oppg"])
        orb_h = sh.get("orb_pct", 0.25)
        orb_a = sa.get("orb_pct", 0.25)
        badge = " 💎 AVANTAGE PHYSIQUE" if (orb_h > orb_a + 0.05) else ""
        odd_home, odd_away, bm_name = fetch_odds(api_key, int(gid), league_id, season)
        line, odd_over, odd_under, _ = fetch_odds_totals(api_key, int(gid), league_id, season)
        meta = {
            "game": g,
            "league_id": league_id,
            "league_name": league_name,
            "season": season,
            "home_id": home_id,
            "away_id": away_id,
            "home_name": home_name,
            "away_name": away_name,
            "proj_h": proj_h,
            "proj_a": proj_a,
            "fair_home": fair_home,
            "fair_away": fair_away,
            "total_proj": total_proj,
        }
        games_with_meta.append(meta)
        cote_display = "En attente de cotes"
        base_match = f"{home_name} vs {away_name}{badge}"
        if odd_home is not None and fair_home < 999:
            ev_pct = ((odd_home - fair_home) / fair_home) * 100.0
            conf = _confidence_score(ev_pct, total_proj, line, pace_h, pace_a)
            signals.append({
                "Date": date_str,
                "Ligue": league_name,
                "Match": base_match,
                "Pari conseillé": home_name,
                "Probabilité %": round(win_h * 100, 1),
                "Cote": odd_home,
                "Expected Value %": round(ev_pct, 1),
                "Confidence": conf,
                "_game_id": gid,
                "_league_id": league_id,
                "_meta": meta,
            })
        else:
            signals.append({
                "Date": date_str,
                "Ligue": league_name,
                "Match": base_match,
                "Pari conseillé": home_name,
                "Probabilité %": round(win_h * 100, 1),
                "Cote": cote_display,
                "Expected Value %": 0.0,
                "Confidence": 50.0,
                "_game_id": gid,
                "_league_id": league_id,
                "_meta": meta,
            })
        if odd_away is not None and fair_away < 999:
            ev_pct = ((odd_away - fair_away) / fair_away) * 100.0
            conf = _confidence_score(ev_pct, total_proj, line, pace_h, pace_a)
            signals.append({
                "Date": date_str,
                "Ligue": league_name,
                "Match": base_match,
                "Pari conseillé": away_name,
                "Probabilité %": round((1 - win_h) * 100, 1),
                "Cote": odd_away,
                "Expected Value %": round(ev_pct, 1),
                "Confidence": conf,
                "_game_id": gid,
                "_league_id": league_id,
                "_meta": meta,
            })
        else:
            signals.append({
                "Date": date_str,
                "Ligue": league_name,
                "Match": base_match,
                "Pari conseillé": away_name,
                "Probabilité %": round((1 - win_h) * 100, 1),
                "Cote": cote_display,
                "Expected Value %": 0.0,
                "Confidence": 50.0,
                "_game_id": gid,
                "_league_id": league_id,
                "_meta": meta,
            })
        if line is not None and (pace_h + pace_a) > 300 and (odd_over or odd_under):
            over_ev = ((total_proj - line) / line) * 100 if line else 0
            odd_ou = odd_over if odd_over is not None else odd_under
            if odd_ou:
                fair_prob_over = min(0.99, max(0.01, (total_proj - line) / 20 + 0.5))
                ev_ou = ((odd_ou - 1 / fair_prob_over) / (1 / fair_prob_over)) * 100
                conf_ou = _confidence_score(ev_ou, total_proj, line, pace_h, pace_a)
                pari_ou = f"Over {line:.1f}"
                signals.append({
                    "Date": date_str,
                    "Ligue": league_name,
                    "Match": base_match,
                    "Pari conseillé": pari_ou,
                    "Probabilité %": round(max(50, 50 + over_ev), 1),
                    "Cote": odd_ou,
                    "Expected Value %": round(ev_ou, 1),
                    "Confidence": conf_ou,
                    "_game_id": gid,
                    "_league_id": league_id,
                    "_meta": meta,
                })
    signals.sort(key=lambda x: (x["Confidence"], x["Expected Value %"]), reverse=True)
    by_game: Dict[Any, dict] = {}
    for s in signals:
        gid = s["_game_id"]
        if gid not in by_game or s["Confidence"] > by_game[gid]["Confidence"]:
            by_game[gid] = s
    triptyque = sorted(by_game.values(), key=lambda x: x["Confidence"], reverse=True)[:3]
    return signals, games_with_meta, triptyque, leagues_no_match


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.title("🚀 Scanner d'Anomalies de Masse")
st.sidebar.caption("90% Edge Finder · Ligues mineures")

api_key = st.sidebar.text_input(
    "Clé API (x-apisports-key)",
    type="password",
    placeholder="Clé API",
    key="api_key",
)

if not api_key or not api_key.strip():
    st.warning("Entrez votre clé API dans la barre latérale.")
    st.stop()

api_key = api_key.strip()

# Anciennes ligues (LEAGUES en dur) — saison par ligue via API (current/prev)
league_options, current_season, prev_season = build_league_options_from_hardcoded(api_key)
league_labels = list(LEAGUES.keys())
default_league_names = ["Betclic Élite (FR)", "Pro B (FR)", "EuroLeague", "LBL Lettonie", "Liban", "Islande"]

selected_labels = st.sidebar.multiselect(
    "Ligues à inclure",
    league_labels,
    default=default_league_names,
    key="leagues",
    help="Betclic Élite, Pro B, EuroLeague, LBA, ACB, LBL Lettonie, LKL, Liban, Islande, Finlande, Suède.",
)

selected_league_options = [o for o in league_options if o["label"] in selected_labels]
if not selected_league_options:
    st.warning("Sélectionnez au moins une ligue dans la barre latérale.")
    st.stop()

st.sidebar.caption(f"Saison(s) API : {current_season} · {len(league_labels)} ligues")

# Scanner 72h : Aujourd'hui, Demain, Après-demain — War Room (une seule liste fusionnée)
today = date.today()
signals_all: List[dict] = []
games_all: List[dict] = []
seen_gid_72h: set = set()
seen_game_72h: set = set()
leagues_no_match_72h: set = set()

for d in range(WINDOW_72H_DAYS):
    day_date = today + timedelta(days=d)
    date_str = day_date.strftime("%Y-%m-%d")
    try:
        signals, games_with_meta, _, leagues_no_match = build_scan_signals(
            api_key, selected_league_options, date_str, current_season, prev_season
        )
    except Exception:
        signals, games_with_meta, leagues_no_match = [], [], {o["label"] for o in selected_league_options}
    leagues_no_match_72h |= leagues_no_match
    for s in signals:
        if s["_game_id"] not in seen_gid_72h:
            seen_gid_72h.add(s["_game_id"])
            signals_all.append(s)
    for g in games_with_meta:
        gid = g.get("game", {}).get("id") if isinstance(g.get("game"), dict) else None
        if gid is not None and gid not in seen_game_72h:
            seen_game_72h.add(gid)
            games_all.append(g)

signals_all.sort(key=lambda x: (x["Expected Value %"], x["Confidence"]), reverse=True)
by_game_72h: Dict[Any, dict] = {}
for s in signals_all:
    gid = s["_game_id"]
    if gid not in by_game_72h or s["Confidence"] > by_game_72h[gid]["Confidence"]:
        by_game_72h[gid] = s
triptyque = sorted(by_game_72h.values(), key=lambda x: x["Confidence"], reverse=True)[:3]
signals = signals_all
games_with_meta = games_all
leagues_no_match_72h = leagues_no_match_72h - {s["Ligue"] for s in signals_all}

# ==============================================================================
# 🚀 LE TRIPTYQUE DE CONFIANCE — Top 3 pour combiné
# ==============================================================================

st.header("🚀 LE TRIPTYQUE DE CONFIANCE")
st.caption(f"Fenêtre 72h (Aujourd'hui, Demain, Après-demain) · Saison(s) API {current_season} · Top 3 Value tout continent")

if triptyque:
    cols = st.columns(3)
    for i, pick in enumerate(triptyque):
        with cols[i]:
            st.markdown('<div class="triptyque">', unsafe_allow_html=True)
            st.metric("Match", pick["Match"])
            st.metric("Ligue", pick["Ligue"])
            st.metric("Pari conseillé", pick["Pari conseillé"])
            st.metric("Probabilité %", f"{pick['Probabilité %']} %")
            cote_val = pick.get("Cote")
            st.metric("Cote", str(cote_val) if isinstance(cote_val, (int, float)) else cote_val)
            st.metric("Expected Value %", f"{pick['Expected Value %']:+.1f} %")
            st.metric("Score confiance", f"{pick['Confidence']:.0f}/100")
    st.caption("Combiné conseillé : les 3 paris ci-dessus (même match = un seul pari par match).")
else:
    st.info("Aucun match avec cotes + stats sur 72h. Les cotes peuvent être en attente.")

# État des ligues (pas de crash : affiche "Pas de match programmé")
with st.expander("État des ligues (72h)", expanded=False):
    if leagues_no_match_72h:
        for league_name in sorted(leagues_no_match_72h):
            st.caption(f"**{league_name}** : Pas de match programmé")
    else:
        st.caption("Toutes les ligues sélectionnées ont au moins un match sur 72h.")

# ==============================================================================
# DASHBOARD SIGNALS ONLY — [Date | Ligue | Match | Pari conseillé | Prob % | Cote | EV%]
# ==============================================================================

st.header("War Room · Tous les matchs (72h)")
st.caption("Tri par Edge %. Vert flash : Edge > 20% · Vert sombre : Edge > 10% · Badge 💎 = Avantage physique")

display_cols = ["Date", "Ligue", "Match", "Pari conseillé", "Probabilité %", "Cote", "Expected Value %"]

if signals:
    df_display = pd.DataFrame([{k: o[k] for k in display_cols} for o in signals])
    def _row_style(row):
        ev = row.get("Expected Value %", 0)
        if isinstance(ev, str):
            return [""] * len(row)
        if ev > 20:
            return ["background-color: #00ff41; color: #0d1117"] * len(row)
        if ev > 10:
            return ["background-color: #238636; color: #e6edf3"] * len(row)
        return [""] * len(row)
    st.dataframe(
        df_display.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Ligue": st.column_config.TextColumn("Ligue", width="small"),
            "Match": st.column_config.TextColumn("Match", width="medium"),
            "Pari conseillé": st.column_config.TextColumn("Pari conseillé", width="small"),
            "Probabilité %": st.column_config.NumberColumn("Probabilité %", format="%.1f"),
            "Cote": st.column_config.TextColumn("Cote"),
            "Expected Value %": st.column_config.NumberColumn("Expected Value %", format="%.1f"),
        },
    )
    st.caption("Cote = « En attente de cotes » si le marché n'a pas encore publié.")
else:
    st.info("Aucun signal sur 72h dans les ligues sélectionnées.")

# ==============================================================================
# DÉTAIL D'UN MATCH (Deep Dive) — Expander pour garder l'écran propre
# ==============================================================================

if not games_with_meta:
    st.info("Aucun match à afficher sur 72h. Changez les ligues.")
else:
    with st.expander("Deep Dive · Détail match & simulations Monte Carlo", expanded=False):
        st.caption("Choisissez un match pour voir la comparaison cotes et les simulations par joueur.")
        game_options = [f"{m['home_name']} vs {m['away_name']} · {m['league_name']}" for m in games_with_meta]
        game_idx = st.selectbox("Choisir un match", range(len(game_options)), format_func=lambda i: game_options[i], key="game_detail")
        meta = games_with_meta[game_idx]
        league_id = meta["league_id"]
        home_id = meta["home_id"]
        away_id = meta["away_id"]
        home_name = meta["home_name"]
        away_name = meta["away_name"]
        proj_home = meta["proj_h"]
        proj_away = meta["proj_a"]
        fair_home = meta["fair_home"]
        fair_away = meta["fair_away"]
        gid = int(meta["game"].get("id", 0))
        total_proj = proj_home + proj_away
        win_prob_home = 1.0 / fair_home if fair_home < 999 else 0.5

        season_match = meta.get("season") or current_season or SEASON_API
        odd_home, odd_away, bookmaker_name = fetch_odds(api_key, gid, league_id, season_match)
        cote_marche_h = odd_home
        cote_marche_a = odd_away
        if cote_marche_h is None:
            cote_marche_h = "En attente de cotes"
        if cote_marche_a is None:
            cote_marche_a = "En attente de cotes"
        edge_h = ((cote_marche_h - fair_home) / fair_home * 100) if (isinstance(cote_marche_h, (int, float)) and fair_home < 999) else 0.0
        edge_a = ((cote_marche_a - fair_away) / fair_away * 100) if (isinstance(cote_marche_a, (int, float)) and fair_away < 999) else 0.0

        rows = [
            {"Équipe": home_name, "Notre Cote (Fair)": fair_home, "Cote Marché": cote_marche_h, "Edge %": round(edge_h, 1)},
            {"Équipe": away_name, "Notre Cote (Fair)": fair_away, "Cote Marché": cote_marche_a, "Edge %": round(edge_a, 1)},
        ]
        df_compare = pd.DataFrame(rows)
        st.dataframe(
            df_compare,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Équipe": st.column_config.TextColumn("Équipe", width="medium"),
                "Notre Cote (Fair)": st.column_config.NumberColumn("Notre Cote (Fair)", format="%.2f"),
                "Cote Marché": st.column_config.TextColumn("Cote Marché"),
                "Edge %": st.column_config.NumberColumn("Edge %", format="%.1f"),
            },
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score projeté", f"{proj_home:.0f} — {proj_away:.0f}", f"Total {total_proj:.0f}")
        with col2:
            st.metric("Prob. victoire Domicile", f"{win_prob_home*100:.1f} %", f"Cote juste {fair_home}")
            if isinstance(cote_marche_h, (int, float)) and fair_home < 999 and cote_marche_h > fair_home:
                st.markdown('<p class="value-detected">✅ VALUE DETECTED — Domicile</p>', unsafe_allow_html=True)
        with col3:
            st.metric("Prob. victoire Extérieur", f"{(1-win_prob_home)*100:.1f} %", f"Cote juste {fair_away}")
            if isinstance(cote_marche_a, (int, float)) and fair_away < 999 and cote_marche_a > fair_away:
                st.markdown('<p class="value-detected">✅ VALUE DETECTED — Extérieur</p>', unsafe_allow_html=True)

        if bookmaker_name:
            st.caption(f"Cotes marché : {bookmaker_name}")

        with st.expander("🔬 Analyse joueur (Fair Value PTS/REB/AST)", expanded=False):
            st.caption("Choisissez une équipe puis un joueur pour la matrice des cotes justes (Monte Carlo) et les distributions simulées.")
            team_choice = st.radio("Équipe", [home_name, away_name], horizontal=True, key="deep_team")
            team_id = home_id if team_choice == home_name else away_id
            players_list = fetch_players(api_key, team_id, season_match)
            if not players_list:
                st.info(f"Aucun joueur récupéré pour {team_choice}. Vérifiez la saison/ligue.")
            else:
                player_options = {f"{p.get('name', '?')} (id {p.get('id', '')})": p for p in players_list}
                player_label = st.selectbox("Joueur", list(player_options.keys()), key="deep_player")
                selected_player = player_options.get(player_label)
                if selected_player:
                    player_id = int(selected_player.get("id", 0))
                    raw_stats, err_stats = fetch_player_stats(api_key, player_id, team_id, season_match)
                    if err_stats:
                        st.warning(f"Stats joueur : {err_stats}")
                    elif not raw_stats:
                        st.info("Aucune statistique de match pour ce joueur cette saison.")
                    else:
                        df_player = build_player_stats_dataframe(raw_stats)
                        if df_player.empty or len(df_player) < 2:
                            st.info("Pas assez de matchs avec minutes/stats pour lancer le Monte Carlo (min. 2 matchs).")
                        else:
                            sim_pts, sim_reb, sim_ast = run_monte_carlo_player(df_player, n_sim=N_SIM)
                            n_sim_actual = len(sim_pts)
                            if n_sim_actual == 0:
                                st.warning("Simulation Monte Carlo impossible.")
                            else:
                                df_fair = build_fair_odds_matrix_player(sim_pts, sim_reb, sim_ast, n_sim_actual)
                                st.subheader("Matrice des cotes justes (Value Bet si Proba > 65 %)")
                                st.dataframe(
                                    df_fair.style.apply(
                                        lambda row: ["background-color: rgba(35,134,54,0.25)" if row["Value Bet"] else "" for _ in row],
                                        axis=1,
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.subheader("Distributions simulées")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    fig_pts = go.Figure(go.Histogram(x=sim_pts, nbinsx=40, marker_color="#58a6ff", name="PTS"))
                                    fig_pts.update_layout(title="PTS simulés", template="plotly_dark", height=220, showlegend=False, margin=dict(t=40,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)")
                                    st.plotly_chart(fig_pts, use_container_width=True)
                                with col2:
                                    fig_reb = go.Figure(go.Histogram(x=sim_reb, nbinsx=40, marker_color="#3fb950", name="REB"))
                                    fig_reb.update_layout(title="REB simulés", template="plotly_dark", height=220, showlegend=False, margin=dict(t=40,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)")
                                    st.plotly_chart(fig_reb, use_container_width=True)
                                with col3:
                                    fig_ast = go.Figure(go.Histogram(x=sim_ast, nbinsx=40, marker_color="#d29922", name="AST"))
                                    fig_ast.update_layout(title="AST simulés", template="plotly_dark", height=220, showlegend=False, margin=dict(t=40,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)")
                                    st.plotly_chart(fig_ast, use_container_width=True)
                                avg_pts, avg_reb, avg_ast = float(np.mean(sim_pts)), float(np.mean(sim_reb)), float(np.mean(sim_ast))
                                st.caption(f"Saison {season_match} · {len(df_player)} matchs · Moyennes simulées : PTS {avg_pts:.1f} · REB {avg_reb:.1f} · AST {avg_ast:.1f}")
