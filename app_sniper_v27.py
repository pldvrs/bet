#!/usr/bin/env python3
"""
Terminal V27 — ML First (Probabilité calibrée + Edge > 5% = 🎯 SNIPER TARGET)
=============================================================================
Moteur principal : Machine Learning (model_proba.pkl + model_spread.pkl).
Probabilité calibrée utilisée pour les calculs de mise (avantage réel).
Si modèles absents ou données insuffisantes → Mode Secours (Pythagore).
"""

import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from database import get_client

try:
    from archetype_engine import get_tactical_clash
except Exception:
    get_tactical_clash = None

try:
    from training_engine import get_trap_info, get_match_style, predict_upset_proba, predict_total_points
except Exception:
    get_trap_info = None
    get_match_style = None
    predict_upset_proba = None
    predict_total_points = None

# Chemins des modèles ML (alignés sur training_engine)
# Antitrap = nouvel entraînement ; ancien = model_proba.pkl etc. (non écrasé)
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PROBA_PATH = SCRIPT_DIR / "model_proba.pkl"
MODEL_SPREAD_PATH = SCRIPT_DIR / "model_spread.pkl"
SCALER_PATH = SCRIPT_DIR / "scaler.pkl"
FEATURES_META_PATH = SCRIPT_DIR / "features_meta.json"
MODEL_PROBA_ANTITRAP_PATH = SCRIPT_DIR / "model_proba_antitrap.pkl"
MODEL_SPREAD_ANTITRAP_PATH = SCRIPT_DIR / "model_spread_antitrap.pkl"
SCALER_ANTITRAP_PATH = SCRIPT_DIR / "scaler_antitrap.pkl"
FEATURES_META_ANTITRAP_PATH = SCRIPT_DIR / "features_meta_antitrap.json"
FEATURES_META_TOTALS_PATH = SCRIPT_DIR / "features_meta_totals.json"

# Modèles chargés au démarrage (pickle + features_meta.json)
_ml_clf: Any = None
_ml_reg: Any = None
_ml_scaler: Any = None
_ml_meta: Dict[str, Any] = {}

# Seuil Edge pour 🎯 SNIPER TARGET : (Proba * Cote) - 1 > 5%
SNIPER_TARGET_EV_THRESHOLD: float = 0.05

# ==============================================================================
# CONFIGURATION
# ==============================================================================

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

LEAGUE_CONFIGS: Dict[int, Dict[str, float]] = {
    2: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.05},
    8: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.02},
    120: {"baseline_pace": 72.0, "home_advantage": 4.0, "defensive_factor": 1.08},
    4: {"baseline_pace": 70.0, "home_advantage": 3.5, "defensive_factor": 1.05},
    52: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.04},
    5: {"baseline_pace": 72.0, "home_advantage": 3.5, "defensive_factor": 1.03},
    194: {"baseline_pace": 74.0, "home_advantage": 4.5, "defensive_factor": 1.02},
    198: {"baseline_pace": 70.0, "home_advantage": 4.0, "defensive_factor": 1.06},
    45: {"baseline_pace": 70.0, "home_advantage": 4.0, "defensive_factor": 1.06},
    206: {"baseline_pace": 72.0, "home_advantage": 3.0, "defensive_factor": 1.02},
}

DEFAULT_LEAGUE_CONFIG: Dict[str, float] = {"baseline_pace": 72.0, "home_advantage": 3.0, "defensive_factor": 1.0}

LEAGUE_IDS: set = set(LEAGUES.values())
SEASONS_TO_TRY: List[str] = ["2025-2026", "2025", "2024-2025", "2024"]

# API
BASE_URL: str = "https://v1.basketball.api-sports.io"
BOOKMAKER_IDS: List[int] = [17, 7, 1]  # Betclic, Unibet, Bwin
BET_HOME_AWAY_ID: int = 2
BET_OVER_UNDER_IDS: List[int] = [4, 5, 8, 16, 18]  # Over/Under, Over/Under Games
API_RETRIES: int = 3
API_RETRY_DELAY: float = 1.0

# Cache
CACHE_COLD_TTL: int = 3600   # 1h pour stats Supabase
CACHE_HOT_TTL: int = 300     # 5 min pour API

# Math
PYTHAGOREAN_EXP: float = 10.2
MIN_GAMES_RELIABLE: int = 3
WEIGHT_RECENT: float = 2.0   # 3 derniers matchs comptent double
N_BOX_SCORES: int = 15

# UI
EDGE_MAX_BET: float = 10.0
EDGE_VALUE: float = 5.0
COLOR_MAX_BET: str = "#22c55e"  # Vert fluo pour MAX BET
COLOR_MAX_BET_BG: str = "#14532d"  # Fond sombre
COLOR_VALUE: str = "#22c55e"
COLOR_PASS: str = "#6b7280"

# Over/Under (ML Total vs ligne bookmaker)
OU_VALUE_THRESHOLD: float = 4.0
OU_MAX_THRESHOLD: float = 5.0   # Si ML_Total > Line + 5 → 🔥 MAX OVER ; Si ML_Total < Line - 5 → 🔥 MAX UNDER

# Radar 4 Factors
FOUR_FACTORS_KEYS: List[str] = ["efg_pct", "orb_pct", "tov_inv", "ft_rate"]
FOUR_FACTORS_LABELS: List[str] = ["Shooting (eFG%)", "Rebounding (ORB%)", "Propreté (1-TOV%)", "Lancers (FT Rate)"]

# Fatigue (Boulazac Factor)
FATIGUE_SHORT_REST_DAYS: int = 2
FATIGUE_SHORT_REST_PENALTY: float = 0.05
FATIGUE_RUST_DAYS: int = 7
FATIGUE_RUST_PENALTY: float = 0.02


# ==============================================================================
# INITIALISATION ML — Chargement au démarrage (pickle + features_meta.json)
# ==============================================================================


def load_ml_models() -> bool:
    """
    Charge les modèles ML : antitrap (model_*_antitrap.pkl) si présents, sinon ancien entraînement.
    Retourne True si les modèles sont prêts, False sinon (→ Mode Secours Pythagore).
    """
    global _ml_clf, _ml_reg, _ml_scaler, _ml_meta
    if _ml_clf is not None and _ml_reg is not None:
        return True
    use_antitrap = MODEL_PROBA_ANTITRAP_PATH.exists() and MODEL_SPREAD_ANTITRAP_PATH.exists()
    path_proba = MODEL_PROBA_ANTITRAP_PATH if use_antitrap else MODEL_PROBA_PATH
    path_spread = MODEL_SPREAD_ANTITRAP_PATH if use_antitrap else MODEL_SPREAD_PATH
    path_scaler = SCALER_ANTITRAP_PATH if use_antitrap else SCALER_PATH
    path_meta = FEATURES_META_ANTITRAP_PATH if use_antitrap else FEATURES_META_PATH
    if not path_proba.exists() or not path_spread.exists():
        return False
    try:
        with open(path_proba, "rb") as f:
            _ml_clf = pickle.load(f)
        with open(path_spread, "rb") as f:
            _ml_reg = pickle.load(f)
        _ml_scaler = None
        if path_scaler.exists():
            try:
                with open(path_scaler, "rb") as f:
                    _ml_scaler = pickle.load(f)
            except Exception:
                pass
        _ml_meta = {}
        if path_meta.exists():
            try:
                with open(path_meta, "r", encoding="utf-8") as f:
                    _ml_meta = json.load(f)
            except Exception:
                pass
        return _ml_clf is not None and _ml_reg is not None
    except Exception:
        _ml_clf = _ml_reg = _ml_scaler = None
        _ml_meta = {}
        return False


def _apply_calibration(raw_prob: float, calibration: List[Dict[str, float]]) -> float:
    """
    Interpole la probabilité brute à travers la courbe de calibration (test set).
    Quand le modèle dit X%, on utilise le taux réel observé (Y%) pour les mises.
    """
    if not calibration or len(calibration) < 2:
        return raw_prob
    preds = [c.get("predicted_bin", 0) for c in calibration]
    actuals = [c.get("actual_win_rate", 0) for c in calibration]
    # Trier par predicted_bin pour np.interp (xp doit être croissant)
    sorted_pairs = sorted(zip(preds, actuals), key=lambda x: x[0])
    preds, actuals = [p for p, _ in sorted_pairs], [a for _, a in sorted_pairs]
    return float(np.clip(np.interp(raw_prob, preds, actuals), 0.0, 1.0))


def get_ml_prediction(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline ML : récupère les features (Rolling N-10, Fatigue, Win Rate, etc.) via training_engine,
    aligne sur FEATURE_NAMES, impute, scale, puis model.predict_proba() et model.predict(spread).
    Retourne prob_home (brute), prob_home_calibrated (si calibration dispo), proj_home, proj_away, spread.
    None si modèles non chargés ou données insuffisantes (→ basculer Mode Secours).
    """
    if _ml_clf is None or _ml_reg is None:
        if not load_ml_models():
            return None
    try:
        from training_engine import build_feature_row_for_match
    except Exception:
        return None
    feature_names = _ml_meta.get("feature_names", [])
    if not feature_names:
        try:
            from training_engine import FEATURE_NAMES
            feature_names = FEATURE_NAMES
        except Exception:
            return None
    train_means = _ml_meta.get("train_means", {})
    date_str = (game_date or str(date.today()))[:10]
    if len(date_str) != 10:
        return None

    row = build_feature_row_for_match(home_id, away_id, date_str, league_id, season)
    if row is None:
        return None

    X = pd.DataFrame([row])
    for col in feature_names:
        if col not in X.columns:
            X[col] = train_means.get(col, 0.0)
    X = X[feature_names]
    for col in feature_names:
        if col in X.columns and (X[col].isna().any() or np.isinf(X[col]).any()):
            X[col] = X[col].fillna(train_means.get(col, 0.0)).replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0)

    if _ml_scaler is not None:
        X = pd.DataFrame(_ml_scaler.transform(X), columns=feature_names)

    prob_home = float(_ml_clf.predict_proba(X)[0, 1])
    spread = float(_ml_reg.predict(X)[0])
    # Total prédit par model_totals.pkl (plus de constante 150)
    predicted_total: Optional[float] = None
    if predict_total_points:
        try:
            predicted_total = predict_total_points(home_id, away_id, date_str, league_id, season)
        except Exception:
            pass
    if predicted_total is None or not (isinstance(predicted_total, (int, float)) and np.isfinite(predicted_total)):
        # Fallback si model_totals absent : moyenne réaliste (jamais 150)
        predicted_total = _default_total_fallback(league_id)
    proj_home = (float(predicted_total) + spread) / 2.0
    proj_away = (float(predicted_total) - spread) / 2.0

    calibration = _ml_meta.get("calibration", [])
    prob_calibrated = _apply_calibration(prob_home, calibration) if calibration else prob_home

    return {
        "prob_home": prob_home,
        "prob_home_calibrated": prob_calibrated,
        "proj_home": proj_home,
        "proj_away": proj_away,
        "spread": spread,
    }


# ==============================================================================
# HELPERS
# ==============================================================================


def _get_league_config(league_id: Optional[int]) -> Dict[str, float]:
    if league_id is not None and league_id in LEAGUE_CONFIGS:
        return LEAGUE_CONFIGS[league_id].copy()
    return DEFAULT_LEAGUE_CONFIG.copy()


# Total par défaut quand model_totals.pkl est absent (jamais 150)
DEFAULT_TOTAL_FALLBACK: float = 165.0


def _default_total_fallback(league_id: Optional[int]) -> float:
    """Fallback réaliste pour le total quand le modèle Totals n'est pas disponible."""
    return DEFAULT_TOTAL_FALLBACK


# ==============================================================================
# FLUX FROID — Supabase (database.py uniquement)
# ==============================================================================


def _get_supabase():
    return get_client()


@st.cache_data(ttl=CACHE_COLD_TTL)
def _cold_get_team_profile(team_id: int) -> Optional[dict]:
    supabase = _get_supabase()
    if not supabase:
        return None
    try:
        r = supabase.table("teams_metadata").select("*").eq("team_id", team_id).limit(1).execute()
        data = r.data or []
        return data[0] if data else None
    except Exception:
        return None


@st.cache_data(ttl=CACHE_COLD_TTL)
def _cold_get_team_name(team_id: int) -> str:
    meta = _cold_get_team_profile(team_id)
    if meta:
        nom = (meta.get("nom_equipe") or meta.get("name") or "").strip()
        if nom:
            return nom
    return f"Équipe {team_id}"


@st.cache_data(ttl=CACHE_COLD_TTL)
def _cold_get_last_n_matches_with_box(team_id: int, n: int = 3) -> List[dict]:
    """
    Derniers n matchs de l'équipe avec box scores et scores finaux.
    Retourne liste de dicts : date, opponent, score_us, score_opp, result (V/D/?), pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate.
    """
    supabase = _get_supabase()
    if not supabase:
        return []
    try:
        r = (
            supabase.table("box_scores")
            .select("game_id, opponent_id, is_home, date, pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate")
            .eq("team_id", team_id)
            .order("date", desc=True)
            .limit(n)
            .execute()
        )
        rows = r.data or []
        if not rows:
            return []
        game_ids = list({x["game_id"] for x in rows})
        games_by_id: Dict[int, dict] = {}
        for gid in game_ids:
            gr = supabase.table("games_history").select("home_id, away_id, home_score, away_score").eq("game_id", gid).limit(1).execute()
            if gr.data:
                games_by_id[gid] = gr.data[0]
        result: List[dict] = []
        for b in rows:
            gid = b["game_id"]
            g = games_by_id.get(gid, {})
            opp_id = b.get("opponent_id")
            is_home = b.get("is_home", True)
            score_us = g.get("home_score") if is_home else g.get("away_score")
            score_opp = g.get("away_score") if is_home else g.get("home_score")
            opp_name = _cold_get_team_name(opp_id) if opp_id else "?"
            if score_us is not None and score_opp is not None:
                res = "V" if score_us > score_opp else "D"
                score_str = f"{score_us} — {score_opp}"
            else:
                res = "?"
                score_str = "—"
            result.append({
                "date": (b.get("date") or "")[:10],
                "opponent": opp_name,
                "score": score_str,
                "result": res,
                "pace": b.get("pace"),
                "off_rtg": b.get("off_rtg"),
                "def_rtg": b.get("def_rtg"),
                "efg_pct": b.get("efg_pct"),
                "orb_pct": b.get("orb_pct"),
                "tov_pct": b.get("tov_pct"),
                "ft_rate": b.get("ft_rate"),
            })
        return result
    except Exception:
        return []


def _compute_strength_from_rows(rows: List[dict], league_id: Optional[int]) -> Optional[dict]:
    """Sous-routine : calcule strength à partir des rows box_scores."""
    if len(rows) < MIN_GAMES_RELIABLE:
        return None
    cfg = _get_league_config(league_id)
    baseline_pace = cfg.get("baseline_pace", 72.0)
    n = len(rows)
    weights = np.array([WEIGHT_RECENT if i < 3 else 1.0 for i in range(n)], dtype=float)
    weights /= weights.sum()
    pace = float(np.average([x.get("pace") or baseline_pace for x in rows], weights=weights))
    off_rtg = float(np.average([x.get("off_rtg") or 100 for x in rows], weights=weights))
    def_rtg = float(np.average([x.get("def_rtg") or 100 for x in rows], weights=weights))
    efg_pct = float(np.average([x.get("efg_pct") or 0.5 for x in rows], weights=weights))
    orb_pct = float(np.average([x.get("orb_pct") or 0.25 for x in rows], weights=weights))
    tov_pct = float(np.average([x.get("tov_pct") or 0.15 for x in rows], weights=weights))
    ft_rate = float(np.average([x.get("ft_rate") or 0.25 for x in rows], weights=weights))
    pts_list = [(x.get("off_rtg") or 100) * (x.get("pace") or baseline_pace) / 100 for x in rows]
    std_pts = float(np.std(pts_list)) if len(pts_list) > 1 else 0.0
    fiabilite = max(0, 100 - std_pts * 2) if std_pts > 0 else 100
    return {
        "pace": round(pace, 1),
        "off_rtg": round(off_rtg, 1),
        "def_rtg": round(def_rtg, 1),
        "efg_pct": round(efg_pct, 3),
        "orb_pct": round(orb_pct, 3),
        "tov_pct": round(tov_pct, 3),
        "ft_rate": round(ft_rate, 3),
        "n_games": n,
        "fiabilite": round(min(100, fiabilite), 1),
    }


@st.cache_data(ttl=CACHE_COLD_TTL)
def calculate_team_strength(team_id: int, league_id: Optional[int] = None) -> Optional[dict]:
    """
    Dean Oliver's 4 Factors + Power Rating depuis box_scores.
    Pondération dégressive : 3 derniers matchs x2.
    """
    supabase = _get_supabase()
    if not supabase:
        return None
    try:
        r = (
            supabase.table("box_scores")
            .select("pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate")
            .eq("team_id", team_id)
            .order("game_id", desc=True)
            .limit(N_BOX_SCORES)
            .execute()
        )
        return _compute_strength_from_rows(r.data or [], league_id)
    except Exception:
        return None


def calculate_team_strength_at_date(
    team_id: int,
    league_id: Optional[int],
    before_date: str,
) -> Optional[dict]:
    """
    Time Machine : stats strictement AVANT before_date (pas de lookahead).
    Pour le backtest : prédiction telle qu'elle aurait été faite avant le match.
    """
    supabase = _get_supabase()
    if not supabase:
        return None
    try:
        date_str = (before_date or "")[:10]
        if len(date_str) != 10:
            return None
        r = (
            supabase.table("box_scores")
            .select("pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate")
            .eq("team_id", team_id)
            .lt("date", date_str)
            .order("date", desc=True)
            .limit(N_BOX_SCORES)
            .execute()
        )
        return _compute_strength_from_rows(r.data or [], league_id)
    except Exception:
        return None


def get_previous_match_date(team_id: int, before_date: str) -> Optional[str]:
    """
    Date du dernier match de l'équipe AVANT before_date (pour fatigue).
    """
    supabase = _get_supabase()
    if not supabase:
        return None
    try:
        date_str = (before_date or "")[:10]
        if len(date_str) != 10:
            return None
        r = (
            supabase.table("games_history")
            .select("date")
            .or_(f"home_id.eq.{team_id},away_id.eq.{team_id}")
            .lt("date", date_str)
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        data = r.data or []
        if not data:
            return None
        d = data[0].get("date")
        return (d or "")[:10] if d else None
    except Exception:
        return None


def get_days_rest(team_id: int, game_date: str) -> Optional[int]:
    """Jours de repos avant le match. None si pas de match précédent trouvé."""
    prev = get_previous_match_date(team_id, game_date)
    if not prev or len(prev) != 10:
        return None
    try:
        d1 = datetime.strptime(prev, "%Y-%m-%d").date()
        d2 = datetime.strptime((game_date or "")[:10], "%Y-%m-%d").date()
        return (d2 - d1).days
    except Exception:
        return None


def apply_fatigue_penalty(off_rtg: float, days_rest: Optional[int]) -> Tuple[float, str]:
    """
    Boulazac Factor : pénalité selon le repos.
    Retourne (off_rtg_adj, label) ex: (95.0, "Short Rest (2j)").
    """
    if days_rest is None:
        return off_rtg, ""
    if days_rest <= FATIGUE_SHORT_REST_DAYS:
        adj = off_rtg * (1.0 - FATIGUE_SHORT_REST_PENALTY)
        return adj, f"Short Rest ({days_rest}j)"
    if days_rest >= FATIGUE_RUST_DAYS:
        adj = off_rtg * (1.0 - FATIGUE_RUST_PENALTY)
        return adj, f"Rouille ({days_rest}j)"
    return off_rtg, ""


# ==============================================================================
# FLUX CHAUD — API-Sports
# ==============================================================================


def _api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
    api_key = (os.environ.get("API_BASKETBALL_KEY") or "").strip()
    if not api_key:
        return None, "API key manquante"
    url = f"{BASE_URL}/{endpoint}"
    headers = {"x-apisports-key": api_key}
    last_err: Optional[str] = None
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params or {}, timeout=15)
            data = r.json() if r.text else {}
            if r.status_code != 200:
                last_err = str(data.get("errors") or f"HTTP {r.status_code}")
                if r.status_code == 429:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                continue
            if data.get("errors"):
                last_err = str(data["errors"])
                continue
            return data, None
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(API_RETRY_DELAY * (attempt + 1))
    return None, last_err


# Nombre de jours à afficher (aujourd'hui + demain + après-demain = 3)
NEXT_DAYS_HORIZON: int = 3


@st.cache_data(ttl=CACHE_HOT_TTL)
def _hot_fetch_games_next_days(ndays: int = NEXT_DAYS_HORIZON) -> List[dict]:
    """Calendrier des N prochains jours via API (aujourd'hui, demain, ...)."""
    today = date.today()
    all_games: List[dict] = []
    seen_gid: set = set()
    for offset in range(ndays):
        d = today + timedelta(days=offset)
        date_str = d.strftime("%Y-%m-%d")
        for league_name, league_id in LEAGUES.items():
            for season in SEASONS_TO_TRY:
                try:
                    data, err = _api_get("games", {
                        "date": date_str,
                        "league": league_id,
                        "season": season,
                        "timezone": "Europe/Paris",
                    })
                    if err or not data:
                        continue
                    resp = data.get("response") or []
                    for g in resp:
                        gid = g.get("id")
                        if gid and gid not in seen_gid:
                            seen_gid.add(gid)
                            g["_league_id"] = league_id
                            g["_league_name"] = league_name
                            g["_season"] = season
                            g["_date"] = date_str
                            g["_day_label"] = (
                                "Aujourd'hui" if offset == 0
                                else "Demain" if offset == 1
                                else f"J+{offset}"
                            )
                            all_games.append(g)
                except Exception:
                    continue
    # Tri par date puis par heure
    all_games.sort(key=lambda x: (x.get("_date", ""), (x.get("date") or "") + (x.get("time") or "")))
    return all_games


@st.cache_data(ttl=CACHE_HOT_TTL)
def _hot_fetch_odds(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float]]:
    """Cotes Home/Away. Priorité Betclic (17), Unibet (7)."""
    try:
        data, err = _api_get("odds", {"game": game_id, "league": league_id, "season": season})
        if err or not data:
            return None, None
        resp = data.get("response")
        if not isinstance(resp, list) or len(resp) == 0:
            return None, None
        item = resp[0]
        for bm in (item.get("bookmakers") or []):
            if bm.get("id") not in BOOKMAKER_IDS:
                continue
            for bet in (bm.get("bets") or []):
                if int(bet.get("id", 0)) != BET_HOME_AWAY_ID:
                    continue
                odd_h, odd_a = None, None
                for v in (bet.get("values") or []):
                    val = (v.get("value") or "").strip().lower()
                    try:
                        odd_f = float(v.get("odd") or 0)
                        if val == "home":
                            odd_h = odd_f
                        elif val == "away":
                            odd_a = odd_f
                    except (TypeError, ValueError):
                        pass
                if odd_h is not None and odd_a is not None:
                    return odd_h, odd_a
        for bm in (item.get("bookmakers") or []):
            for bet in (bm.get("bets") or []):
                if int(bet.get("id", 0)) != BET_HOME_AWAY_ID:
                    continue
                odd_h, odd_a = None, None
                for v in (bet.get("values") or []):
                    val = (v.get("value") or "").strip().lower()
                    try:
                        odd_f = float(v.get("odd") or 0)
                        if val == "home":
                            odd_h = odd_f
                        elif val == "away":
                            odd_a = odd_f
                    except (TypeError, ValueError):
                        pass
                if odd_h is not None and odd_a is not None:
                    return odd_h, odd_a
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=CACHE_HOT_TTL)
def _hot_fetch_odds_spread(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Spread bookmaker: (spread_home, cote_home, cote_away)."""
    try:
        data, err = _api_get("odds", {"game": game_id, "league": league_id, "season": season})
        if err or not data:
            return None, None, None
        resp = data.get("response")
        if not isinstance(resp, list) or len(resp) == 0:
            return None, None, None
        item = resp[0]
        for bm in (item.get("bookmakers") or []):
            if bm.get("id") not in BOOKMAKER_IDS:
                continue
            for bet in (bm.get("bets") or []):
                name = (bet.get("name") or "").lower()
                if "handicap" not in name and "spread" not in name:
                    continue
                if "1st" in name or "half" in name:
                    continue
                values = bet.get("values") or []
                spread_val = odd_h = odd_a = None
                for v in values:
                    val_raw = (v.get("value") or "").strip()
                    try:
                        odd_f = float(v.get("odd") or 0) if v.get("odd") else None
                        m = re.search(r"-?[\d]+[.,]?\d*", val_raw)
                        if m and spread_val is None:
                            spread_val = float(m.group(0).replace(",", "."))
                        if "home" in val_raw.lower() or val_raw.startswith("-"):
                            odd_h = odd_f
                        elif "away" in val_raw.lower() or val_raw.startswith("+"):
                            odd_a = odd_f
                    except (TypeError, ValueError):
                        pass
                if spread_val is not None and (odd_h or odd_a):
                    return spread_val, odd_h, odd_a
    except Exception:
        pass
    return None, None, None


@st.cache_data(ttl=CACHE_HOT_TTL)
def _hot_fetch_odds_totals(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Cotes Over/Under Total Points. Retourne (ligne, cote_over, cote_under).
    """
    try:
        data, err = _api_get("odds", {"game": game_id, "league": league_id, "season": season})
        if err or not data:
            return None, None, None
        resp = data.get("response")
        if not isinstance(resp, list) or len(resp) == 0:
            return None, None, None
        item = resp[0]
        for bm in (item.get("bookmakers") or []):
            if bm.get("id") not in BOOKMAKER_IDS:
                continue
            for bet in (bm.get("bets") or []):
                bid = int(bet.get("id", 0))
                name = (bet.get("name") or "").lower()
                if bid not in BET_OVER_UNDER_IDS and "over" not in name and "under" not in name and "total" not in name and "points" not in name:
                    continue
                if "1st" in name or "half" in name:
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
                        return float(num_match.group(0).replace(",", ".")), odd_over, odd_under
        for bm in (item.get("bookmakers") or []):
            for bet in (bm.get("bets") or []):
                name = (bet.get("name") or "").lower()
                if "over" not in name and "under" not in name:
                    continue
                if "1st" in name or "half" in name:
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
    except Exception:
        pass
    return None, None, None


# ==============================================================================
# MOTEUR MATHÉMATIQUE
# ==============================================================================


def predict_match(
    home_strength: Optional[dict],
    away_strength: Optional[dict],
    league_id: Optional[int],
    game_date: Optional[str] = None,
    home_id: Optional[int] = None,
    away_id: Optional[int] = None,
) -> Tuple[float, float, float, str, str]:
    """
    Prédiction avec intégration Fatigue (Boulazac Factor).
    Retourne (prob_home, proj_home, proj_away, fatigue_home_label, fatigue_away_label).
    """
    cfg = _get_league_config(league_id)
    home_adv = cfg.get("home_advantage", 3.0)
    baseline = cfg.get("baseline_pace", 72.0)

    off_h = (home_strength or {}).get("off_rtg") or 100.0
    off_a = (away_strength or {}).get("off_rtg") or 100.0
    fatigue_h, fatigue_a = "", ""

    if game_date and home_id is not None:
        dr_h = get_days_rest(home_id, game_date)
        off_h_adj, fatigue_h = apply_fatigue_penalty(off_h, dr_h)
        off_h = off_h_adj
    if game_date and away_id is not None:
        dr_a = get_days_rest(away_id, game_date)
        off_a_adj, fatigue_a = apply_fatigue_penalty(off_a, dr_a)
        off_a = off_a_adj

    pace_h = (home_strength or {}).get("pace") or baseline
    pace_a = (away_strength or {}).get("pace") or baseline
    pace_avg = (pace_h + pace_a) / 2.0

    proj_home = (off_h / 100.0) * pace_avg + home_adv
    proj_away = (off_a / 100.0) * pace_avg

    denom = (proj_home ** PYTHAGOREAN_EXP) + (proj_away ** PYTHAGOREAN_EXP)
    prob_home = float((proj_home ** PYTHAGOREAN_EXP) / denom) if denom > 0 else 0.5

    return prob_home, proj_home, proj_away, fatigue_h, fatigue_a


def calculate_win_probability(
    home_strength: Optional[dict],
    away_strength: Optional[dict],
    league_id: Optional[int],
) -> Tuple[float, float, float]:
    """
    Pythagorean Avancé (sans fatigue). Pour rétrocompatibilité.
    """
    prob, proj_h, proj_a, _, _ = predict_match(home_strength, away_strength, league_id)
    return prob, proj_h, proj_a


def calculate_fair_spread(proj_home: float, proj_away: float, home_name: str, away_name: str) -> str:
    spread = proj_home - proj_away
    if abs(spread) < 0.1:
        return "Égalité"
    if spread > 0:
        return f"{home_name} {spread:+.1f}"
    return f"{away_name} {abs(spread):+.1f}"


def ev_pct(prob: float, odd: Optional[float]) -> float:
    if odd is None or odd <= 0:
        return 0.0
    return ((prob * odd) - 1.0) * 100.0


def _mise_bucket(edge: float, fiabilite: float) -> str:
    if fiabilite < 30:
        return "🛑 PASS"
    if edge >= EDGE_MAX_BET:
        return "🔥 MAX BET"
    if edge >= EDGE_VALUE:
        return "✅ VALUE"
    if edge < 0:
        return "🛑 PASS"
    return "🛑 PASS"


def _maths_value_str(prob: float, odd: Optional[float]) -> str:
    """Formule explicite : (Proba% * Cote) - 1. Si > 0% → Value."""
    if odd is None or odd <= 0:
        return "N/A"
    ev = (prob * odd) - 1.0
    pct = ev * 100.0
    return f"{pct:+.1f}%"


def _ou_confiance_bucket(diff: float, fiabilite: float) -> str:
    """Over/Under : MAX OVER/UNDER si |diff| >= 5, VALUE si >= 4, sinon PASS."""
    if fiabilite < 30:
        return "🛑 PASS"
    if diff >= OU_MAX_THRESHOLD:
        return "🔥 MAX OVER"
    if diff <= -OU_MAX_THRESHOLD:
        return "🔥 MAX UNDER"
    if abs(diff) >= OU_VALUE_THRESHOLD:
        return "✅ VALUE"
    return "🛑 PASS"


# ==============================================================================
# BACKTEST (Time Machine)
# ==============================================================================


@st.cache_data(ttl=CACHE_COLD_TTL)
def run_backtest(days: int = 3) -> Dict[str, Any]:
    """
    Backtest honnête : prédiction telle qu'elle aurait été faite avant le match.
    Récupère les matchs terminés des N derniers jours (games_history).
    Pour chaque match : stats box_scores strictement antérieures à la date du match.
    """
    supabase = _get_supabase()
    if not supabase:
        return {"error": "Supabase indisponible", "rows": [], "n_correct": 0, "n_total": 0}

    today = date.today()
    start = (today - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        r = (
            supabase.table("games_history")
            .select("*")
            .gte("date", start)
            .lte("date", today.strftime("%Y-%m-%d"))
            .order("date", desc=True)
            .execute()
        )
        games = r.data or []
    except Exception as e:
        return {"error": str(e), "rows": [], "n_correct": 0, "n_total": 0}

    rows: List[dict] = []
    n_correct = 0
    n_total = 0

    for g in games:
        game_id = g.get("game_id")
        date_str = (g.get("date") or "")[:10]
        league_id = g.get("league_id")
        home_id = g.get("home_id")
        away_id = g.get("away_id")
        home_score = g.get("home_score")
        away_score = g.get("away_score")

        if home_score is None or away_score is None:
            continue
        if league_id not in LEAGUE_IDS or not home_id or not away_id:
            continue

        home_name = _cold_get_team_name(home_id)
        away_name = _cold_get_team_name(away_id)
        season = (g.get("season") or "").strip() or "2025"

        # Priorité ML si modèles chargés
        ml_pred = get_ml_prediction(home_id, away_id, date_str, league_id, season) if load_ml_models() else None
        if ml_pred is not None:
            prob_home = ml_pred.get("prob_home_calibrated", ml_pred["prob_home"])
            pred_spread = ml_pred["spread"]
            pred_winner = home_name if prob_home >= 0.5 else away_name
        else:
            home_st = calculate_team_strength_at_date(home_id, league_id, date_str)
            away_st = calculate_team_strength_at_date(away_id, league_id, date_str)
            if home_st is None or away_st is None:
                pred_winner = "—"
                pred_spread = 0.0
                prob_home = 0.5
            else:
                prob_home, proj_home, proj_away, _, _ = predict_match(
                    home_st, away_st, league_id, date_str, home_id, away_id
                )
                pred_spread = proj_home - proj_away
                pred_winner = home_name if prob_home >= 0.5 else away_name

        real_winner = home_name if home_score > away_score else away_name
        real_spread = home_score - away_score
        is_correct = pred_winner == real_winner
        spread_error = abs(pred_spread - real_spread) if pred_winner != "—" else None

        if pred_winner != "—":
            n_total += 1
            if is_correct:
                n_correct += 1

        rows.append({
            "Match": f"{home_name} vs {away_name}",
            "Date": date_str,
            "Prédiction": pred_winner,
            "Résultat Réel": f"{home_score} — {away_score}",
            "Verdict": "✅" if is_correct else "❌",
            "Erreur Spread": f"{spread_error:.1f}" if spread_error is not None else "—",
            "_spread_error": spread_error,
        })

    return {"rows": rows, "n_correct": n_correct, "n_total": n_total, "error": None}


# ==============================================================================
# UI — Bloomberg Style
# ==============================================================================


def _format_heure(g: dict) -> str:
    d = g.get("date") or ""
    t = g.get("time") or ""
    if not d:
        return "—"
    try:
        if t and ":" in str(t):
            return f"{d[8:10]}/{d[5:7]} {t[:5]}"
        return f"{d[8:10]}/{d[5:7]}"
    except Exception:
        return d[:10] if len(d) >= 10 else "—"


def _plot_radar_four_factors(home_st: dict, away_st: dict, home_name: str, away_name: str) -> go.Figure:
    def _val(st: dict, key: str) -> float:
        if key == "tov_inv":
            return 1.0 - (st.get("tov_pct") or 0.15)
        return st.get(key, 0.5 if key == "efg_pct" else 0.25)
    def _scale(v: float, k: str) -> float:
        if k == "efg_pct":
            return min(100, v * 100)
        return min(100, v * 100)
    theta = FOUR_FACTORS_LABELS + [FOUR_FACTORS_LABELS[0]]
    r_h = [_scale(_val(home_st or {}, k), k) for k in FOUR_FACTORS_KEYS]
    r_h = r_h + [r_h[0]]
    r_a = [_scale(_val(away_st or {}, k), k) for k in FOUR_FACTORS_KEYS]
    r_a = r_a + [r_a[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_h, theta=theta, fill="toself", name=home_name, line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatterpolar(r=r_a, theta=theta, fill="toself", name=away_name, line=dict(color="#ef4444")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=True, title="4 Factors (Dean Oliver)", height=450)
    return fig


# ==============================================================================
# BUILD SNIPER TABLE
# ==============================================================================


# Offline-First : lecture depuis daily_projections (pas d'API ni ML)
OFFLINE_MODE: bool = bool(os.environ.get("SNIPER_OFFLINE", "1") == "1")

# Modes de sélection du cerveau (Standard vs Chasseur)
BRAIN_MODE_AUTO = "auto"
BRAIN_MODE_STANDARD = "standard"
BRAIN_MODE_CHASSEUR = "chasseur"


@st.cache_data(ttl=60)
def _fetch_daily_projections() -> List[dict]:
    """Lit daily_projections depuis Supabase (mode Offline-First)."""
    supabase = _get_supabase()
    if not supabase:
        return []
    try:
        r = supabase.table("daily_projections").select("*").order("edge_percent", desc=True).execute()
        return r.data or []
    except Exception:
        return []


def build_sniper_table_from_cache() -> pd.DataFrame:
    """Construit le DataFrame à partir de daily_projections (lecture seule, instantané)."""
    rows = _fetch_daily_projections()
    if not rows:
        return pd.DataFrame()

    out: List[dict] = []
    for r in rows:
        oh = r.get("odds_home")
        oa = r.get("odds_away")
        cotes_str = f"{oh:.2f} | {oa:.2f}" if (oh is not None and oa is not None) else "—"
        proba_book_h = (100.0 / oh) if oh else None
        proba_book_a = (100.0 / oa) if oa else None
        proba_book_str = f"{proba_book_h:.0f}% | {proba_book_a:.0f}%" if (proba_book_h and proba_book_a) else "—"
        prob_cal = r.get("proba_calibree")
        prob_cal_str = f"{prob_cal*100:.1f}%" if prob_cal is not None else "—"
        edge = r.get("edge_percent") or 0
        edge_str = f"{edge:+.0f}%" if edge != 0 else "—"
        line_book = r.get("line_bookmaker")
        ligne_book_str = f"{line_book:.1f}" if line_book is not None else "En attente"
        diff = r.get("diff_total")
        diff_str = f"{diff:+.1f} pts" if diff is not None else "—"

        out.append({
            "_game_id": r.get("game_id"),
            "_date": str(r.get("date", ""))[:10] if r.get("date") else None,
            "_league_id": r.get("league_id"),
            "_season": r.get("season"),
            "_home_id": r.get("home_id"),
            "_away_id": r.get("away_id"),
            "_home_name": (r.get("match_name") or "").split(" vs ")[0] if r.get("match_name") else "",
            "_away_name": (r.get("match_name") or "").split(" vs ")[1] if r.get("match_name") and " vs " in r.get("match_name", "") else "",
            "Jour": r.get("jour", "—"),
            "Heure": "",
            "MATCH": r.get("match_name", ""),
            "Match": r.get("match_name", ""),
            "COTES (H/A)": cotes_str,
            "PROBA BOOK (%)": proba_book_str,
            "PROBA SNIPER (%)": prob_cal_str,
            "EDGE (%)": edge_str,
            "LE PARI": r.get("le_pari", ""),
            "Cerveau utilisé": r.get("brain_used", ""),
            "Proba ML": prob_cal_str,
            "Proba calibrée": prob_cal_str,
            "Style de Match": r.get("style_match", "—"),
            "🚨 ALERTE TRAPPE": r.get("alerte_trappe", "—"),
            "🎯 PARI OUTSIDER": r.get("pari_outsider", "—"),
            "Message de Contexte": r.get("message_contexte", ""),
            "Confiance": r.get("confiance_label", ""),
            "Fiabilité": f"{r.get('fiabilite', 50):.0f}%",
            "_edge": edge,
            "_prob_home": prob_cal or 0.5,
            "_proj_home": 75.0,
            "_proj_away": 75.0,
            "_ml_total_predicted": r.get("predicted_total"),
            "_fair_spread": "",
            "_fatigue": "",
            "_is_domestic_trap": False,
            "LIGNE BOOK": ligne_book_str,
            "PROJETÉ SNIPER": round(r.get("predicted_total") or 150, 1),
            "DIFF": diff_str,
            "PARI TOTAL": r.get("pari_total", ""),
            "CONFIANCE": r.get("confiance_ou", "—"),
            "_diff_total": r.get("diff_total"),
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("_edge", ascending=False, ignore_index=True)


def build_sniper_table(brain_mode: str = BRAIN_MODE_AUTO) -> pd.DataFrame:
    """
    brain_mode: "auto" = arbitrage automatique ; "standard" = toujours Standard ; "chasseur" = prioriser Upset si cote outsider > 2.50.
    En OFFLINE_MODE : lit daily_projections (pas d'API ni ML).
    """
    if OFFLINE_MODE:
        return build_sniper_table_from_cache()
    games = _hot_fetch_games_next_days()
    if not games:
        return pd.DataFrame()

    rows: List[dict] = []
    for g in games:
        gid = g.get("id")
        league_id = g.get("_league_id")
        season = g.get("_season", "2025")
        teams = g.get("teams") or {}
        home_obj = teams.get("home") or {}
        away_obj = teams.get("away") or {}
        home_id = int(home_obj.get("id", 0)) if isinstance(home_obj, dict) else 0
        away_id = int(away_obj.get("id", 0)) if isinstance(away_obj, dict) else 0

        if not home_id or not away_id:
            continue

        home_name = (home_obj.get("name") or "").strip() or _cold_get_team_name(home_id)
        away_name = (away_obj.get("name") or "").strip() or _cold_get_team_name(away_id)

        home_st = calculate_team_strength(home_id, league_id)
        away_st = calculate_team_strength(away_id, league_id)

        game_date = (g.get("_date") or (g.get("date") or "")[:10] or date.today().strftime("%Y-%m-%d"))[:10]
        day_label = g.get("_day_label", "—")

        # Prédiction Total Points (model_totals.pkl) pour Over/Under
        ml_total_predicted: Optional[float] = None
        if predict_total_points:
            try:
                ml_total_predicted = predict_total_points(home_id, away_id, game_date, league_id, season)
            except Exception:
                pass

        # Pipeline ML en priorité (modèles chargés au démarrage)
        ml_pred = get_ml_prediction(home_id, away_id, game_date, league_id, season)
        mode_secours = False
        prob_home_calibrated: Optional[float] = None

        if ml_pred is not None:
            prob_home = ml_pred["prob_home"]
            prob_home_calibrated = ml_pred.get("prob_home_calibrated", prob_home)
            proj_home = ml_pred["proj_home"]
            proj_away = ml_pred["proj_away"]
            fair_spread = calculate_fair_spread(proj_home, proj_away, home_name, away_name)
            fatigue_label = ""
            conf = min(home_st.get("fiabilite", 50), away_st.get("fiabilite", 50)) / 100.0 if (home_st and away_st) else 0.7
        else:
            mode_secours = True
            if home_st is None or away_st is None:
                conf = 0.0
                prob_home = 0.5
                proj_home = 75.0
                proj_away = 75.0
                fair_spread = "Data Insuffisante"
                fatigue_label = ""
            else:
                conf = min(home_st.get("fiabilite", 50), away_st.get("fiabilite", 50)) / 100.0
                prob_home, proj_home, proj_away, fatigue_h, fatigue_a = predict_match(
                    home_st, away_st, league_id, game_date, home_id, away_id
                )
                fair_spread = calculate_fair_spread(proj_home, proj_away, home_name, away_name)
                parts = []
                if fatigue_h:
                    parts.append(f"⚠️ {home_name}: {fatigue_h}")
                if fatigue_a:
                    parts.append(f"⚠️ {away_name}: {fatigue_a}")
                fatigue_label = " | ".join(parts) if parts else ""

        odd_home, odd_away = _hot_fetch_odds(gid, league_id, season)
        spread_bm, cote_spread_home, cote_spread_away = _hot_fetch_odds_spread(gid, league_id, season)

        # Trap Radar : contexte EuroLeague / fatigue
        trap_info: Dict[str, Any] = {}
        if get_trap_info:
            try:
                trap_info = get_trap_info(
                    home_id, away_id, game_date, league_id, season, home_name, away_name
                )
            except Exception:
                trap_info = {"is_domestic_trap": False, "context_message": ""}
        else:
            trap_info = {"is_domestic_trap": False, "context_message": ""}
        is_domestic_trap = trap_info.get("is_domestic_trap", False)
        context_message = trap_info.get("context_message", "") or ""

        # Style de match (Shootout / Defensive Battle / Run & Gun / Verrou / Balanced)
        style_match = "—"
        if get_match_style:
            try:
                style_match = get_match_style(home_id, away_id, game_date, league_id, season)
            except Exception:
                pass

        # ——— Ensemble : Arbitrage Standard vs Chasseur de Surprises (model_upset) ———
        prob_for_ev = prob_home_calibrated if prob_home_calibrated is not None else prob_home
        brain_used = "🧠 Standard"

        upset_prob: Optional[float] = None
        if predict_upset_proba:
            try:
                upset_prob = predict_upset_proba(home_id, away_id, game_date, league_id, season)
            except Exception:
                pass

        odd_outsider = (odd_away if (odd_away or 0) > (odd_home or 0) else odd_home) or 0
        use_upset = False
        if brain_mode == BRAIN_MODE_STANDARD:
            pass  # toujours Standard
        elif brain_mode == BRAIN_MODE_CHASSEUR:
            # Forcer Chasseur dès que cote outsider > 2.50 et modèle Upset disponible
            use_upset = odd_outsider > 2.50 and upset_prob is not None
        else:
            # Auto : Chasseur seulement si cote > 2.50 ET P(outsider) > 40%
            use_upset = odd_outsider > 2.50 and upset_prob is not None and upset_prob > 0.40

        if use_upset and upset_prob is not None:
            if (odd_away or 0) > (odd_home or 0):
                prob_for_ev = 1.0 - upset_prob
            else:
                prob_for_ev = upset_prob
            brain_used = "🔥 Chasseur de Surprises"

        # Edge = (Proba * Cote) - 1. Toujours avec la proba du modèle qui a pris la main.
        edge_home = (prob_for_ev * (odd_home or 0) - 1.0) * 100.0 if odd_home else 0.0
        edge_away = ((1.0 - prob_for_ev) * (odd_away or 0) - 1.0) * 100.0 if odd_away else 0.0
        edge = max(edge_home, edge_away)
        edge_ratio = edge / 100.0
        if edge_ratio > SNIPER_TARGET_EV_THRESHOLD:
            confiance = "🎯 SNIPER TARGET"
        else:
            confiance = _mise_bucket(edge, conf * 100)

        if edge_home > edge_away and edge_home > 0:
            le_pari = f"{home_name} (@ {odd_home:.1f})" if odd_home else f"{home_name}"
        elif edge_away > 0:
            le_pari = f"{away_name} (@ {odd_away:.1f})" if odd_away else f"{away_name}"
        else:
            le_pari = "—"

        # 🎯 PARI OUTSIDER : proba réelle 35–55% sur outsider avec cote élevée → Handicap positif
        prob_outsider = 1.0 - prob_for_ev
        pari_outsider_str = "—"
        if 0.35 <= prob_outsider <= 0.55 and spread_bm is not None:
            if prob_for_ev >= 0.5:
                outsider_name, odd_out = away_name, odd_away
                handicap = -float(spread_bm) if spread_bm else 0
                cote_handicap = cote_spread_away
            else:
                outsider_name, odd_out = home_name, odd_home
                handicap = float(spread_bm) if spread_bm else 0
                cote_handicap = cote_spread_home
            if (odd_out or 0) >= 2.0 and handicap > 0 and cote_handicap:
                pari_outsider_str = f"Outsider +{handicap:.1f} @ {cote_handicap:.2f}"
            elif (odd_out or 0) >= 2.0 and handicap != 0:
                pari_outsider_str = f"{outsider_name} @ {odd_out:.1f}"

        edge_str = f"{edge:+.0f}%" if edge > 0 else "—"
        league_name = g.get("_league_name", "—")
        prob_ml_str = f"{prob_home*100:.1f}%" if ml_pred is not None else "—"
        # Proba utilisée pour l’Edge = celle du modèle qui a pris la main (Standard ou Chasseur)
        prob_cal_str = f"{prob_for_ev*100:.1f}%"
        alerte_trappe_str = "⚠️ Trap" if is_domestic_trap else "—"
        heure_str = _format_heure(g)
        cotes_str = f"{odd_home:.2f} | {odd_away:.2f}" if (odd_home and odd_away) else "—"
        proba_book_h = (100.0 / odd_home) if odd_home else None
        proba_book_a = (100.0 / odd_away) if odd_away else None
        proba_book_str = f"{proba_book_h:.0f}% | {proba_book_a:.0f}%" if (proba_book_h is not None and proba_book_a is not None) else "—"

        rows.append({
            "_game_id": gid,
            "_date": game_date,
            "_league_id": league_id,
            "_season": season,
            "_home_id": home_id,
            "_away_id": away_id,
            "_home_name": home_name,
            "_away_name": away_name,
            "_home_st": home_st,
            "_away_st": away_st,
            "_mode_secours": mode_secours,
            "Jour": day_label,
            "Heure": heure_str,
            "MATCH": f"{home_name} vs {away_name} · {heure_str}",
            "Match": f"{home_name} vs {away_name}",
            "COTES (H/A)": cotes_str,
            "PROBA BOOK (%)": proba_book_str,
            "PROBA SNIPER (%)": prob_cal_str,
            "EDGE (%)": edge_str,
            "LE PARI": le_pari,
            "Cerveau utilisé": brain_used,
            "Proba ML": prob_ml_str,
            "Proba calibrée": prob_cal_str,
            "Style de Match": style_match,
            "🚨 ALERTE TRAPPE": alerte_trappe_str,
            "🎯 PARI OUTSIDER": pari_outsider_str,
            "Message de Contexte": context_message[:80] + "…" if len(context_message) > 80 else context_message,
            "Confiance": confiance,
            "Fiabilité": f"{conf*100:.0f}%",
            "_edge": edge,
            "_prob_home": prob_home,
            "_proj_home": proj_home,
            "_proj_away": proj_away,
            "_ml_total_predicted": ml_total_predicted,
            "_fair_spread": fair_spread,
            "_fatigue": fatigue_label,
            "_is_domestic_trap": is_domestic_trap,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["_edge"], ascending=[False], ignore_index=True)


def build_sniper_totals_table(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Tableau Over/Under refonte : MATCH | LIGNE BOOK | PROJETÉ SNIPER | DIFF | STYLE | PARI TOTAL | CONFIANCE.
    PROJETÉ SNIPER = model_totals.pkl (jamais 150). STYLE = Pace/Défense (Shootout, Verrou, etc.).
    En OFFLINE_MODE : df contient déjà les colonnes depuis daily_projections.
    """
    if df_ml.empty:
        return pd.DataFrame()
    if OFFLINE_MODE and "PARI TOTAL" in df_ml.columns and "LIGNE BOOK" in df_ml.columns:
        cols = ["Match", "LIGNE BOOK", "PROJETÉ SNIPER", "DIFF", "Style de Match", "PARI TOTAL", "CONFIANCE", "_diff_total"]
        sub = df_ml[[c for c in cols if c in df_ml.columns]].copy()
        if "Style de Match" in sub.columns:
            sub = sub.rename(columns={"Style de Match": "STYLE"})
        if "_diff_total" in sub.columns:
            sub["_abs_diff"] = sub["_diff_total"].apply(lambda x: abs(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0)
            sub = sub.sort_values("_abs_diff", ascending=False, ignore_index=True).drop(columns=["_abs_diff", "_diff_total"], errors="ignore")
        else:
            sub = sub.drop(columns=["_diff_total"], errors="ignore")
        return sub
    rows: List[dict] = []
    for _, row in df_ml.iterrows():
        gid = row.get("_game_id")
        league_id = row.get("_league_id")
        season = row.get("_season", "2025")
        match_str = row.get("Match", "")
        style_match = row.get("Style de Match", "—")
        proj_home = row.get("_proj_home") or 75.0
        proj_away = row.get("_proj_away") or 75.0
        ml_total = row.get("_ml_total_predicted")
        conf = float(str(row.get("Fiabilité", "50")).replace("%", "")) / 100.0

        projected_total = float(ml_total) if ml_total is not None else (proj_home + proj_away)
        line, cote_over, cote_under = _hot_fetch_odds_totals(gid, league_id, season)

        if line is None:
            pari_total = "En attente"
            diff_str = "—"
            confiance_ou = "—"
        else:
            diff = projected_total - line
            diff_str = f"{diff:+.1f} pts"
            if diff >= OU_VALUE_THRESHOLD:
                pari_total = f"OVER {line:.1f}"
            elif diff <= -OU_VALUE_THRESHOLD:
                pari_total = f"UNDER {line:.1f}"
            else:
                pari_total = "—"
            confiance_ou = _ou_confiance_bucket(diff, conf * 100)

        ligne_book_str = f"{line:.1f}" if line is not None else "En attente"

        rows.append({
            "Match": match_str,
            "LIGNE BOOK": ligne_book_str,
            "PROJETÉ SNIPER": round(projected_total, 1),
            "DIFF": diff_str,
            "STYLE": style_match,
            "PARI TOTAL": pari_total,
            "CONFIANCE": confiance_ou,
            "_diff": projected_total - line if line is not None else None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_abs_diff"] = df["_diff"].apply(lambda x: abs(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0)
    df = df.sort_values("_abs_diff", ascending=False, ignore_index=True)
    return df.drop(columns=["_diff", "_abs_diff"], errors="ignore")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    st.set_page_config(page_title="Sniper V26", layout="wide", page_icon="🎯", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
    .stApp { background: #0f172a; }
    .bloomberg-header { display: flex; gap: 1rem; margin-bottom: 1rem; font-family: monospace; }
    .status { padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; }
    .status-db { background: #166534; color: white; }
    .status-api { background: #1e40af; color: white; }
    </style>
    """, unsafe_allow_html=True)

    db_ok = _get_supabase() is not None
    if OFFLINE_MODE:
        ml_ok = True
        api_ok = False
    else:
        ml_ok = load_ml_models()
        api_ok = False
        try:
            _ = _hot_fetch_games_next_days()
            api_ok = True
        except Exception:
            pass

    st.markdown(
        f'<div class="bloomberg-header">'
        f'<span class="status status-db">{"🟢 DB: Connected" if db_ok else "🔴 DB: Off"}</span>'
        f'<span class="status status-api">{"📦 Offline-First" if OFFLINE_MODE else ("📡 API: Live" if api_ok else "🔴 API: Off")}</span>'
        f'<span class="status status-api">{"🧠 ML: On" if ml_ok else "⚠️ Mode Secours (Pythagore)"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.title("🎯 Sniper Scanner V27 — ML First (Probabilité calibrée)")
    st.caption(
        "📦 Offline-First : lecture daily_projections (instantané)" if OFFLINE_MODE
        else "Moteur ML (model_proba.pkl) · Edge > 5% = 🎯 SNIPER TARGET · Fallback Pythagore si modèles absents"
    )
    if not OFFLINE_MODE and not ml_ok:
        st.warning("⚠️ **Mode Secours** : modèles .pkl non trouvés ou erreur de chargement. Prédictions via Pythagore.")

    if not db_ok:
        st.error("❌ Supabase indisponible. Vérifiez .env")
        st.stop()

    if OFFLINE_MODE:
        brain_mode = BRAIN_MODE_AUTO
    else:
        brain_mode = st.selectbox(
            "**Cerveau utilisé**",
            [
                (BRAIN_MODE_AUTO, "🔄 Auto (Standard sauf si outsider > 2.50 et P(outsider) > 40 %)"),
                (BRAIN_MODE_STANDARD, "🧠 Toujours Standard"),
                (BRAIN_MODE_CHASSEUR, "🔥 Prioriser Chasseur (si cote outsider > 2.50)"),
            ],
            format_func=lambda x: x[1],
            index=0,
            key="brain_mode",
        )
        brain_mode = brain_mode[0] if isinstance(brain_mode, tuple) else brain_mode

    if OFFLINE_MODE:
        df = build_sniper_table(brain_mode=brain_mode)
    else:
        with st.spinner("Chargement…"):
            df = build_sniper_table(brain_mode=brain_mode)

    if df.empty:
        st.warning("Aucun match sur les 3 prochains jours. Vérifiez les ligues ou la connexion API.")
        return

    st.success(f"✅ {len(df)} match(s) sur les {NEXT_DAYS_HORIZON} prochains jours")

    tab_scanner, tab_deep, tab_backtest = st.tabs(["📊 Sniper Scanner", "🔬 Deep Dive", "📈 Backtest"])

    with tab_scanner:
        st.subheader("🎯 Vainqueur (Moneyline)")
        display_cols = [
            "MATCH",
            "COTES (H/A)",
            "PROBA BOOK (%)",
            "PROBA SNIPER (%)",
            "EDGE (%)",
            "LE PARI",
            "Confiance",
            "Cerveau utilisé",
            "Style de Match",
            "🚨 ALERTE TRAPPE",
            "🎯 PARI OUTSIDER",
            "Message de Contexte",
            "Fiabilité",
        ]
        display_df = df[[c for c in display_cols if c in df.columns]]

        def _style_confiance(row):
            c = row.get("Confiance", "")
            trap = row.get("🚨 ALERTE TRAPPE", "")
            brain = row.get("Cerveau utilisé", "")
            if "SNIPER" in str(c):
                return [f"background-color: #7c3aed; color: white; font-weight: 700"] * len(row)
            if "MAX" in str(c):
                return [f"background-color: {COLOR_MAX_BET_BG}; color: {COLOR_MAX_BET}; font-weight: 700"] * len(row)
            if "VALUE" in str(c):
                return [f"background-color: rgba(34,197,94,0.2); color: {COLOR_VALUE}"] * len(row)
            if "Chasseur" in str(brain):
                return [f"background-color: rgba(234,88,12,0.2); color: #ea580c; font-weight: 600"] * len(row)
            if "Trap" in str(trap):
                return [f"background-color: rgba(249,115,22,0.25); color: #ea580c"] * len(row)
            return [f"color: {COLOR_PASS}"] * len(row)

        st.dataframe(
            display_df.style.apply(_style_confiance, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "MATCH": st.column_config.TextColumn("MATCH", width="large", help="Équipes · Heure"),
                "COTES (H/A)": st.column_config.TextColumn("COTES (H/A)", help="Cotes Betclic Domicile | Extérieur"),
                "PROBA BOOK (%)": st.column_config.TextColumn("PROBA BOOK (%)", help="Probabilité implicite 1/cote"),
                "PROBA SNIPER (%)": st.column_config.TextColumn("PROBA SNIPER (%)", help="Notre probabilité calibrée"),
                "EDGE (%)": st.column_config.TextColumn("EDGE (%)", help="(Proba × Cote) − 1. > 5% → 🎯 SNIPER TARGET"),
                "Style de Match": st.column_config.TextColumn("Style de Match", help="🔥 Shootout / 🛡️ Defensive Battle / 🏃 Run & Gun / 🔒 Verrou / ⚖️ Balanced"),
                "Cerveau utilisé": st.column_config.TextColumn("Cerveau utilisé", help="🧠 Standard = ML classique ; 🔥 Chasseur = modèle Upset (cote outsider > 2.50 et P(outsider) > 40%)"),
                "Proba ML": st.column_config.TextColumn("Proba ML", help="Probabilité Standard (domicile)"),
                "Proba calibrée": st.column_config.TextColumn("Proba calibrée", help="Taux réel observé (test) → utilisée pour l’Edge"),
                "🚨 ALERTE TRAPPE": st.column_config.TextColumn("🚨 ALERTE TRAPPE", help="Favori EuroLeague en trappe (< 72h repos)"),
                "🎯 PARI OUTSIDER": st.column_config.TextColumn("🎯 PARI OUTSIDER", help="Handicap positif si proba outsider 35–55%"),
                "Message de Contexte": st.column_config.TextColumn("Message de Contexte", help="Contexte fatigue / rotation"),
                "LE PARI": st.column_config.TextColumn("🎯 LE PARI", width="medium"),
                "Confiance": st.column_config.TextColumn("Confiance", help="🎯 SNIPER TARGET si Edge > 5%"),
                "Fiabilité": st.column_config.TextColumn("Fiabilité"),
            },
        )

        st.subheader("📊 Total Points (Over/Under)")
        mae_total: Optional[float] = None
        if FEATURES_META_TOTALS_PATH.exists():
            try:
                with open(FEATURES_META_TOTALS_PATH, "r", encoding="utf-8") as f:
                    meta_totals = json.load(f)
                mae_total = meta_totals.get("mae_total")
            except Exception:
                pass
        if mae_total is not None:
            if mae_total < 8:
                st.caption("🔥 **ML PROJECTED** — model_totals.pkl (MAE **%.1f** pts, excellent)." % mae_total)
            elif mae_total < 10:
                st.caption("🔥 **ML PROJECTED** — model_totals.pkl (MAE **%.1f** pts)." % mae_total)
            else:
                st.caption("Total : model_totals.pkl (MAE %.1f pts)." % mae_total)
        else:
            st.caption("Total : model_totals.pkl ou fallback (pas de constante 150).")
        df_totals = build_sniper_totals_table(df)

        def _style_ou_confiance(row):
            c = row.get("CONFIANCE", "")
            if "MAX" in str(c):
                return [f"background-color: {COLOR_MAX_BET_BG}; color: {COLOR_MAX_BET}; font-weight: 700"] * len(row)
            if "VALUE" in str(c):
                return [f"background-color: rgba(34,197,94,0.2); color: {COLOR_VALUE}"] * len(row)
            return [f"color: {COLOR_PASS}"] * len(row)

        if not df_totals.empty:
            display_totals = df_totals[["Match", "LIGNE BOOK", "PROJETÉ SNIPER", "DIFF", "STYLE", "PARI TOTAL", "CONFIANCE"]]
            st.dataframe(
                display_totals.style.apply(_style_ou_confiance, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Match": st.column_config.TextColumn("MATCH", width="large"),
                    "LIGNE BOOK": st.column_config.TextColumn("LIGNE BOOK", help="Ligne O/U bookmaker"),
                    "PROJETÉ SNIPER": st.column_config.NumberColumn("PROJETÉ SNIPER", format="%.1f", help="Total prédit par model_totals.pkl"),
                    "DIFF": st.column_config.TextColumn("DIFF", help="Écart projeté − ligne (ex: +7.5 pts)"),
                    "STYLE": st.column_config.TextColumn("STYLE", help="🔥 Shootout / 🛡️ Verrou / ⚖️ Équilibré"),
                    "PARI TOTAL": st.column_config.TextColumn("🎯 PARI TOTAL"),
                    "CONFIANCE": st.column_config.TextColumn("CONFIANCE", help="Indice basé sur écart et MAE modèle"),
                },
            )
        else:
            st.info("Aucune donnée Over/Under.")

    with tab_deep:
        st.subheader("Analyse Détaillée")
        opts = df["Match"].tolist()
        if not opts:
            st.info("Aucun match.")
        else:
            sel = st.selectbox("Match", opts, key="deep_match")
            if sel:
                row = df[df["Match"] == sel].iloc[0]
                home_id = int(row["_home_id"])
                away_id = int(row["_away_id"])
                home_name = str(row["_home_name"])
                away_name = str(row["_away_name"])
                home_st = row.get("_home_st")
                away_st = row.get("_away_st")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score projeté", f"{row['_proj_home']:.1f} — {row['_proj_away']:.1f}", "Total")
                with col2:
                    mode_s = "Mode Secours" if row.get("_mode_secours") else "ML"
                    st.metric("Proba Domicile", f"{row['_prob_home']*100:.1f}%", mode_s)
                with col3:
                    st.metric("Fair Spread", row.get("_fair_spread", "—"), "Book")

                # Pourquoi ce score ? — Pace vs Def_Rtg (défendable)
                st.markdown("**Pourquoi ce score ?**")
                if home_st and away_st:
                    pace_h = home_st.get("pace") or 72
                    pace_a = away_st.get("pace") or 72
                    def_h = home_st.get("def_rtg") or 100
                    def_a = away_st.get("def_rtg") or 100
                    off_h = home_st.get("off_rtg") or 100
                    off_a = away_st.get("off_rtg") or 100
                    pace_moy = (pace_h + pace_a) / 2
                    # Domicile marque contre la défense extérieure (Def_Rtg away)
                    raison_home = "défense adverse permissive" if def_a >= 108 else ("défense adverse solide" if def_a <= 100 else "défense adverse moyenne")
                    raison_away = "défense adverse permissive" if def_h >= 108 else ("défense adverse solide" if def_h <= 100 else "défense adverse moyenne")
                    st.markdown(
                        f"- **{home_name}** projeté **{row['_proj_home']:.0f} pts** : Def_Rtg de {away_name} = **{def_a:.1f}** → {raison_home}. "
                        f"Off_Rtg domicile {off_h:.1f}."
                    )
                    st.markdown(
                        f"- **{away_name}** projeté **{row['_proj_away']:.0f} pts** : Def_Rtg de {home_name} = **{def_h:.1f}** → {raison_away}. "
                        f"Off_Rtg extérieur {off_a:.1f}."
                    )
                    st.caption(f"Pace moyen du match : **{pace_moy:.1f}** poss. — Plus le Def_Rtg adverse est élevé, plus on marque.")
                else:
                    st.caption("Données Pace / Def_Rtg indisponibles pour ce match.")

                st.markdown("**Radar 4 Factors (Dean Oliver)**")
                fig = _plot_radar_four_factors(home_st, away_st, home_name, away_name)
                st.plotly_chart(fig, use_container_width=True)

                if home_st and away_st:
                    st.markdown("**Comparaison directe**")
                    comp = pd.DataFrame([
                        {"Metric": "Pace", "Domicile": home_st.get("pace"), "Extérieur": away_st.get("pace")},
                        {"Metric": "Off Rtg", "Domicile": home_st.get("off_rtg"), "Extérieur": away_st.get("off_rtg")},
                        {"Metric": "Def Rtg", "Domicile": home_st.get("def_rtg"), "Extérieur": away_st.get("def_rtg")},
                        {"Metric": "eFG%", "Domicile": f"{100*(home_st.get('efg_pct') or 0):.1f}%", "Extérieur": f"{100*(away_st.get('efg_pct') or 0):.1f}%"},
                    ])
                    st.dataframe(comp, use_container_width=True, hide_index=True)

                st.markdown("**3 derniers matchs — Box Scores**")
                col_home, col_away = st.columns(2)
                with col_home:
                    st.caption(f"🏠 {home_name}")
                    matches_home = _cold_get_last_n_matches_with_box(home_id, 3)
                    if matches_home:
                        df_home = pd.DataFrame(matches_home)
                        df_home_display = df_home.rename(columns={
                            "date": "Date", "opponent": "Adversaire", "score": "Score", "result": "R",
                            "pace": "Pace", "off_rtg": "Off", "def_rtg": "Def",
                            "efg_pct": "eFG%", "orb_pct": "ORB%", "tov_pct": "TOV%", "ft_rate": "FT Rate",
                        })
                        st.dataframe(df_home_display, use_container_width=True, hide_index=True, column_config={
                            "Date": st.column_config.TextColumn("Date"),
                            "Adversaire": st.column_config.TextColumn("Adversaire"),
                            "Score": st.column_config.TextColumn("Score"),
                            "R": st.column_config.TextColumn("R"),
                            "Pace": st.column_config.NumberColumn("Pace", format="%.1f"),
                            "Off": st.column_config.NumberColumn("Off", format="%.1f"),
                            "Def": st.column_config.NumberColumn("Def", format="%.1f"),
                            "eFG%": st.column_config.NumberColumn("eFG%", format="%.2f"),
                            "ORB%": st.column_config.NumberColumn("ORB%", format="%.2f"),
                            "TOV%": st.column_config.NumberColumn("TOV%", format="%.2f"),
                            "FT Rate": st.column_config.NumberColumn("FT Rate", format="%.2f"),
                        })
                    else:
                        st.info("Aucun box score récent.")
                with col_away:
                    st.caption(f"✈️ {away_name}")
                    matches_away = _cold_get_last_n_matches_with_box(away_id, 3)
                    if matches_away:
                        df_away = pd.DataFrame(matches_away)
                        df_away_display = df_away.rename(columns={
                            "date": "Date", "opponent": "Adversaire", "score": "Score", "result": "R",
                            "pace": "Pace", "off_rtg": "Off", "def_rtg": "Def",
                            "efg_pct": "eFG%", "orb_pct": "ORB%", "tov_pct": "TOV%", "ft_rate": "FT Rate",
                        })
                        st.dataframe(df_away_display, use_container_width=True, hide_index=True, column_config={
                            "Date": st.column_config.TextColumn("Date"),
                            "Adversaire": st.column_config.TextColumn("Adversaire"),
                            "Score": st.column_config.TextColumn("Score"),
                            "R": st.column_config.TextColumn("R"),
                            "Pace": st.column_config.NumberColumn("Pace", format="%.1f"),
                            "Off": st.column_config.NumberColumn("Off", format="%.1f"),
                            "Def": st.column_config.NumberColumn("Def", format="%.1f"),
                            "eFG%": st.column_config.NumberColumn("eFG%", format="%.2f"),
                            "ORB%": st.column_config.NumberColumn("ORB%", format="%.2f"),
                            "TOV%": st.column_config.NumberColumn("TOV%", format="%.2f"),
                            "FT Rate": st.column_config.NumberColumn("FT Rate", format="%.2f"),
                        })
                    else:
                        st.info("Aucun box score récent.")

                if get_tactical_clash:
                    st.markdown("**Alerte Kryptonite**")
                    clash = get_tactical_clash(home_id, away_id)
                    if clash.get("kryptonite"):
                        st.error(f"⚠️ {clash.get('kryptonite_msg', '')}")
                    else:
                        st.success("Pas de clash majeur.")
                    if clash.get("alert"):
                        st.warning(clash["alert"])

                fatigue_str = row.get("_fatigue", "")
                if fatigue_str:
                    st.markdown("**Fatigue**")
                    st.warning(fatigue_str)

    with tab_backtest:
        st.subheader("Backtest (Time Machine)")
        st.caption("Prédictions recréées avec données strictement antérieures à chaque match.")
        if st.button("Lancer le Backtest (3 derniers jours)"):
            with st.spinner("Backtest en cours…"):
                bt = run_backtest(days=3)
            st.session_state["backtest"] = bt
        bt = st.session_state.get("backtest")
        if bt:
            if bt.get("error"):
                st.error(bt["error"])
            else:
                n_c, n_t = bt["n_correct"], bt["n_total"]
                acc = (n_c / n_t * 100) if n_t > 0 else 0
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Paris corrects", f"{n_c}/{n_t}", f"{acc:.1f}%")
                with col2:
                    errs = [r.get("_spread_error") for r in bt["rows"] if r.get("_spread_error") is not None]
                    mae = sum(errs) / len(errs) if errs else 0
                    st.metric("MAE Spread", f"{mae:.1f} pts", "Erreur moyenne")
                df_bt = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in bt["rows"]])
                if not df_bt.empty:
                    st.dataframe(df_bt[["Match", "Date", "Prédiction", "Résultat Réel", "Verdict", "Erreur Spread"]], use_container_width=True, hide_index=True)
        else:
            st.info("Cliquez sur le bouton pour lancer le backtest.")


if __name__ == "__main__":
    main()
