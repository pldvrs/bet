#!/usr/bin/env python3
"""
Training Engine — ML pour prédiction Vainqueur + Spread (v2)
============================================================
- Exploitation totale de la base (pagination).
- Split temporel strict : Test = 2 derniers mois.
- Features avancées : Diff_OffDef, Pace_Match, Volatilité (std OffRtg last 5),
  Avantage terrain (HomeWin% last 2 seasons).
- StandardScaler pour normalisation.
- XGBoost / LightGBM avec TimeSeriesSplit CV.
- Calibration : "Quand le modèle dit 70%, gagne-t-on vraiment 70% du temps ?"
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from database import get_client

# Chemins des modèles sauvegardés (dossier du script)
# Suffixe _antitrap : nouvel entraînement (EuroLeague trap, upset) → ne pas écraser l'ancien
SCRIPT_DIR = Path(__file__).resolve().parent
ANTITRAP_SUFFIX = "_antitrap"
MODEL_PROBA_PATH = SCRIPT_DIR / "model_proba.pkl"
MODEL_SPREAD_PATH = SCRIPT_DIR / "model_spread.pkl"
MODEL_UPSET_PATH = SCRIPT_DIR / "model_upset.pkl"
MODEL_TOTALS_PATH = SCRIPT_DIR / "model_totals.pkl"
SCALER_TOTALS_PATH = SCRIPT_DIR / "scaler_totals.pkl"
SCALER_PATH = SCRIPT_DIR / "scaler.pkl"
FEATURES_META_PATH = SCRIPT_DIR / "features_meta.json"
FEATURES_META_TOTALS_PATH = SCRIPT_DIR / "features_meta_totals.json"
# Fichiers écrits par l'entraînement Anti-Trap (noms différents = ancien entraînement préservé)
MODEL_PROBA_ANTITRAP_PATH = SCRIPT_DIR / f"model_proba{ANTITRAP_SUFFIX}.pkl"
MODEL_SPREAD_ANTITRAP_PATH = SCRIPT_DIR / f"model_spread{ANTITRAP_SUFFIX}.pkl"
SCALER_ANTITRAP_PATH = SCRIPT_DIR / f"scaler{ANTITRAP_SUFFIX}.pkl"
FEATURES_META_ANTITRAP_PATH = SCRIPT_DIR / f"features_meta{ANTITRAP_SUFFIX}.json"

# Paramètres feature engineering
N_LAST_GAMES: int = 10
N_LAST_FOR_VOLATILITY: int = 5
MIN_GAMES_FOR_ROW: int = 3
BASELINE_PACE: float = 72.0
BASELINE_OFF: float = 100.0
BASELINE_DEF: float = 100.0
BASELINE_EFG: float = 0.5
BASELINE_ORB: float = 0.25
BASELINE_TOV: float = 0.15
BASELINE_FT: float = 0.25

# Split temporel : Test = 2 derniers mois (conditions réelles)
TEST_MONTHS: int = 2
TS_CV_SPLITS: int = 5

# Facteur EuroLeague / Anti-Trap
EUROLEAGUE_LEAGUE_ID: int = 120
DOMESTIC_LEAGUE_IDS: Tuple[int, ...] = (2, 5, 52, 4, 8, 194, 198, 45, 206)
TRAP_REST_HOURS: float = 72.0  # Moins de 72h = trappe
INTENSITY_LOOKBACK_DAYS: int = 10
PACE_UNDERDOG_SLOW: float = 68.0  # Outsider joue lent → style clash
UPSET_IMPLIED_PROB_MAX: float = 0.40  # Cote > 2.50 ≈ prob < 0.40


# ==============================================================================
# CONNEXION & CHARGEMENT BRUT (pagination = intégralité de la base)
# ==============================================================================

CHUNK_SIZE: int = 1000


def _get_supabase():
    return get_client()


def _fetch_table_paginated(
    table: str,
    select_cols: str = "*",
    order_col: str = "date",
    ascending: bool = True,
    extra_filters: Optional[Any] = None,
) -> pd.DataFrame:
    """Récupère l'intégralité d'une table Supabase par chunks (pas de limite implicite)."""
    supabase = _get_supabase()
    if not supabase:
        return pd.DataFrame()
    all_data: List[dict] = []
    offset = 0
    try:
        while True:
            q = (
                supabase.table(table)
                .select(select_cols)
                .order(order_col, desc=not ascending)
                .range(offset, offset + CHUNK_SIZE - 1)
            )
            if extra_filters:
                q = extra_filters(q)
            r = q.execute()
            chunk = r.data or []
            if not chunk:
                break
            all_data.extend(chunk)
            if len(chunk) < CHUNK_SIZE:
                break
            offset += CHUNK_SIZE
        return pd.DataFrame(all_data)
    except Exception:
        return pd.DataFrame()


def _fetch_games_with_scores() -> pd.DataFrame:
    """Tous les matchs terminés (home_score et away_score non null), ordonnés par date."""
    supabase = _get_supabase()
    if not supabase:
        return pd.DataFrame()
    all_data: List[dict] = []
    offset = 0
    try:
        while True:
            r = (
                supabase.table("games_history")
                .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score")
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .order("date", desc=False)
                .range(offset, offset + CHUNK_SIZE - 1)
                .execute()
            )
            chunk = r.data or []
            if not chunk:
                break
            all_data.extend(chunk)
            if len(chunk) < CHUNK_SIZE:
                break
            offset += CHUNK_SIZE
        return pd.DataFrame(all_data)
    except Exception:
        return pd.DataFrame()


def _fetch_all_box_scores() -> pd.DataFrame:
    """Tous les box_scores (pagination pour intégralité)."""
    return _fetch_table_paginated("box_scores", "*", "game_id", ascending=True)


def _fetch_all_games_for_context() -> pd.DataFrame:
    """Tous les matchs pour Win_Rate, Streak, H2H (pagination)."""
    return _fetch_table_paginated("games_history", "game_id, date, league_id, season, home_id, away_id, home_score, away_score", "date", ascending=True)


# ==============================================================================
# FEATURE ENGINEERING (N-10 + Contexte)
# ==============================================================================


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _avg_last_n(df: pd.DataFrame, team_id: int, before_date: str, cols: List[str], n: int) -> Dict[str, float]:
    """
    Moyennes sur les N derniers matchs du team_id avant before_date.
    df = box_scores avec colonne date (string YYYY-MM-DD).
    Retourne un dict {col: mean} avec default si vide.
    """
    df_team = df[(df["team_id"] == team_id) & (df["date"].astype(str) < before_date)].copy()
    df_team = df_team.sort_values("date", ascending=False).head(n)
    out: Dict[str, float] = {}
    defaults = {
        "pace": BASELINE_PACE,
        "off_rtg": BASELINE_OFF,
        "def_rtg": BASELINE_DEF,
        "efg_pct": BASELINE_EFG,
        "orb_pct": BASELINE_ORB,
        "tov_pct": BASELINE_TOV,
        "ft_rate": BASELINE_FT,
    }
    for col in cols:
        if col not in df_team.columns:
            out[col] = defaults.get(col, 0.0)
            continue
        vals = df_team[col].replace([np.inf, -np.inf], np.nan).dropna()
        out[col] = float(vals.mean()) if len(vals) > 0 else defaults.get(col, 0.0)
    return out


def _std_last_n(df: pd.DataFrame, team_id: int, before_date: str, col: str, n: int) -> float:
    """Écart-type sur les N derniers matchs (ex: Off_Rtg pour volatilité). 0 si < 2 valeurs."""
    df_team = df[(df["team_id"] == team_id) & (df["date"].astype(str) < before_date)].copy()
    df_team = df_team.sort_values("date", ascending=False).head(n)
    if col not in df_team.columns:
        return 0.0
    vals = df_team[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.std()) if len(vals) >= 2 else 0.0


def _previous_season(season_str: str) -> str:
    """Ex: '2024-2025' -> '2023-2024', '2024' -> '2023'."""
    s = str(season_str).strip()
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            try:
                y1, y2 = int(parts[0]), int(parts[1])
                return f"{y1 - 1}-{y2 - 1}"
            except ValueError:
                pass
    try:
        return str(int(s) - 1)
    except ValueError:
        return s


def _home_win_rate_at_home_last_2_seasons(
    team_id: int,
    game_date: str,
    league_id: Any,
    current_season: str,
    df_games: pd.DataFrame,
) -> float:
    """% de victoires à domicile de l'équipe sur les 2 dernières saisons (avant game_date)."""
    prev_season = _previous_season(current_season)
    seasons = [str(current_season), str(prev_season)]
    mask = (
        (df_games["league_id"] == league_id)
        & (df_games["season"].astype(str).isin(seasons))
        & (df_games["date"].astype(str) < game_date)
        & (df_games["home_id"] == team_id)
        & df_games["home_score"].notna()
        & df_games["away_score"].notna()
    )
    sub = df_games.loc[mask]
    if sub.empty:
        return 0.5
    wins = (sub["home_score"] > sub["away_score"]).sum()
    return float(wins / len(sub))


def _days_rest(team_id: int, game_date: str, df_games: pd.DataFrame) -> Optional[int]:
    """Jours de repos avant game_date pour team_id (dernier match avant cette date)."""
    before = df_games[
        ((df_games["home_id"] == team_id) | (df_games["away_id"] == team_id))
        & (df_games["date"].astype(str) < game_date)
    ].copy()
    if before.empty:
        return None
    before = before.sort_values("date", ascending=False)
    last_date = str(before.iloc[0]["date"])[:10]
    try:
        from datetime import datetime
        d1 = datetime.strptime(last_date, "%Y-%m-%d").date()
        d2 = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
        return (d2 - d1).days
    except Exception:
        return None


def _win_rate_season(team_id: int, game_date: str, league_id: Any, season: Any, df_games: pd.DataFrame) -> float:
    """Taux de victoires de l'équipe dans la saison (league_id + season) avant game_date."""
    mask = (
        (df_games["league_id"] == league_id)
        & (df_games["season"].astype(str) == str(season))
        & (df_games["date"].astype(str) < game_date)
        & df_games["home_score"].notna()
        & df_games["away_score"].notna()
    )
    sub = df_games.loc[mask]
    if sub.empty:
        return 0.5
    wins = 0
    total = 0
    for _, row in sub.iterrows():
        if row["home_id"] == team_id or row["away_id"] == team_id:
            total += 1
            if row["home_id"] == team_id and row["home_score"] > row["away_score"]:
                wins += 1
            elif row["away_id"] == team_id and row["away_score"] > row["home_score"]:
                wins += 1
    return wins / total if total > 0 else 0.5


def _streak(team_id: int, game_date: str, df_games: pd.DataFrame) -> int:
    """
    Série en cours : positif = victoires, négatif = défaites.
    Compte à partir du dernier match avant game_date.
    """
    before = df_games[
        ((df_games["home_id"] == team_id) | (df_games["away_id"] == team_id))
        & (df_games["date"].astype(str) < game_date)
        & df_games["home_score"].notna()
    ].sort_values("date", ascending=False)
    if before.empty:
        return 0
    streak = 0
    for _, row in before.iterrows():
        won = (row["home_id"] == team_id and row["home_score"] > row["away_score"]) or (
            row["away_id"] == team_id and row["away_score"] > row["home_score"]
        )
        if streak == 0:
            streak = 1 if won else -1
        elif (streak > 0 and won) or (streak < 0 and not won):
            streak = streak + 1 if won else streak - 1
        else:
            break
    return streak


def _last_h2h_winner(home_id: int, away_id: int, game_date: str, df_games: pd.DataFrame) -> float:
    """
    Dernière confrontation : 1 = domicile a gagné, 0 = extérieur a gagné, 0.5 = pas de H2H.
    """
    h2h = df_games[
        (((df_games["home_id"] == home_id) & (df_games["away_id"] == away_id))
         | ((df_games["home_id"] == away_id) & (df_games["away_id"] == home_id)))
        & (df_games["date"].astype(str) < game_date)
        & df_games["home_score"].notna()
    ].sort_values("date", ascending=False)
    if h2h.empty:
        return 0.5
    row = h2h.iloc[0]
    if row["home_id"] == home_id and row["away_id"] == away_id:
        return 1.0 if row["home_score"] > row["away_score"] else 0.0
    return 0.0 if row["home_score"] > row["away_score"] else 1.0  # away was home in that game, so home (our home_id) won iff away_score > home_score


# ==============================================================================
# FEATURE ENGINEERING — Facteur EuroLeague / Anti-Trap
# ==============================================================================


def _last_game_league_and_rest_hours(
    team_id: int, game_date: str, df_games: pd.DataFrame
) -> Tuple[Optional[int], Optional[float]]:
    """Dernier match du team_id avant game_date : (league_id du match, rest en heures)."""
    before = df_games[
        ((df_games["home_id"] == team_id) | (df_games["away_id"] == team_id))
        & (df_games["date"].astype(str) < game_date)
    ].sort_values("date", ascending=False)
    if before.empty:
        return None, None
    row = before.iloc[0]
    last_date = str(row["date"])[:10]
    try:
        d1 = datetime.strptime(last_date, "%Y-%m-%d")
        d2 = datetime.strptime(game_date[:10], "%Y-%m-%d")
        rest_hours = (d2 - d1).days * 24.0
    except Exception:
        rest_hours = None
    league_id = row.get("league_id")
    if league_id is not None:
        league_id = int(league_id)
    return league_id, rest_hours


def _n_euroleague_games_last_n_days(
    team_id: int, game_date: str, df_games: pd.DataFrame, n_days: int = INTENSITY_LOOKBACK_DAYS
) -> int:
    """Nombre de matchs EuroLeague (league_id 120) du team_id dans les n derniers jours."""
    try:
        d2 = datetime.strptime(game_date[:10], "%Y-%m-%d")
        d1 = d2 - timedelta(days=n_days)
        date_lo = d1.strftime("%Y-%m-%d")
    except Exception:
        return 0
    sub = df_games[
        ((df_games["home_id"] == team_id) | (df_games["away_id"] == team_id))
        & (df_games["date"].astype(str) >= date_lo)
        & (df_games["date"].astype(str) < game_date)
        & (df_games["league_id"] == EUROLEAGUE_LEAGUE_ID)
    ]
    return len(sub)


def _is_domestic_league(league_id: Any) -> bool:
    """True si ligue domestique (pas EuroLeague)."""
    try:
        lid = int(league_id)
        return lid in DOMESTIC_LEAGUE_IDS
    except (TypeError, ValueError):
        return False


def _is_euroleague_team_recent(team_id: int, game_date: str, df_games: pd.DataFrame) -> bool:
    """True si l'équipe a joué en EuroLeague dans les 30 derniers jours."""
    last_league, rest_h = _last_game_league_and_rest_hours(team_id, game_date, df_games)
    if last_league == EUROLEAGUE_LEAGUE_ID:
        return True
    n_el = _n_euroleague_games_last_n_days(team_id, game_date, df_games, n_days=30)
    return n_el > 0


def _is_domestic_trap_team(
    team_id: int, game_date: str, current_league_id: Any, df_games: pd.DataFrame
) -> bool:
    """True si équipe EuroLeague joue un match domestique avec moins de 72h de repos."""
    if not _is_domestic_league(current_league_id):
        return False
    last_league, rest_hours = _last_game_league_and_rest_hours(team_id, game_date, df_games)
    if rest_hours is None:
        return False
    if rest_hours >= TRAP_REST_HOURS:
        return False
    return last_league == EUROLEAGUE_LEAGUE_ID


def _intensity_proxy(
    team_id: int, game_date: str, current_league_id: Any, df_games: pd.DataFrame
) -> float:
    """
    Score 0–1 : plus c'est bas, plus l'équipe est fatiguée (EuroLeague récente) pour un match domestique.
    Une équipe qui enchaîne 3 matchs EL en 10 jours a un score bas pour le championnat national.
    """
    if not _is_domestic_league(current_league_id):
        return 1.0
    n_el = _n_euroleague_games_last_n_days(team_id, game_date, df_games, INTENSITY_LOOKBACK_DAYS)
    # 0 EL → 1.0, 1 → ~0.67, 2 → ~0.33, 3+ → 0
    return max(0.0, 1.0 - n_el / 3.0)


def _underdog_style_clash(
    win_rate_home: float, win_rate_away: float, pace_home: float, pace_away: float
) -> float:
    """
     Underdog = équipe avec plus faible win rate. Si son Pace < 68, style clash (hold-up possible).
    Retourne 1.0 si outsider joue très lent, 0.0 sinon.
    """
    if win_rate_home <= win_rate_away:
        underdog_pace = pace_home
    else:
        underdog_pace = pace_away
    return 1.0 if underdog_pace < PACE_UNDERDOG_SLOW else 0.0


def _league_avg_total(league_id: Any, game_date: str, df_games: pd.DataFrame) -> float:
    """Moyenne historique des points totaux (home_score + away_score) dans cette ligue avant game_date."""
    mask = (
        (df_games["league_id"] == league_id)
        & (df_games["date"].astype(str) < game_date)
        & df_games["home_score"].notna()
        & df_games["away_score"].notna()
    )
    sub = df_games.loc[mask]
    if sub.empty:
        # Fallback : moyenne globale des matchs terminés (pas de constante 150)
        ok = df_games["home_score"].notna() & df_games["away_score"].notna()
        if not ok.any():
            return 165.0
        global_totals = df_games.loc[ok, "home_score"].astype(float) + df_games.loc[ok, "away_score"].astype(float)
        return float(global_totals.mean())
    totals = sub["home_score"].astype(float) + sub["away_score"].astype(float)
    return float(totals.mean())


def _league_avg_pace(
    league_id: Any, game_date: str, df_box: pd.DataFrame, df_games: pd.DataFrame
) -> float:
    """Moyenne du pace (possessions) dans la ligue avant game_date. Utilisé pour Pace Factor."""
    if df_box.empty or df_games.empty or "game_id" not in df_box.columns or "league_id" not in df_games.columns:
        return BASELINE_PACE
    df_g = df_games[["game_id", "league_id", "date"]].copy().drop_duplicates("game_id")
    df_g["date"] = df_g["date"].astype(str).str[:10]
    merge = df_box[["game_id", "pace"]].copy()
    merge = merge.merge(df_g, on="game_id", how="inner")
    date_col = "date"
    if date_col not in merge.columns:
        return BASELINE_PACE
    sub = merge[(merge["league_id"] == league_id) & (merge[date_col].astype(str) < game_date)]
    if sub.empty or "pace" not in sub.columns:
        return BASELINE_PACE
    vals = sub["pace"].replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) > 0 else BASELINE_PACE


def _avg_opponent_metric_last_n(
    df_box: pd.DataFrame,
    df_games: pd.DataFrame,
    team_id: int,
    before_date: str,
    metric_col: str,
    n: int,
) -> float:
    """
    Moyenne d'une métrique des *adversaires* sur les N derniers matchs de team_id.
    - metric_col='efg_pct' → Opponent eFG% (comment la défense limite le tir adverse).
    - metric_col='tov_pct' → Turnover Forced (TOV% des adversaires face à cette équipe).
    """
    if df_games.empty or df_box.empty or metric_col not in df_box.columns:
        return BASELINE_EFG if metric_col == "efg_pct" else BASELINE_TOV
    df_g = df_games.copy()
    df_g["date"] = df_g["date"].astype(str).str[:10]
    team_games = df_g[
        ((df_g["home_id"] == team_id) | (df_g["away_id"] == team_id))
        & (df_g["date"] < before_date)
    ].sort_values("date", ascending=False).head(n)
    if team_games.empty:
        return BASELINE_EFG if metric_col == "efg_pct" else BASELINE_TOV
    vals: List[float] = []
    default = BASELINE_EFG if metric_col == "efg_pct" else BASELINE_TOV
    for _, row in team_games.iterrows():
        gid = row["game_id"]
        opp_id = int(row["away_id"]) if row["home_id"] == team_id else int(row["home_id"])
        br = df_box[(df_box["game_id"] == gid) & (df_box["team_id"] == opp_id)]
        if not br.empty and metric_col in br.columns:
            v = br[metric_col].iloc[0]
            if np.isfinite(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else default


# Features spécifiques au modèle Total Points (régression score cumulé)
# Style & Défense : Def_Rtg, Opponent eFG%, Pace Factor, Turnover Forced + match-up
FEATURE_NAMES_TOTALS: List[str] = [
    "Combined_Pace",           # Somme des Pace N-10 (Home + Away)
    "Combined_OffRtg",         # Efficacité offensive cumulée (OffRtg Home + OffRtg Away)
    "Combined_eFG_Pct",        # adresse cumulée prévue
    "Combined_TOV_Pct",        # plus de pertes = moins de tirs = moins de points
    "Rest_Total",              # somme des jours de repos (fatigue)
    "League_Avg_Total",        # moyenne historique points de la ligue spécifique
    # Style & Défense (obligatoire)
    "Def_Rtg_Home",            # Points encaissés / 100 poss. (solidité défensive)
    "Def_Rtg_Away",
    "Combined_Def_Rtg",        # (Def_Rtg_Home + Def_Rtg_Away) / 2 — match-up défensif
    "Opponent_eFG_Home",       # eFG% des adversaires face au domicile (défense gêne le tir)
    "Opponent_eFG_Away",
    "Turnover_Forced_Home",    # TOV% des adversaires face au domicile (défense provoque pertes)
    "Turnover_Forced_Away",
    "Pace_Factor_Home",        # Pace équipe / League_Avg_Pace (>1 = Run & Gun)
    "Pace_Factor_Away",
    "Pace_Factor_Match",       # Moyenne des deux — Fast vs Fast → total explosif
]


# ==============================================================================
# CONSTRUCTION DU DATASET
# ==============================================================================


# Ordre des features (doit être identique à l'entraînement et à l'inférence)
FEATURE_NAMES: List[str] = [
    "Home_Avg_Off_Rtg_Last10",
    "Home_Avg_Def_Rtg_Last10",
    "Home_Avg_Pace_Last10",
    "Home_Avg_eFG_Last10",
    "Home_Avg_TOV_Last10",
    "Home_Avg_ORB_Last10",
    "Home_Avg_FT_Rate_Last10",
    "Away_Avg_Off_Rtg_Last10",
    "Away_Avg_Def_Rtg_Last10",
    "Away_Avg_Pace_Last10",
    "Away_Avg_eFG_Last10",
    "Away_Avg_TOV_Last10",
    "Away_Avg_ORB_Last10",
    "Away_Avg_FT_Rate_Last10",
    "Diff_OffDef_Home",   # Home_Off - Away_Def (avantage attaque domicile vs défense extérieur)
    "Diff_OffDef_Away",   # Away_Off - Home_Def
    "Pace_Match",         # Moyenne pace prévue pour le match
    "Home_OffRtg_Std_Last5",  # Volatilité attaque domicile
    "Away_OffRtg_Std_Last5",  # Volatilité attaque extérieur
    "Home_HomeWinPct_Last2Seasons",  # Avantage terrain réel (domicile)
    "Away_HomeWinPct_Last2Seasons",  # Idem pour l'équipe extérieur (quand elle jouait à domicile)
    "Home_Rest_Days",
    "Away_Rest_Days",
    "Home_Win_Rate_Season",
    "Away_Win_Rate_Season",
    "Home_Streak",
    "Away_Streak",
    "H2H_Last_Home_Win",
    # Anti-Trap / EuroLeague
    "Is_Domestic_Trap",       # 1 si favori EuroLeague joue domestique avec < 72h repos
    "Intensity_Proxy_Home",   # 0–1 : fatigue cumulative (EL récent) pour match domestique
    "Intensity_Proxy_Away",
    "Underdog_Style_Clash",   # 1 si outsider joue très lent (Pace < 68)
]


def _build_row(
    game_date: str,
    home_id: int,
    away_id: int,
    league_id: Any,
    season: Any,
    df_box: pd.DataFrame,
    df_games: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Construit une ligne de features pour un match (données strictement avant game_date)."""
    box_cols = ["pace", "off_rtg", "def_rtg", "efg_pct", "orb_pct", "tov_pct", "ft_rate"]
    if "date" not in df_box.columns:
        return None
    df_box["date"] = df_box["date"].astype(str).str[:10]

    home_avg = _avg_last_n(df_box, home_id, game_date, box_cols, N_LAST_GAMES)
    away_avg = _avg_last_n(df_box, away_id, game_date, box_cols, N_LAST_GAMES)

    n_home = len(df_box[(df_box["team_id"] == home_id) & (df_box["date"] < game_date)])
    n_away = len(df_box[(df_box["team_id"] == away_id) & (df_box["date"] < game_date)])
    if n_home < MIN_GAMES_FOR_ROW or n_away < MIN_GAMES_FOR_ROW:
        return None

    rest_h = _days_rest(home_id, game_date, df_games)
    rest_a = _days_rest(away_id, game_date, df_games)
    win_rate_h = _win_rate_season(home_id, game_date, league_id, season, df_games)
    win_rate_a = _win_rate_season(away_id, game_date, league_id, season, df_games)
    streak_h = _streak(home_id, game_date, df_games)
    streak_a = _streak(away_id, game_date, df_games)
    h2h = _last_h2h_winner(home_id, away_id, game_date, df_games)

    # Différentiels Off/Def (ce qui fait gagner)
    diff_offdef_home = home_avg["off_rtg"] - away_avg["def_rtg"]
    diff_offdef_away = away_avg["off_rtg"] - home_avg["def_rtg"]
    pace_match = (home_avg["pace"] + away_avg["pace"]) / 2.0
    home_off_std = _std_last_n(df_box, home_id, game_date, "off_rtg", N_LAST_FOR_VOLATILITY)
    away_off_std = _std_last_n(df_box, away_id, game_date, "off_rtg", N_LAST_FOR_VOLATILITY)
    home_home_win_pct = _home_win_rate_at_home_last_2_seasons(home_id, game_date, league_id, season, df_games)
    away_home_win_pct = _home_win_rate_at_home_last_2_seasons(away_id, game_date, league_id, season, df_games)

    # Facteur EuroLeague / Anti-Trap
    is_trap_h = _is_domestic_trap_team(home_id, game_date, league_id, df_games)
    is_trap_a = _is_domestic_trap_team(away_id, game_date, league_id, df_games)
    is_domestic_trap = 1.0 if (is_trap_h or is_trap_a) else 0.0
    intensity_home = _intensity_proxy(home_id, game_date, league_id, df_games)
    intensity_away = _intensity_proxy(away_id, game_date, league_id, df_games)
    underdog_style_clash = _underdog_style_clash(
        win_rate_h, win_rate_a, home_avg["pace"], away_avg["pace"]
    )

    # Features spécifiques Total Points (régression score cumulé)
    combined_pace = home_avg["pace"] + away_avg["pace"]  # Somme des Pace N-10
    combined_off_rtg = home_avg["off_rtg"] + away_avg["off_rtg"]
    combined_efg_pct = (home_avg["efg_pct"] + away_avg["efg_pct"]) / 2.0
    combined_tov_pct = (home_avg["tov_pct"] + away_avg["tov_pct"]) / 2.0
    rest_total = (float(rest_h) if rest_h is not None else 3.0) + (float(rest_a) if rest_a is not None else 3.0)
    league_avg_total = _league_avg_total(league_id, game_date, df_games)

    # Style & Défense : Def_Rtg, Opponent eFG%, Turnover Forced, Pace Factor
    league_avg_pace = _league_avg_pace(league_id, game_date, df_box, df_games)
    opp_efg_home = _avg_opponent_metric_last_n(df_box, df_games, home_id, game_date, "efg_pct", N_LAST_GAMES)
    opp_efg_away = _avg_opponent_metric_last_n(df_box, df_games, away_id, game_date, "efg_pct", N_LAST_GAMES)
    opp_tov_home = _avg_opponent_metric_last_n(df_box, df_games, home_id, game_date, "tov_pct", N_LAST_GAMES)
    opp_tov_away = _avg_opponent_metric_last_n(df_box, df_games, away_id, game_date, "tov_pct", N_LAST_GAMES)
    pace_factor_home = (home_avg["pace"] / league_avg_pace) if league_avg_pace and league_avg_pace > 0 else 1.0
    pace_factor_away = (away_avg["pace"] / league_avg_pace) if league_avg_pace and league_avg_pace > 0 else 1.0
    combined_def_rtg = (home_avg["def_rtg"] + away_avg["def_rtg"]) / 2.0
    pace_factor_match = (pace_factor_home + pace_factor_away) / 2.0

    return {
        "game_id": None,
        "date": game_date,
        "Home_Avg_Off_Rtg_Last10": home_avg["off_rtg"],
        "Home_Avg_Def_Rtg_Last10": home_avg["def_rtg"],
        "Home_Avg_Pace_Last10": home_avg["pace"],
        "Home_Avg_eFG_Last10": home_avg["efg_pct"],
        "Home_Avg_TOV_Last10": home_avg["tov_pct"],
        "Home_Avg_ORB_Last10": home_avg["orb_pct"],
        "Home_Avg_FT_Rate_Last10": home_avg["ft_rate"],
        "Away_Avg_Off_Rtg_Last10": away_avg["off_rtg"],
        "Away_Avg_Def_Rtg_Last10": away_avg["def_rtg"],
        "Away_Avg_Pace_Last10": away_avg["pace"],
        "Away_Avg_eFG_Last10": away_avg["efg_pct"],
        "Away_Avg_TOV_Last10": away_avg["tov_pct"],
        "Away_Avg_ORB_Last10": away_avg["orb_pct"],
        "Away_Avg_FT_Rate_Last10": away_avg["ft_rate"],
        "Diff_OffDef_Home": diff_offdef_home,
        "Diff_OffDef_Away": diff_offdef_away,
        "Pace_Match": pace_match,
        "Home_OffRtg_Std_Last5": home_off_std,
        "Away_OffRtg_Std_Last5": away_off_std,
        "Home_HomeWinPct_Last2Seasons": home_home_win_pct,
        "Away_HomeWinPct_Last2Seasons": away_home_win_pct,
        "Home_Rest_Days": float(rest_h) if rest_h is not None else 3.0,
        "Away_Rest_Days": float(rest_a) if rest_a is not None else 3.0,
        "Home_Win_Rate_Season": win_rate_h,
        "Away_Win_Rate_Season": win_rate_a,
        "Home_Streak": float(streak_h),
        "Away_Streak": float(streak_a),
        "H2H_Last_Home_Win": h2h,
        "Is_Domestic_Trap": is_domestic_trap,
        "Intensity_Proxy_Home": intensity_home,
        "Intensity_Proxy_Away": intensity_away,
        "Underdog_Style_Clash": underdog_style_clash,
        # Total Points (model_totals)
        "Combined_Pace": combined_pace,
        "Combined_OffRtg": combined_off_rtg,
        "Combined_eFG_Pct": combined_efg_pct,
        "Combined_TOV_Pct": combined_tov_pct,
        "Rest_Total": rest_total,
        "League_Avg_Total": league_avg_total,
        # Style & Défense
        "Def_Rtg_Home": home_avg["def_rtg"],
        "Def_Rtg_Away": away_avg["def_rtg"],
        "Combined_Def_Rtg": combined_def_rtg,
        "Opponent_eFG_Home": opp_efg_home,
        "Opponent_eFG_Away": opp_efg_away,
        "Turnover_Forced_Home": opp_tov_home,
        "Turnover_Forced_Away": opp_tov_away,
        "Pace_Factor_Home": pace_factor_home,
        "Pace_Factor_Away": pace_factor_away,
        "Pace_Factor_Match": pace_factor_match,
        # Alias pour modèle Totals simplifié (Pace_Home, Pace_Away, Def_Rtg_Home, Def_Rtg_Away)
        "Pace_Home": home_avg["pace"],
        "Pace_Away": away_avg["pace"],
        "Def_Rtg_Home": home_avg["def_rtg"],
        "Def_Rtg_Away": away_avg["def_rtg"],
    }


def build_training_dataset() -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parcourt tous les matchs terminés, génère les features (N-10 + contexte).
    Retourne (DataFrame avec X + Win_Home + Score_Diff, error_message).
    """
    df_games = _fetch_games_with_scores()
    df_box = _fetch_all_box_scores()
    df_all_games = _fetch_all_games_for_context()

    if df_games.empty:
        return pd.DataFrame(), "Aucun match terminé en base"
    if df_box.empty:
        return pd.DataFrame(), "Aucun box_score en base"

    if "date" not in df_box.columns:
        supabase = _get_supabase()
        if not supabase:
            return pd.DataFrame(), "Supabase indisponible"
        df_gh = supabase.table("games_history").select("game_id, date").execute().data or []
        df_gh = pd.DataFrame(df_gh)
        df_box = df_box.merge(df_gh, on="game_id", how="left")
    date_col = df_box.get("date")
    if date_col is None and "date_x" in df_box.columns and "date_y" in df_box.columns:
        date_col = df_box["date_x"].combine_first(df_box["date_y"])
    elif date_col is None:
        date_col = df_box.get("date_x", df_box.get("date_y", pd.Series()))
    df_box["date"] = pd.to_datetime(date_col, errors="coerce").dt.strftime("%Y-%m-%d")
    df_all_games["date"] = pd.to_datetime(df_all_games["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    rows: List[Dict[str, Any]] = []
    for _, g in df_games.iterrows():
        game_date = str(g.get("date") or "")[:10]
        if len(game_date) != 10:
            continue
        home_id = int(g.get("home_id", 0))
        away_id = int(g.get("away_id", 0))
        league_id = g.get("league_id")
        season = g.get("season", "")
        if not home_id or not away_id:
            continue

        row = _build_row(game_date, home_id, away_id, league_id, season, df_box.copy(), df_all_games)
        if row is None:
            continue

        row["game_id"] = g.get("game_id")
        row["league_id"] = g.get("league_id")
        row["Win_Home"] = 1 if (g["home_score"] > g["away_score"]) else 0
        row["Score_Diff"] = float(g["home_score"] - g["away_score"])
        row["Total_Points"] = float(g["home_score"]) + float(g["away_score"])

        rows.append(row)

    if not rows:
        return pd.DataFrame(), "Aucune ligne générée (pas assez de matchs passés par équipe)"

    df = pd.DataFrame(rows)

    # Imputation : remplacer NaN par la moyenne de la colonne
    for col in FEATURE_NAMES:
        if col not in df.columns:
            continue
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    for col in FEATURE_NAMES_TOTALS:
        if col not in df.columns:
            continue
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    df = df.fillna(0)

    return df, None


# ==============================================================================
# ENTRAÎNEMENT
# ==============================================================================


def _get_classifier():
    """LightGBM > XGBoost > RandomForest (relations non-linéaires, évite libomp sur Mac)."""
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
    except Exception:
        pass
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_estimators=200, max_depth=8)
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)


def _get_regressor():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
    except Exception:
        pass
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=200, max_depth=8)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)


def train_and_evaluate(
    df: pd.DataFrame,
    test_months: int = TEST_MONTHS,
) -> Dict[str, Any]:
    """
    Split temporel strict : Test = les N derniers mois (conditions réelles).
    StandardScaler pour que Pace (70-80) ne domine pas eFG% (0.4-0.6).
    TimeSeriesSplit CV pour éviter l'overfitting.
    Calibration : quand le modèle dit 70%, gagne-t-on vraiment 70% du temps ?
    """
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    if df.empty or "Win_Home" not in df.columns or "Score_Diff" not in df.columns:
        return {"error": "Dataset invalide (Win_Home / Score_Diff manquants)"}

    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    max_date = df["date"].max()
    if pd.isna(max_date):
        return {"error": "Dates invalides"}
    test_start = max_date - timedelta(days=test_months * 31)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()

    if len(train_df) < 50:
        return {"error": "Pas assez de données pour l'entraînement (min 50 lignes)"}
    if len(test_df) < 5:
        return {"error": "Pas assez de matchs dans les 2 derniers mois pour le test"}

    X_train = train_df[FEATURE_NAMES].copy()
    y_win_train = train_df["Win_Home"]
    y_diff_train = train_df["Score_Diff"]
    X_test = test_df[FEATURE_NAMES].copy()
    y_win_test = test_df["Win_Home"]
    y_diff_test = test_df["Score_Diff"]

    for col in FEATURE_NAMES:
        if col in X_train.columns and X_train[col].isna().any():
            m = X_train[col].mean()
            X_train[col] = X_train[col].fillna(m)
            X_test[col] = X_test[col].fillna(m)
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    # Normalisation : StandardScaler (fit sur train uniquement)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Pour les modèles qui attendent des DataFrame (noms de colonnes)
    X_train_s = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES, index=X_train.index)
    X_test_s = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES, index=X_test.index)

    # TimeSeriesSplit CV (K-Fold temporel)
    tscv = TimeSeriesSplit(n_splits=min(TS_CV_SPLITS, len(train_df) // 20))
    cv_logloss, cv_mae = [], []
    for train_idx, val_idx in tscv.split(X_train_s):
        X_tr, X_val = X_train_s.iloc[train_idx], X_train_s.iloc[val_idx]
        y_win_tr, y_win_val = y_win_train.iloc[train_idx], y_win_train.iloc[val_idx]
        y_diff_tr, y_diff_val = y_diff_train.iloc[train_idx], y_diff_train.iloc[val_idx]
        if len(X_tr) < 10 or len(X_val) < 5:
            continue
        clf_cv = _get_classifier()
        reg_cv = _get_regressor()
        clf_cv.fit(X_tr, y_win_tr)
        reg_cv.fit(X_tr, y_diff_tr)
        p = clf_cv.predict_proba(X_val)[:, 1]
        cv_logloss.append(log_loss(y_win_val, p))
        cv_mae.append(mean_absolute_error(y_diff_val, reg_cv.predict(X_val)))
    metrics = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_start_date": str(test_start.date()),
        "cv_log_loss_mean": float(np.mean(cv_logloss)) if cv_logloss else None,
        "cv_mae_mean": float(np.mean(cv_mae)) if cv_mae else None,
    }

    # Entraînement final sur tout le train
    clf = _get_classifier()
    reg = _get_regressor()
    clf.fit(X_train_s, y_win_train)
    reg.fit(X_train_s, y_diff_train)

    y_proba = clf.predict_proba(X_test_s)[:, 1]
    y_pred_win = clf.predict(X_test_s)
    y_pred_diff = reg.predict(X_test_s)

    metrics["accuracy"] = float(accuracy_score(y_win_test, y_pred_win))
    metrics["roc_auc"] = float(roc_auc_score(y_win_test, y_proba)) if len(np.unique(y_win_test)) > 1 else 0.5
    metrics["log_loss"] = float(log_loss(y_win_test, y_proba))
    metrics["mae_spread"] = float(mean_absolute_error(y_diff_test, y_pred_diff))

    # Importance des features
    imp_clf = getattr(clf, "feature_importances_", None)
    imp_reg = getattr(reg, "feature_importances_", None)
    if imp_clf is not None and imp_reg is not None:
        imp_avg = (np.array(imp_clf) + np.array(imp_reg)) / 2
        metrics["feature_importance"] = dict(zip(FEATURE_NAMES, [float(x) for x in imp_avg]))
    elif imp_clf is not None:
        metrics["feature_importance"] = dict(zip(FEATURE_NAMES, [float(x) for x in imp_clf]))
    elif imp_reg is not None:
        metrics["feature_importance"] = dict(zip(FEATURE_NAMES, [float(x) for x in imp_reg]))
    else:
        metrics["feature_importance"] = {}

    # Calibration : quand le modèle dit X%, gagne-t-on vraiment X% du temps ?
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_win_test, y_proba, n_bins=10)
        metrics["calibration"] = [
            {"predicted_bin": round(float(p), 2), "actual_win_rate": round(float(t), 2)}
            for t, p in zip(prob_true, prob_pred)
        ]
    except Exception:
        metrics["calibration"] = []

    # Modèle Upset : entraîné sur matchs "outsider" (cote > 2.50 ≈ prob < 0.40), pondération trappe
    upset_clf = None
    train_df["_underdog_won"] = (
        ((train_df["Home_Win_Rate_Season"] < train_df["Away_Win_Rate_Season"]) & (train_df["Win_Home"] == 1))
        | ((train_df["Away_Win_Rate_Season"] < train_df["Home_Win_Rate_Season"]) & (train_df["Win_Home"] == 0))
    ).astype(int)
    test_df["_underdog_won"] = (
        ((test_df["Home_Win_Rate_Season"] < test_df["Away_Win_Rate_Season"]) & (test_df["Win_Home"] == 1))
        | ((test_df["Away_Win_Rate_Season"] < test_df["Home_Win_Rate_Season"]) & (test_df["Win_Home"] == 0))
    ).astype(int)
    prob_home_train = clf.predict_proba(X_train_s)[:, 1]
    underdog_prob_train = np.where(
        train_df["Home_Win_Rate_Season"].values < train_df["Away_Win_Rate_Season"].values,
        prob_home_train,
        1.0 - prob_home_train,
    )
    mask_upset = underdog_prob_train < UPSET_IMPLIED_PROB_MAX
    if mask_upset.sum() >= 30:
        X_upset = X_train_s.loc[mask_upset]
        y_upset = train_df.loc[mask_upset, "_underdog_won"]
        sample_w = np.where(y_upset.values == 1, 2.0, 1.0)
        upset_clf = _get_classifier()
        try:
            upset_clf.fit(X_upset, y_upset, sample_weight=sample_w)
        except TypeError:
            upset_clf.fit(X_upset, y_upset)
        y_upset_proba_test = upset_clf.predict_proba(X_test_s)[:, 1]
        metrics["upset_n_train"] = int(mask_upset.sum())
        metrics["upset_accuracy"] = float((upset_clf.predict(X_test_s) == test_df["_underdog_won"]).mean())
        bin_20_40 = (y_upset_proba_test >= 0.20) & (y_upset_proba_test <= 0.40)
        if bin_20_40.sum() >= 5:
            actual_in_bin = test_df.loc[bin_20_40, "_underdog_won"].mean()
            metrics["calibration_upset_20_40"] = {
                "predicted_range": [0.20, 0.40],
                "actual_win_rate": round(float(actual_in_bin), 3),
                "n_samples": int(bin_20_40.sum()),
            }
        else:
            metrics["calibration_upset_20_40"] = None
    else:
        metrics["calibration_upset_20_40"] = None

    # Modèle Total Points (XGBRegressor optimisé pour score cumulé)
    totals_reg = None
    scaler_totals = None
    if "Total_Points" in train_df.columns and all(c in train_df.columns for c in FEATURE_NAMES_TOTALS):
        X_totals_train = train_df[FEATURE_NAMES_TOTALS].copy().fillna(0).replace([np.inf, -np.inf], 0)
        y_totals_train = train_df["Total_Points"]
        X_totals_test = test_df[FEATURE_NAMES_TOTALS].copy().fillna(0).replace([np.inf, -np.inf], 0)
        y_totals_test = test_df["Total_Points"]
        scaler_totals = StandardScaler()
        X_totals_train_s = scaler_totals.fit_transform(X_totals_train)
        X_totals_test_s = scaler_totals.transform(X_totals_test)
        try:
            import xgboost as xgb
            totals_reg = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )
        except Exception:
            totals_reg = _get_regressor()
        totals_reg.fit(X_totals_train_s, y_totals_train)
        y_totals_pred = totals_reg.predict(X_totals_test_s)
        mae_total = float(mean_absolute_error(y_totals_test, y_totals_pred))
        metrics["mae_total"] = mae_total
        metrics["mae_total_excellent"] = mae_total < 8.0
    else:
        metrics["mae_total"] = None
        metrics["mae_total_excellent"] = False

    return {
        "metrics": metrics,
        "classifier": clf,
        "regressor": reg,
        "upset_classifier": upset_clf,
        "totals_regressor": totals_reg,
        "scaler": scaler,
        "scaler_totals": scaler_totals,
        "feature_names": FEATURE_NAMES,
        "train_means": X_train.mean().to_dict(),
        "train_stds": X_train.std().replace(0, 1).to_dict(),
        "totals_train_means": (
            train_df[[c for c in FEATURE_NAMES_TOTALS if c in train_df.columns]].mean().to_dict()
            if totals_reg is not None else {}
        ),
    }


def save_models(result: Dict[str, Any]) -> Optional[str]:
    """Sauvegarde classifier, regressor, upset_clf, totals_reg, scalers et meta dans .pkl et .json."""
    import pickle
    clf = result.get("classifier")
    reg = result.get("regressor")
    scaler = result.get("scaler")
    upset_clf = result.get("upset_classifier")
    totals_reg = result.get("totals_regressor")
    scaler_totals = result.get("scaler_totals")
    if clf is None or reg is None:
        return "Modèles manquants dans result"
    try:
        # Écriture dans les fichiers _antitrap pour ne pas écraser l'ancien entraînement
        with open(MODEL_PROBA_ANTITRAP_PATH, "wb") as f:
            pickle.dump(clf, f)
        with open(MODEL_SPREAD_ANTITRAP_PATH, "wb") as f:
            pickle.dump(reg, f)
        if upset_clf is not None:
            with open(MODEL_UPSET_PATH, "wb") as f:
                pickle.dump(upset_clf, f)
        if totals_reg is not None:
            with open(MODEL_TOTALS_PATH, "wb") as f:
                pickle.dump(totals_reg, f)
        if scaler is not None:
            with open(SCALER_ANTITRAP_PATH, "wb") as f:
                pickle.dump(scaler, f)
        if scaler_totals is not None:
            with open(SCALER_TOTALS_PATH, "wb") as f:
                pickle.dump(scaler_totals, f)
        meta = {
            "feature_names": result.get("feature_names", FEATURE_NAMES),
            "train_means": result.get("train_means", {}),
            "train_stds": result.get("train_stds", {}),
        }
        metrics = result.get("metrics", {})
        if metrics.get("calibration"):
            meta["calibration"] = metrics["calibration"]
        if metrics.get("calibration_upset_20_40"):
            meta["calibration_upset_20_40"] = metrics["calibration_upset_20_40"]
        with open(FEATURES_META_ANTITRAP_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if totals_reg is not None:
            metrics = result.get("metrics", {})
            meta_totals = {
                "feature_names": FEATURE_NAMES_TOTALS,
                "train_means": result.get("totals_train_means", {}),
                "mae_total": metrics.get("mae_total"),
            }
            with open(FEATURES_META_TOTALS_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_totals, f, indent=2)
        return None
    except Exception as e:
        return str(e)


# ==============================================================================
# PRÉDICTION (inférence)
# ==============================================================================


def build_feature_row_for_match(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Construit une ligne de features pour un match futur (ou à prédire).
    Utilise les données strictement avant game_date (ou aujourd'hui si None).
    """
    from datetime import date
    date_str = (game_date or str(date.today()))[:10]
    if len(date_str) != 10:
        return None

    df_box = _fetch_all_box_scores()
    df_games = _fetch_all_games_for_context()
    if df_box.empty:
        return None
    if df_games.empty or "date" not in df_games.columns:
        return None
    if "date" not in df_box.columns:
        supabase = _get_supabase()
        if supabase:
            gh = supabase.table("games_history").select("game_id, date").execute().data or []
            df_gh = pd.DataFrame(gh)
            df_box = df_box.merge(df_gh, on="game_id", how="left")
    df_box["date"] = pd.to_datetime(df_box.get("date", pd.Series())).dt.strftime("%Y-%m-%d")
    df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    row = _build_row(date_str, home_id, away_id, league_id or 0, season or "", df_box, df_games)
    return row


def get_trap_info(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
    home_name: Optional[str] = None,
    away_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Contexte "Anti-Trap" : détecte si le favori est en Domestic Trap (EuroLeague + < 72h repos).
    Retourne is_domestic_trap, message explicatif, intensity_home/away.
    """
    from datetime import date
    date_str = (game_date or str(date.today()))[:10]
    if len(date_str) != 10:
        return {"is_domestic_trap": False, "context_message": "", "intensity_home": 1.0, "intensity_away": 1.0}
    df_games = _fetch_all_games_for_context()
    if df_games.empty or "date" not in df_games.columns:
        return {"is_domestic_trap": False, "context_message": "", "intensity_home": 1.0, "intensity_away": 1.0}
    df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    lid = int(league_id) if league_id is not None else 0
    is_trap_h = _is_domestic_trap_team(home_id, date_str, lid, df_games)
    is_trap_a = _is_domestic_trap_team(away_id, date_str, lid, df_games)
    intensity_home = _intensity_proxy(home_id, date_str, lid, df_games)
    intensity_away = _intensity_proxy(away_id, date_str, lid, df_games)
    is_domestic_trap = is_trap_h or is_trap_a
    parts: List[str] = []
    if is_trap_h and home_name:
        parts.append(f"{home_name} sort d'un match EuroLeague avec peu de repos, rotation probable.")
    if is_trap_a and away_name:
        parts.append(f"{away_name} sort d'un match EuroLeague avec peu de repos, rotation probable.")
    if not parts and _is_domestic_league(lid):
        if intensity_home < 0.7 or intensity_away < 0.7:
            parts.append("Fatigue cumulative EuroLeague possible sur une équipe.")
    context_message = " ".join(parts) if parts else ""
    if is_domestic_trap and context_message:
        context_message = "⚠️ Danger : " + context_message
    return {
        "is_domestic_trap": is_domestic_trap,
        "context_message": context_message.strip(),
        "intensity_home": intensity_home,
        "intensity_away": intensity_away,
    }


# Seuils pour le style de match (Total Points / confrontation de styles)
PACE_FACTOR_SHOOTOUT = 1.04   # Fast vs Fast
PACE_FACTOR_DEFENSIVE = 0.96  # Slow, jeu placé
DEF_RTG_SHOOTOUT = 105.0      # Def_Rtg élevé = défenses laissent marquer
DEF_RTG_DEFENSIVE = 100.0     # Def_Rtg bas = verrou défensif


def get_match_style(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> str:
    """
    Classifie le style de match à partir des features (Pace, Def_Rtg, Off_Rtg).
    - 🔥 Shootout : deux équipes Run & Gun, défenses permissives → total élevé.
    - 🛡️ Defensive Battle : rythme lent, défenses solides → total bas.
    - ⚖️ Balanced : entre les deux.
    """
    row = build_feature_row_for_match(home_id, away_id, game_date, league_id, season)
    if row is None:
        return "—"
    pace_factor = row.get("Pace_Factor_Match") or 1.0
    combined_def = row.get("Combined_Def_Rtg") or 100.0
    if pace_factor >= PACE_FACTOR_SHOOTOUT and combined_def >= DEF_RTG_SHOOTOUT:
        return "🔥 Shootout"
    if pace_factor <= PACE_FACTOR_DEFENSIVE and combined_def <= DEF_RTG_DEFENSIVE:
        return "🛡️ Defensive Battle"
    if pace_factor >= PACE_FACTOR_SHOOTOUT:
        return "🏃 Run & Gun"
    if combined_def <= DEF_RTG_DEFENSIVE:
        return "🔒 Verrou"
    return "⚖️ Balanced"


def _model_paths_for_load() -> Tuple[Path, Path, Path, Path]:
    """Chemins pour chargement : antitrap si présents, sinon ancien entraînement."""
    if MODEL_PROBA_ANTITRAP_PATH.exists() and MODEL_SPREAD_ANTITRAP_PATH.exists():
        return (
            MODEL_PROBA_ANTITRAP_PATH,
            MODEL_SPREAD_ANTITRAP_PATH,
            SCALER_ANTITRAP_PATH,
            FEATURES_META_ANTITRAP_PATH,
        )
    return MODEL_PROBA_PATH, MODEL_SPREAD_PATH, SCALER_PATH, FEATURES_META_PATH


def predict_upset_proba(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> Optional[float]:
    """
    Probabilité que l'outsider gagne (model_upset.pkl). None si modèle absent.
    Utile pour la colonne "PARI OUTSIDER" quand prob_outsider calibrée est élevée.
    """
    import pickle
    if not MODEL_UPSET_PATH.exists():
        return None
    row = build_feature_row_for_match(home_id, away_id, game_date, league_id, season)
    if row is None:
        return None
    _, _, scaler_path, meta_path = _model_paths_for_load()
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    feature_names = meta.get("feature_names", FEATURE_NAMES)
    train_means = meta.get("train_means", {})
    X = pd.DataFrame([row])
    for col in feature_names:
        if col not in X.columns:
            X[col] = train_means.get(col, 0.0)
    X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X = pd.DataFrame(scaler.transform(X), columns=feature_names)
        except Exception:
            pass
    try:
        with open(MODEL_UPSET_PATH, "rb") as f:
            upset_clf = pickle.load(f)
        return float(upset_clf.predict_proba(X)[0, 1])
    except Exception:
        return None


def predict_total_points(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> Optional[float]:
    """
    Prédit le total de points (Home_Score + Away_Score) via model_totals.pkl.
    Retourne None si modèle absent ou données insuffisantes.
    """
    import pickle
    if not MODEL_TOTALS_PATH.exists():
        return None
    row = build_feature_row_for_match(home_id, away_id, game_date, league_id, season)
    if row is None:
        return None
    meta_totals: Dict[str, Any] = {}
    if FEATURES_META_TOTALS_PATH.exists():
        try:
            with open(FEATURES_META_TOTALS_PATH, "r", encoding="utf-8") as f:
                meta_totals = json.load(f)
        except Exception:
            pass
    feature_names = meta_totals.get("feature_names", FEATURE_NAMES_TOTALS)
    train_means = meta_totals.get("train_means", {})
    X = pd.DataFrame([row])
    for col in feature_names:
        if col not in X.columns:
            X[col] = train_means.get(col, 0.0)
    X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    if SCALER_TOTALS_PATH.exists():
        try:
            with open(SCALER_TOTALS_PATH, "rb") as f:
                scaler_totals = pickle.load(f)
            X = scaler_totals.transform(X)
        except Exception:
            pass
    X_arr = np.asarray(X, dtype=np.float64).reshape(1, -1)
    try:
        with open(MODEL_TOTALS_PATH, "rb") as f:
            totals_reg = pickle.load(f)
        raw = float(totals_reg.predict(X_arr)[0])
        # Calibration par ligue (Reality Check) : Prediction_Finale = Brute + (avg_real - avg_pred)
        league_calibration = meta_totals.get("league_calibration", {})
        if league_id is not None and league_calibration:
            lid = int(league_id)
            delta = league_calibration.get(lid, league_calibration.get(str(lid), 0.0))
            if isinstance(delta, (int, float)):
                raw = raw + float(delta)
        return raw
    except Exception:
        return None


def predict_with_model(
    home_id: int,
    away_id: int,
    game_date: Optional[str] = None,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Charge les modèles .pkl et prédit pour (home_id, away_id).
    Retourne {"prob_home": float, "proj_home": float, "proj_away": float, "spread": float}
    ou None si modèles absents / erreur.
    """
    import pickle
    path_proba, path_spread, path_scaler, path_meta = _model_paths_for_load()
    if not path_proba.exists() or not path_spread.exists():
        return None
    try:
        with open(path_proba, "rb") as f:
            clf = pickle.load(f)
        with open(path_spread, "rb") as f:
            reg = pickle.load(f)
    except Exception:
        return None

    meta: Dict[str, Any] = {}
    if path_meta.exists():
        try:
            with open(path_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    feature_names = meta.get("feature_names", FEATURE_NAMES)
    train_means = meta.get("train_means", {})

    row = build_feature_row_for_match(home_id, away_id, game_date, league_id, season)
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

    # Normalisation si scaler sauvegardé (modèles v2 / antitrap)
    scaler = None
    if path_scaler.exists():
        try:
            with open(path_scaler, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            pass
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_names)

    prob_home = float(clf.predict_proba(X)[0, 1])
    spread = float(reg.predict(X)[0])
    # Total prédit par model_totals.pkl (plus de constante 150)
    predicted_total = predict_total_points(home_id, away_id, game_date, league_id, season)
    if predicted_total is None or not np.isfinite(predicted_total):
        predicted_total = 150.0
    proj_home = (float(predicted_total) + spread) / 2.0
    proj_away = (float(predicted_total) - spread) / 2.0

    # Calibration automatique : si le test set montre un taux réel différent, on l'utilise pour Edge/mise
    calibration = meta.get("calibration", [])
    prob_home_calibrated = prob_home
    if calibration and len(calibration) >= 2:
        preds = [c.get("predicted_bin", 0) for c in calibration]
        actuals = [c.get("actual_win_rate", 0) for c in calibration]
        sorted_pairs = sorted(zip(preds, actuals), key=lambda x: x[0])
        preds, actuals = [p for p, _ in sorted_pairs], [a for _, a in sorted_pairs]
        prob_home_calibrated = float(np.clip(np.interp(prob_home, preds, actuals), 0.0, 1.0))

    return {
        "prob_home": prob_home,
        "prob_home_calibrated": prob_home_calibrated,
        "proj_home": proj_home,
        "proj_away": proj_away,
        "spread": spread,
    }


def models_available() -> bool:
    """True si un jeu de modèles existe (antitrap ou ancien entraînement)."""
    path_proba, path_spread, _, _ = _model_paths_for_load()
    return path_proba.exists() and path_spread.exists()


# ==============================================================================
# CLI : entraînement depuis la ligne de commande
# ==============================================================================


def main_cli(mode: str = "full") -> None:
    """Lance build_training_dataset → train_and_evaluate → save_models et affiche les métriques."""
    print("Building training dataset (full base, pagination)...")
    df, err = build_training_dataset()
    if err:
        print("Error:", err)
        return
    print(f"Dataset: {len(df)} rows, {len(FEATURE_NAMES)} features")

    print("Training (split = last 2 months = Test, StandardScaler, TimeSeriesSplit CV)...")
    result = train_and_evaluate(df)
    if "error" in result:
        print("Error:", result["error"])
        return

    m = result["metrics"]
    print("--- Split ---")
    print(f"  Train: {m.get('n_train', 0)} rows")
    print(f"  Test:  {m.get('n_test', 0)} rows (from {m.get('test_start_date', '?')})")
    if m.get("cv_log_loss_mean") is not None:
        print(f"  CV Log Loss (mean): {m['cv_log_loss_mean']:.4f}")
    if m.get("cv_mae_mean") is not None:
        print(f"  CV MAE (mean):      {m['cv_mae_mean']:.4f} pts")
    print("--- Metrics (Test Set) ---")
    print(f"  Accuracy (Win): {m.get('accuracy', 0):.4f}")
    print(f"  ROC-AUC:        {m.get('roc_auc', 0):.4f}")
    print(f"  Log Loss:       {m.get('log_loss', 0):.4f}")
    print(f"  MAE Spread:     {m.get('mae_spread', 0):.4f} pts")
    print("--- Feature importance (top 15) ---")
    fi = m.get("feature_importance", {})
    for name, imp in sorted(fi.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {imp:.4f}")
    print("--- Calibration (quand le modèle dit X%, gagne-t-on vraiment X% du temps ?) ---")
    cal = m.get("calibration", [])
    if cal:
        for c in cal:
            p, t = c.get("predicted_bin", 0), c.get("actual_win_rate", 0)
            print(f"  Prédit ~{p*100:.0f}% → Taux réel: {t*100:.0f}%")
    else:
        print("  (non disponible)")
    if m.get("calibration_upset_20_40"):
        cu = m["calibration_upset_20_40"]
        print("--- Upset model (calibration [20%-40%]) ---")
        print(f"  Tranche 20-40% : taux réel outsider = {cu.get('actual_win_rate', 0)*100:.1f}% (n={cu.get('n_samples', 0)})")
    if m.get("upset_n_train") is not None:
        print(f"  Upset train samples: {m['upset_n_train']}, accuracy: {m.get('upset_accuracy', 0):.4f}")
    if m.get("mae_total") is not None:
        exc = " (excellent)" if m.get("mae_total_excellent") else ""
        print(f"  MAE Total Points: {m['mae_total']:.2f} pts{exc}")

    save_err = save_models(result)
    if save_err:
        print("Save error:", save_err)
    else:
        extras = []
        if result.get("upset_classifier") is not None:
            extras.append(str(MODEL_UPSET_PATH))
        if result.get("totals_regressor") is not None:
            extras.append(str(MODEL_TOTALS_PATH))
        ups = ", " + ", ".join(extras) if extras else ""
        print("Models saved (antitrap, ancien non écrasé):", MODEL_PROBA_ANTITRAP_PATH, MODEL_SPREAD_ANTITRAP_PATH, SCALER_ANTITRAP_PATH, ups)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Engine — ML Vainqueur + Spread")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "update"],
        help="full = entraînement complet ; update = même chose (pour pipeline quotidien)",
    )
    args = parser.parse_args()
    main_cli(mode=args.mode)
