#!/usr/bin/env python3
"""
03_predict_daily.py — Le Cerveau (Architecture ETL Sniper V1 Pro)
===================================================================
Charge les modèles (proba, spread, totals), récupère les matchs à venir depuis
Supabase (games_history, sans appels API), calcule les prédictions et enregistre
les résultats dans daily_projections.

Totals : model_totals.predict() → total réel (ex: 168 pts), puis
  proj_home = (total + spread) / 2, proj_away = (total - spread) / 2.

Usage:
  python 03_predict_daily.py
  python 03_predict_daily.py --max-games 100
"""

import json
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

from database import get_client

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PROBA_PATH = SCRIPT_DIR / "model_proba.pkl"
MODEL_SPREAD_PATH = SCRIPT_DIR / "model_spread.pkl"
MODEL_TOTALS_PATH = SCRIPT_DIR / "model_totals.pkl"
SCALER_PATH = SCRIPT_DIR / "scaler.pkl"
SCALER_TOTALS_PATH = SCRIPT_DIR / "scaler_totals.pkl"
FEATURES_META_PATH = SCRIPT_DIR / "features_meta.json"
FEATURES_META_TOTALS_PATH = SCRIPT_DIR / "features_meta_totals.json"

# Seuils Edge (alignés app_sniper_v27)
SNIPER_TARGET_EV_THRESHOLD = 0.05
EDGE_MAX_BET = 10.0
EDGE_VALUE = 5.0
DEFAULT_TOTAL_FALLBACK = 165.0


def _get_team_name(supabase, team_id: int) -> str:
    """Nom d'équipe depuis teams_metadata."""
    if not supabase:
        return f"Équipe {team_id}"
    try:
        r = supabase.table("teams_metadata").select("nom_equipe, name").eq("team_id", team_id).limit(1).execute()
        data = r.data or []
        if data:
            nom = (data[0].get("nom_equipe") or data[0].get("name") or "").strip()
            if nom:
                return nom
    except Exception:
        pass
    return f"Équipe {team_id}"


def _day_label(game_date: str) -> str:
    """Aujourd'hui / Demain / J+2."""
    try:
        d = datetime.strptime((game_date or "")[:10], "%Y-%m-%d").date()
        today = date.today()
        delta = (d - today).days
        if delta == 0:
            return "Aujourd'hui"
        if delta == 1:
            return "Demain"
        if delta >= 2:
            return f"J+{delta}"
    except Exception:
        pass
    return "—"


def _fiabilite_from_box_scores(supabase, home_id: int, away_id: int, n: int = 15) -> float:
    """Fiabilité 0–100 : min des deux équipes (assez de matchs récents)."""
    if not supabase:
        return 70.0
    try:
        def _n_games(tid: int) -> int:
            r = supabase.table("box_scores").select("game_id").eq("team_id", tid).limit(n).execute()
            return len(r.data or [])

        n_h, n_a = _n_games(home_id), _n_games(away_id)
        min_games = min(n_h, n_a)
        if min_games >= 10:
            return 85.0
        if min_games >= 5:
            return 70.0
        if min_games >= 3:
            return 50.0
        return 30.0
    except Exception:
        return 70.0


def _mise_bucket(edge: float, fiabilite: float) -> str:
    """Confiance : MAX BET / VALUE / PASS."""
    if fiabilite < 30:
        return "🛑 PASS"
    if edge >= EDGE_MAX_BET:
        return "🔥 MAX BET"
    if edge >= EDGE_VALUE:
        return "✅ VALUE"
    if edge < 0:
        return "🛑 PASS"
    return "🛑 PASS"


def _apply_calibration(raw_prob: float, calibration: List[Dict[str, float]]) -> float:
    """Interpole la probabilité brute avec la courbe de calibration."""
    if not calibration or len(calibration) < 2:
        return raw_prob
    preds = [c.get("predicted_bin", 0) for c in calibration]
    actuals = [c.get("actual_win_rate", 0) for c in calibration]
    sorted_pairs = sorted(zip(preds, actuals), key=lambda x: x[0])
    preds, actuals = [p for p, _ in sorted_pairs], [a for _, a in sorted_pairs]
    return float(np.clip(np.interp(raw_prob, preds, actuals), 0.0, 1.0))


# -----------------------------------------------------------------------------
# Chargement des modèles
# -----------------------------------------------------------------------------


def load_models() -> Optional[Dict[str, Any]]:
    """Charge classifier, regressor, totals_reg, scalers et meta. Retourne None si absent."""
    if not MODEL_PROBA_PATH.exists() or not MODEL_SPREAD_PATH.exists():
        return None
    try:
        with open(MODEL_PROBA_PATH, "rb") as f:
            clf = pickle.load(f)
        with open(MODEL_SPREAD_PATH, "rb") as f:
            reg = pickle.load(f)
        scaler = None
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        meta = {}
        if FEATURES_META_PATH.exists():
            with open(FEATURES_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

        totals_reg = None
        scaler_totals = None
        meta_totals = {}
        if MODEL_TOTALS_PATH.exists():
            with open(MODEL_TOTALS_PATH, "rb") as f:
                totals_reg = pickle.load(f)
            if SCALER_TOTALS_PATH.exists():
                with open(SCALER_TOTALS_PATH, "rb") as f:
                    scaler_totals = pickle.load(f)
            if FEATURES_META_TOTALS_PATH.exists():
                with open(FEATURES_META_TOTALS_PATH, "r", encoding="utf-8") as f:
                    meta_totals = json.load(f)

        return {
            "classifier": clf,
            "regressor": reg,
            "totals_regressor": totals_reg,
            "scaler": scaler,
            "scaler_totals": scaler_totals,
            "feature_names": meta.get("feature_names", []),
            "train_means": meta.get("train_means", {}),
            "calibration": meta.get("calibration", []),
            "totals_feature_names": meta_totals.get("feature_names", []),
            "totals_train_means": meta_totals.get("train_means", {}),
        }
    except Exception:
        return None


def get_ml_prediction(
    models: Dict[str, Any],
    home_id: int,
    away_id: int,
    game_date: str,
    league_id: Optional[int],
    season: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Prédiction ML : proba home, spread, total (model_totals), puis proj_home, proj_away.
    Utilise training_engine pour la ligne de features et predict_total_points.
    """
    from training_engine import build_feature_row_for_match, predict_total_points

    feature_names = models.get("feature_names") or []
    if not feature_names:
        return None
    row = build_feature_row_for_match(home_id, away_id, game_date, league_id, season)
    if row is None:
        return None

    X = pd.DataFrame([row])
    train_means = models.get("train_means", {})
    for col in feature_names:
        if col not in X.columns:
            X[col] = train_means.get(col, 0.0)
    X = X[feature_names]
    for col in feature_names:
        if col in X.columns and (X[col].isna().any() or np.isinf(X[col]).any()):
            X[col] = X[col].fillna(train_means.get(col, 0.0)).replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0)

    scaler = models.get("scaler")
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_names)

    clf = models.get("classifier")
    reg = models.get("regressor")
    if clf is None or reg is None:
        return None

    prob_home = float(clf.predict_proba(X)[0, 1])
    spread = float(reg.predict(X)[0])

    predicted_total = predict_total_points(home_id, away_id, game_date, league_id, season)
    if predicted_total is None or not (isinstance(predicted_total, (int, float)) and np.isfinite(predicted_total)):
        predicted_total = DEFAULT_TOTAL_FALLBACK
    proj_home = (float(predicted_total) + spread) / 2.0
    proj_away = (float(predicted_total) - spread) / 2.0

    calibration = models.get("calibration", [])
    prob_calibrated = _apply_calibration(prob_home, calibration) if calibration else prob_home

    return {
        "prob_home": prob_home,
        "prob_home_calibrated": prob_calibrated,
        "proj_home": proj_home,
        "proj_away": proj_away,
        "spread": spread,
        "predicted_total": float(predicted_total),
    }


def fetch_future_games(supabase, max_games: int = 200) -> List[dict]:
    """Matchs à venir : games_history où home_score IS NULL, date >= aujourd'hui."""
    if not supabase:
        return []
    today = date.today().isoformat()
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, date, league_id, season, home_id, away_id, home_odd, away_odd")
            .is_("home_score", "null")
            .gte("date", today)
            .order("date", desc=False)
            .limit(max_games)
            .execute()
        )
        return r.data or []
    except Exception:
        # Colonnes home_odd/away_odd absentes si schema_migration_odds non exécuté
        try:
            r = (
                supabase.table("games_history")
                .select("game_id, date, league_id, season, home_id, away_id")
                .is_("home_score", "null")
                .gte("date", today)
                .order("date", desc=False)
                .limit(max_games)
                .execute()
            )
            data = r.data or []
            for row in data:
                row.setdefault("home_odd", None)
                row.setdefault("away_odd", None)
            return data
        except Exception:
            return []


def run_predictions(max_games: int = 200) -> int:
    """
    Récupère les matchs à venir, calcule les prédictions ML, écrit dans daily_projections.
    Retourne le nombre de lignes upsertées.
    """
    supabase = get_client()
    if not supabase:
        print("❌ Connexion Supabase impossible.")
        return 0

    models = load_models()
    if not models:
        print("❌ Modèles absents. Lance 02_train_models.py d'abord.")
        return 0

    games = fetch_future_games(supabase, max_games=max_games)
    if not games:
        print("   Aucun match à venir (games_history avec home_score NULL et date >= aujourd'hui).")
        return 0

    print(f"\n📊 {len(games)} match(s) à venir → prédictions ML...")

    try:
        from training_engine import get_trap_info, get_match_style
    except Exception:
        get_trap_info = None
        get_match_style = None

    try:
        from training_engine import predict_upset_proba
    except Exception:
        predict_upset_proba = None

    now = datetime.utcnow().isoformat()
    upserted = 0

    for g in games:
        gid = g.get("game_id")
        game_date = str(g.get("date") or "")[:10]
        league_id = g.get("league_id")
        season = g.get("season") or ""
        home_id = g.get("home_id")
        away_id = g.get("away_id")
        odd_home = g.get("home_odd")
        odd_away = g.get("away_odd")

        if not gid or not home_id or not away_id or len(game_date) != 10:
            continue

        home_name = _get_team_name(supabase, home_id)
        away_name = _get_team_name(supabase, away_id)
        match_name = f"{home_name} vs {away_name}"
        jour = _day_label(game_date)
        fiabilite = _fiabilite_from_box_scores(supabase, home_id, away_id)

        ml_pred = get_ml_prediction(models, home_id, away_id, game_date, league_id, season)
        if ml_pred is None:
            continue

        prob_home = ml_pred["prob_home"]
        prob_calibrated = ml_pred.get("prob_home_calibrated", prob_home)
        proj_home = ml_pred["proj_home"]
        proj_away = ml_pred["proj_away"]
        predicted_total = ml_pred.get("predicted_total")

        # Brain : Standard ou Chasseur (upset)
        prob_for_ev = prob_calibrated
        brain_used = "🧠 Standard"
        if predict_upset_proba and (odd_home or odd_away):
            try:
                upset_prob = predict_upset_proba(home_id, away_id, game_date, league_id, season)
            except Exception:
                upset_prob = None
            odd_outsider = (odd_away if (odd_away or 0) > (odd_home or 0) else odd_home) or 0
            if odd_outsider > 2.50 and upset_prob is not None and upset_prob > 0.40:
                if (odd_away or 0) > (odd_home or 0):
                    prob_for_ev = 1.0 - upset_prob
                else:
                    prob_for_ev = upset_prob
                brain_used = "🔥 Chasseur de Surprises"

        edge_home = (prob_for_ev * (odd_home or 0) - 1.0) * 100.0 if odd_home else 0.0
        edge_away = ((1.0 - prob_for_ev) * (odd_away or 0) - 1.0) * 100.0 if odd_away else 0.0
        edge = max(edge_home, edge_away)
        if (prob_for_ev * (odd_home or 0) - 1.0) > SNIPER_TARGET_EV_THRESHOLD:
            confiance = "🎯 SNIPER TARGET"
        else:
            confiance = _mise_bucket(edge, fiabilite)

        if edge_home > edge_away and edge_home > 0:
            le_pari = f"{home_name} (@ {odd_home:.1f})" if odd_home else home_name
        elif edge_away > 0:
            le_pari = f"{away_name} (@ {odd_away:.1f})" if odd_away else away_name
        else:
            le_pari = "—"

        pari_outsider = "—"
        alerte_trappe = "—"
        context_message = ""
        if get_trap_info:
            try:
                trap_info = get_trap_info(home_id, away_id, game_date, league_id, season, home_name, away_name)
                if trap_info.get("is_domestic_trap"):
                    alerte_trappe = "⚠️ Trap"
                context_message = (trap_info.get("context_message") or "")[:500]
            except Exception:
                pass
        style_match = "—"
        if get_match_style:
            try:
                style_match = get_match_style(home_id, away_id, game_date, league_id, season)
            except Exception:
                pass

        payload = {
            "game_id": gid,
            "match_name": match_name,
            "date": game_date,
            "time": None,
            "jour": jour,
            "league_id": league_id,
            "season": season,
            "home_id": home_id,
            "away_id": away_id,
            "proba_ml": prob_home,
            "proba_calibree": prob_calibrated,
            "edge_percent": float(edge),
            "brain_used": brain_used,
            "confiance_label": confiance,
            "le_pari": le_pari,
            "pari_outsider": pari_outsider,
            "alerte_trappe": alerte_trappe,
            "message_contexte": context_message,
            "fiabilite": fiabilite,
            "predicted_total": predicted_total,
            "line_bookmaker": None,
            "diff_total": None,
            "pari_total": "En attente",
            "confiance_ou": "—",
            "style_match": style_match,
            "odds_home": odd_home,
            "odds_away": odd_away,
            "updated_at": now,
        }

        try:
            supabase.table("daily_projections").upsert(payload, on_conflict="game_id").execute()
            upserted += 1
        except Exception as e:
            print(f"   ⚠️ game_id {gid}: {e}")

    return upserted


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prédictions ML → daily_projections (Sniper ETL)")
    parser.add_argument("--max-games", type=int, default=200, help="Nombre max de matchs à traiter")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("🧠 03_predict_daily — Le Cerveau")
    print("=" * 50)

    n = run_predictions(max_games=args.max_games)
    print(f"\n✅ {n} projection(s) enregistrées dans daily_projections.\n")


if __name__ == "__main__":
    main()
