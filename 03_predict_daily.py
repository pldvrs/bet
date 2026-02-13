#!/usr/bin/env python3
"""
03_predict_daily.py — Le Cerveau (Write Once, Read Many)
=========================================================
DÉTERMINISTE : charge uniquement les modèles .pkl existants (jamais de ré-entraînement).
Génère les prédictions pour tous les matchs J+1 à J+3, insert/upsert dans daily_projections_v2.
Si une ligne existe déjà pour (game_id, date_prediction=aujourd'hui), on ne la modifie pas
sauf si les cotes ont changé significativement (seuil 5%).
Génère reasoning_text pour expliquer le pari.

À lancer UNE FOIS par jour (ex: 08h00 via run_pipeline ou cron). Le dashboard ne lance jamais ce script.

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
import pytz

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
PARIS_TZ = pytz.timezone("Europe/Paris")


def _today_paris() -> date:
    """Date du jour en heure de Paris (évite UTC/serveur)."""
    return datetime.now(PARIS_TZ).date()


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
    """Aujourd'hui / Demain / J+2 (référence = jour en Europe/Paris)."""
    try:
        d = datetime.strptime((game_date or "")[:10], "%Y-%m-%d").date()
        today = _today_paris()
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
    supabase=None,
) -> Optional[Dict[str, Any]]:
    """
    Prédiction ML : proba home, spread, total (model_totals), puis proj_home, proj_away.
    Si model_totals renvoie NaN ou plante : fallback (Avg_Points_Home_Last5 + Avg_Points_Away_Last5) / 2.
    Toujours retourne un predicted_total numérique (jamais vide).
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

    predicted_total = None
    totals_reg = models.get("totals_regressor")
    scaler_totals = models.get("scaler_totals")
    totals_feature_names = models.get("totals_feature_names") or []
    totals_train_means = models.get("totals_train_means") or {}
    if totals_reg is not None and scaler_totals is not None and totals_feature_names and row is not None:
        try:
            X_t = pd.DataFrame([row])
            for col in totals_feature_names:
                if col not in X_t.columns:
                    X_t[col] = totals_train_means.get(col, 0.0)
            X_t = X_t[totals_feature_names].fillna(0).replace([np.inf, -np.inf], 0)
            X_t_s = scaler_totals.transform(X_t)
            predicted_total = float(totals_reg.predict(X_t_s)[0])
        except Exception:
            predicted_total = None
    if predicted_total is None or not (isinstance(predicted_total, (int, float)) and np.isfinite(predicted_total)):
        try:
            predicted_total = predict_total_points(home_id, away_id, game_date, league_id, season)
        except Exception:
            pass
    if predicted_total is None or not (isinstance(predicted_total, (int, float)) and np.isfinite(predicted_total)):
        # Fallback : (Avg_Points_Home_Last5 + Avg_Points_Away_Last5) / 2
        if supabase:
            avg_h = _avg_points_team_last_n(supabase, home_id, game_date, 5)
            avg_a = _avg_points_team_last_n(supabase, away_id, game_date, 5)
            if avg_h is not None and avg_a is not None:
                predicted_total = (float(avg_h) + float(avg_a)) / 2.0
        if predicted_total is None or not np.isfinite(predicted_total):
            predicted_total = DEFAULT_TOTAL_FALLBACK
    predicted_total = float(predicted_total)
    proj_home = (predicted_total + spread) / 2.0
    proj_away = (predicted_total - spread) / 2.0

    calibration = models.get("calibration", [])
    prob_calibrated = _apply_calibration(prob_home, calibration) if calibration else prob_home

    return {
        "prob_home": prob_home,
        "prob_home_calibrated": prob_calibrated,
        "proj_home": proj_home,
        "proj_away": proj_away,
        "spread": spread,
        "predicted_total": predicted_total,
    }


def _avg_points_team_last_n(supabase, team_id: int, game_date: str, n: int = 5) -> Optional[float]:
    """Moyenne des points marqués par l'équipe sur ses n derniers matchs (date < game_date)."""
    if not supabase or not game_date or len(game_date) < 10:
        return None
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, home_id, away_id, home_score, away_score, date")
            .or_(f"home_id.eq.{team_id},away_id.eq.{team_id}")
            .lt("date", game_date[:10])
            .not_.is_("home_score", "null")
            .order("date", desc=True)
            .limit(n * 2)
            .execute()
        )
        rows = r.data or []
        pts = []
        for g in rows:
            if g.get("home_id") == team_id and g.get("home_score") is not None:
                pts.append(float(g["home_score"]))
            elif g.get("away_id") == team_id and g.get("away_score") is not None:
                pts.append(float(g["away_score"]))
            if len(pts) >= n:
                break
        if not pts:
            return None
        return sum(pts) / len(pts)
    except Exception:
        return None


def fetch_future_games(supabase, date_str: str, max_games: int = 200) -> List[dict]:
    """Matchs à venir pour une date donnée : game_date == date_str, home_score IS NULL."""
    if not supabase or not date_str or len(date_str) < 10:
        return []
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, date, league_id, season, home_id, away_id, home_odd, away_odd")
            .eq("date", date_str[:10])
            .is_("home_score", "null")
            .order("date", desc=False)
            .limit(max_games)
            .execute()
        )
        return r.data or []
    except Exception:
        try:
            r = (
                supabase.table("games_history")
                .select("game_id, date, league_id, season, home_id, away_id")
                .eq("date", date_str[:10])
                .is_("home_score", "null")
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


def fetch_future_games_j1_to_j3(supabase, max_games_per_day: int = 100) -> List[dict]:
    """Matchs à venir : aujourd'hui + J+1 à J+3 (référence = Paris). Aligné sur ce que 01 insère (today, today+1, today+2) + J+3."""
    today = _today_paris()
    seen = set()
    out = []
    for delta in range(0, 4):  # 0 = aujourd'hui, 1 = J+1, 2 = J+2, 3 = J+3
        d = (today + timedelta(days=delta)).isoformat()
        games = fetch_future_games(supabase, d, max_games=max_games_per_day)
        for g in games:
            gid = g.get("game_id")
            if gid and gid not in seen:
                seen.add(gid)
                out.append(g)
    return out


ODDS_CHANGE_THRESHOLD = 0.05  # 5% de changement relatif pour considérer "cotes significativement changées"


def _odds_changed_significantly(
    existing_home: Optional[float],
    existing_away: Optional[float],
    new_home: Optional[float],
    new_away: Optional[float],
) -> bool:
    """True si les cotes ont changé de plus de ODDS_CHANGE_THRESHOLD (relatif)."""
    if existing_home is None and existing_away is None:
        return True  # pas de ligne existante → on insère
    def _rel_change(old: float, new: float) -> float:
        if old is None or new is None or old <= 0:
            return 1.0
        return abs(new - old) / old
    h = _rel_change(existing_home, new_home) if (existing_home or new_home) else 0.0
    a = _rel_change(existing_away, new_away) if (existing_away or new_away) else 0.0
    return h > ODDS_CHANGE_THRESHOLD or a > ODDS_CHANGE_THRESHOLD


def _get_existing_projection(
    supabase, game_id: int, date_prediction: str
) -> Optional[dict]:
    """Récupère la projection existante pour (game_id, date_prediction) si elle existe."""
    if not supabase or not date_prediction or len(date_prediction) < 10:
        return None
    try:
        r = (
            supabase.table("daily_projections_v2")
            .select("game_id, date_prediction, bookmaker_odds_home, bookmaker_odds_away")
            .eq("game_id", game_id)
            .eq("date_prediction", date_prediction[:10])
            .limit(1)
            .execute()
        )
        data = r.data or []
        return data[0] if data else None
    except Exception:
        return None


def _build_reasoning_text(
    home_name: str,
    away_name: str,
    edge_home: float,
    edge_away: float,
    confiance: str,
    context_message: str,
    style_match: str,
    brain_used: str,
    alerte_trappe: str,
    pari_outsider: str,
) -> str:
    """Génère une phrase explicative du pari pour la colonne reasoning_text."""
    parts = []
    if alerte_trappe and "Trap" in alerte_trappe and context_message:
        parts.append(context_message.strip()[:200])
    if brain_used and "Chasseur" in brain_used and pari_outsider and pari_outsider != "—":
        parts.append("Value détectée sur l'outsider (modèle Chasseur de Surprises).")
    elif edge_home > 0 and edge_home >= edge_away:
        parts.append(f"Avantage ML domicile : {home_name} avec edge +{edge_home:.1f}%.")
    elif edge_away > 0:
        parts.append(f"Avantage ML extérieur : {away_name} avec edge +{edge_away:.1f}%.")
    if style_match and style_match != "—":
        parts.append(f"Style : {style_match}.")
    if context_message and "Trap" not in (context_message or ""):
        msg = (context_message or "").strip()[:150]
        if msg:
            parts.append(msg)
    if not parts:
        parts.append("Projection ML standard.")
    return " ".join(parts).strip() or "Projection ML."


def run_predictions(max_games: int = 200) -> int:
    """
    Récupère les matchs J+1 à J+3, calcule les prédictions ML (modèles .pkl uniquement),
    écrit dans daily_projections_v2. Si une ligne existe déjà pour (game_id, date_prediction=aujourd'hui),
    on ne la modifie pas sauf si les cotes ont changé significativement.
    Retourne le nombre de lignes insérées ou mises à jour.
    """
    supabase = get_client()
    if not supabase:
        print("❌ Connexion Supabase impossible.")
        return 0

    models = load_models()
    if not models:
        print("❌ Modèles absents. Lance 02_train_models.py d'abord.")
        return 0

    date_prediction_str = _today_paris().isoformat()
    games = fetch_future_games_j1_to_j3(supabase, max_games_per_day=max_games)
    if not games:
        print(f"   Aucun match à venir (aujourd'hui + J+1 à J+3) dans games_history.")
        print(f"   → Vérifier que 01_ingest_data a bien tourné et que l'API renvoie des matchs pour ces dates.")
        return 0

    print(f"\n📊 {len(games)} match(s) (aujourd'hui + J+1 à J+3) → prédictions ML (figées dans daily_projections_v2)...")

    try:
        from training_engine import get_trap_info, get_match_style
    except Exception:
        get_trap_info = None
        get_match_style = None

    try:
        from training_engine import predict_upset_proba
    except Exception:
        predict_upset_proba = None

    upserted = 0
    skipped_existing = 0

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

        # Ne pas projeter un match déjà passé (date du match < date du run)
        if game_date < date_prediction_str:
            continue

        # Ne pas écraser une projection du jour sauf si cotes significativement changées
        existing = _get_existing_projection(supabase, gid, date_prediction_str)
        if existing and not _odds_changed_significantly(
            existing.get("bookmaker_odds_home"),
            existing.get("bookmaker_odds_away"),
            odd_home,
            odd_away,
        ):
            skipped_existing += 1
            continue

        home_name = _get_team_name(supabase, home_id)
        away_name = _get_team_name(supabase, away_id)
        match_name = f"{home_name} vs {away_name}"
        fiabilite = _fiabilite_from_box_scores(supabase, home_id, away_id)

        ml_pred = get_ml_prediction(models, home_id, away_id, game_date, league_id, season, supabase=supabase)
        if ml_pred is None:
            continue

        prob_home = ml_pred["prob_home"]
        prob_calibrated = ml_pred.get("prob_home_calibrated", prob_home)
        proj_home = ml_pred["proj_home"]
        proj_away = ml_pred["proj_away"]
        predicted_total = ml_pred.get("predicted_total")
        if predicted_total is None or (isinstance(predicted_total, float) and not np.isfinite(predicted_total)):
            predicted_total = DEFAULT_TOTAL_FALLBACK
        predicted_total = float(predicted_total)

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
        edge_ml = max(edge_home, edge_away)
        if (prob_for_ev * (odd_home or 0) - 1.0) > SNIPER_TARGET_EV_THRESHOLD:
            confiance_label = "🎯 SNIPER TARGET"
        else:
            confiance_label = _mise_bucket(edge_ml, fiabilite)

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

        if edge_ml > 0 and brain_used == "🔥 Chasseur de Surprises":
            pari_outsider = f"{away_name}" if edge_away >= edge_home else f"{home_name}"

        reasoning_text = _build_reasoning_text(
            home_name, away_name, edge_home, edge_away, confiance_label,
            context_message, style_match, brain_used, alerte_trappe, pari_outsider,
        )

        # Prono ML (aligné Deep Dive) : Victoire [Équipe] (@ cote) ou PASSER
        if edge_ml < 0:
            le_pari = "PASSER"
        elif edge_home >= edge_away and edge_home > 0:
            le_pari = f"Victoire {home_name}" + (f" (@ {odd_home:.2f})" if odd_home else "")
        else:
            le_pari = f"Victoire {away_name}" + (f" (@ {odd_away:.2f})" if odd_away else "")

        # Ligne bookmaker total (non fournie par games_history ici) → null, edge_total null
        bookmaker_line_total = None
        edge_total = None

        payload = {
            "game_id": gid,
            "date_prediction": date_prediction_str,
            "match_name": match_name,
            "league": str(league_id) if league_id is not None else None,
            "start_time": g.get("date"),  # ISO string si dispo
            "proba_ml": float(prob_home),
            "proba_ml_calibrated": float(prob_calibrated),
            "projected_score_home": float(proj_home),
            "projected_score_away": float(proj_away),
            "total_points_projected": float(predicted_total),
            "bookmaker_odds_home": float(odd_home) if odd_home is not None else None,
            "bookmaker_odds_away": float(odd_away) if odd_away is not None else None,
            "bookmaker_line_total": bookmaker_line_total,
            "edge_ml": float(edge_ml),
            "edge_total": edge_total,
            "confidence_score": float(fiabilite),
            "reasoning_text": reasoning_text,
            "le_pari": le_pari,
            "style_match": style_match or None,
        }

        try:
            supabase.table("daily_projections_v2").upsert(
                payload,
                on_conflict="game_id,date_prediction",
            ).execute()
            upserted += 1
        except Exception as e:
            print(f"   ⚠️ game_id {gid}: {e}")

    if skipped_existing:
        print(f"   ({skipped_existing} projection(s) déjà présentes, cotes inchangées — non modifiées)")
    return upserted


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prédictions ML → daily_projections_v2 (Write Once)")
    parser.add_argument("--max-games", type=int, default=200, help="Nombre max de matchs par jour (J+1 à J+3)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("🧠 03_predict_daily — Le Cerveau (Write Once, Read Many)")
    print("=" * 50)

    n = run_predictions(max_games=args.max_games)
    print(f"\n✅ {n} projection(s) enregistrées dans daily_projections_v2.\n")


if __name__ == "__main__":
    main()
