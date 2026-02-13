#!/usr/bin/env python3
"""
99_simulate_past_predictions.py — Simulateur rétroactif
=======================================================
Génère des prédictions "comme si" elles avaient été faites
le matin des matchs passés, puis insère dans daily_projections_v2
avec date_prediction = date du match.

Usage:
  python 99_simulate_past_predictions.py --days 10
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytz

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

from database import get_client

# Option lenient: allow sparse team history for retro-simulations.
try:
    import training_engine as _te
    _te.MIN_GAMES_FOR_ROW = 0
except Exception:
    _te = None

# Réutilisation de la logique 03_predict_daily (modèles + prédiction + raisonnement)
from importlib.util import spec_from_file_location, module_from_spec
_script_dir = Path(__file__).resolve().parent
_spec = spec_from_file_location("predict_daily", _script_dir / "03_predict_daily.py")
_predict_daily = module_from_spec(_spec)
_spec.loader.exec_module(_predict_daily)

load_models = _predict_daily.load_models
get_ml_prediction = _predict_daily.get_ml_prediction
_get_team_name = _predict_daily._get_team_name
_fiabilite_from_box_scores = _predict_daily._fiabilite_from_box_scores
_build_reasoning_text = _predict_daily._build_reasoning_text
_mise_bucket = _predict_daily._mise_bucket
SNIPER_TARGET_EV_THRESHOLD = _predict_daily.SNIPER_TARGET_EV_THRESHOLD
EDGE_MAX_BET = _predict_daily.EDGE_MAX_BET
EDGE_VALUE = _predict_daily.EDGE_VALUE
DEFAULT_TOTAL_FALLBACK = _predict_daily.DEFAULT_TOTAL_FALLBACK

PARIS_TZ = pytz.timezone("Europe/Paris")
CHUNK_SIZE = 1000


def _today_paris() -> date:
    return datetime.now(PARIS_TZ).date()


def _fetch_finished_games_for_date(supabase, day_str: str) -> List[dict]:
    if not supabase:
        return []
    try:
        d0 = datetime.strptime(day_str[:10], "%Y-%m-%d")
        d1 = d0 + timedelta(days=1)
        day_start = d0.strftime("%Y-%m-%d")
        day_end = d1.strftime("%Y-%m-%d")
    except Exception:
        day_start = day_str[:10]
        day_end = day_str[:10]
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score, home_odd, away_odd")
            .gte("date", day_start)
            .lt("date", day_end)
            .not_.is_("home_score", "null")
            .not_.is_("away_score", "null")
            .order("date", desc=False)
            .execute()
        )
        data = r.data or []
        for row in data:
            row.setdefault("home_odd", None)
            row.setdefault("away_odd", None)
        return data
    except Exception:
        try:
            r = (
                supabase.table("games_history")
                .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score")
                .gte("date", day_start)
                .lt("date", day_end)
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .order("date", desc=False)
                .execute()
            )
            data = r.data or []
            for row in data:
                row.setdefault("home_odd", None)
                row.setdefault("away_odd", None)
            return data
        except Exception:
            return []


def _fetch_existing_predictions(supabase, from_date: str, to_date: str) -> Set[Tuple[int, str]]:
    if not supabase:
        return set()
    existing: Set[Tuple[int, str]] = set()
    offset = 0
    while True:
        r = (
            supabase.table("daily_projections_v2")
            .select("game_id, date_prediction")
            .gte("date_prediction", from_date[:10])
            .lte("date_prediction", to_date[:10])
            .order("date_prediction", desc=False)
            .range(offset, offset + CHUNK_SIZE - 1)
            .execute()
        )
        rows = r.data or []
        if not rows:
            break
        for row in rows:
            existing.add((row.get("game_id"), str(row.get("date_prediction") or "")[:10]))
        if len(rows) < CHUNK_SIZE:
            break
        offset += CHUNK_SIZE
    return existing


def simulate_days(days: int = 10) -> int:
    supabase = get_client()
    if not supabase:
        print("❌ Connexion Supabase impossible.")
        return 0

    models = load_models()
    if not models:
        print("❌ Modèles absents. Lance 02_train_models.py d'abord.")
        return 0

    today = _today_paris()
    start_date = today - timedelta(days=days)
    end_date = today - timedelta(days=1)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    existing = _fetch_existing_predictions(supabase, start_str, end_str)
    if existing:
        print(f"   → {len(existing)} projection(s) déjà présentes, elles seront ignorées.")

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
    for d in range(days, 0, -1):
        day_str = (today - timedelta(days=d)).isoformat()
        games = _fetch_finished_games_for_date(supabase, day_str)
        if not games:
            continue

        print(f"\n📅 {day_str} — {len(games)} match(s) terminé(s)")
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
            if (gid, game_date) in existing:
                continue

            home_name = _get_team_name(supabase, home_id)
            away_name = _get_team_name(supabase, away_id)
            match_name = f"{home_name} vs {away_name}"
            fiabilite = _fiabilite_from_box_scores(supabase, home_id, away_id)

            ml_pred = get_ml_prediction(models, home_id, away_id, game_date, league_id, season, supabase=supabase)
            if ml_pred is None and _te is not None:
                # Tentative lenient (historique incomplet)
                _te.MIN_GAMES_FOR_ROW = 0
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
            reasoning_text = "[SIMULATION] " + (reasoning_text or "").strip()

            if edge_ml < 0:
                le_pari = "PASSER"
            elif edge_home >= edge_away and edge_home > 0:
                le_pari = f"Victoire {home_name}" + (f" (@ {odd_home:.2f})" if odd_home else "")
            else:
                le_pari = f"Victoire {away_name}" + (f" (@ {odd_away:.2f})" if odd_away else "")

            payload = {
                "game_id": gid,
                "date_prediction": game_date,
                "match_name": match_name,
                "league": str(league_id) if league_id is not None else None,
                "start_time": g.get("date"),
                "proba_ml": float(prob_home),
                "proba_ml_calibrated": float(prob_calibrated),
                "projected_score_home": float(proj_home),
                "projected_score_away": float(proj_away),
                "total_points_projected": float(predicted_total),
                "bookmaker_odds_home": float(odd_home) if odd_home is not None else None,
                "bookmaker_odds_away": float(odd_away) if odd_away is not None else None,
                "bookmaker_line_total": None,
                "edge_ml": float(edge_ml),
                "edge_total": None,
                "confidence_score": float(fiabilite),
                "reasoning_text": reasoning_text,
                "le_pari": le_pari,
                "style_match": style_match or None,
            }

            try:
                supabase.table("daily_projections_v2").insert(payload).execute()
                upserted += 1
            except Exception as e:
                print(f"   ⚠️ game_id {gid}: {e}")

    return upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation de prédictions passées → daily_projections_v2")
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Nombre de jours passés à simuler (défaut: 10).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("🕰️ 99_simulate_past_predictions — Simulation rétroactive")
    print("=" * 55)
    print(f"   Fenêtre : J-{args.days} → J-1")

    n = simulate_days(days=args.days)
    print(f"\n✅ {n} projection(s) simulées enregistrées dans daily_projections_v2.\n")


if __name__ == "__main__":
    main()
