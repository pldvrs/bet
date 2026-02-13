#!/usr/bin/env python3
"""
05_true_backtest.py — Time Machine (Backtest sans lookahead)
=============================================================
Pour chaque jour J parmi les N derniers (défaut 30) :
  - Masque toutes les données après J (train strictement sur date < J).
  - Entraîne les modèles sur les données < J.
  - Génère les prédictions pour les matchs du jour J.
  - Compare avec les résultats réels (games_history).
  - Enregistre : Date | Match | Pari | Cote | Résultat | Profit.

Résultats écrits dans backtest_results.json pour affichage dans le dashboard.

Usage:
  python 05_true_backtest.py
  python 05_true_backtest.py --days 30
  python 05_true_backtest.py --days 60
"""

import importlib.util
import json
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
BACKTEST_RESULTS_PATH = SCRIPT_DIR / "backtest_results.json"
PARIS_TZ = pytz.timezone("Europe/Paris")
EDGE_VALUE = 5.0  # seuil Edge pour considérer un pari (aligné dashboard)


def _today_paris() -> date:
    return datetime.now(PARIS_TZ).date()


def _get_team_name(supabase, team_id: int) -> str:
    if not supabase:
        return f"Équipe {team_id}"
    try:
        r = supabase.table("teams_metadata").select("nom_equipe, name").eq("team_id", team_id).limit(1).execute()
        if r.data:
            nom = (r.data[0].get("nom_equipe") or r.data[0].get("name") or "").strip()
            if nom:
                return nom
    except Exception:
        pass
    return f"Équipe {team_id}"


def _load_training_module():
    """Charge 02_train_models pour _train_proba_spread et _train_totals (sans exécuter main)."""
    spec = importlib.util.spec_from_file_location("_train_mod", SCRIPT_DIR / "02_train_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_predict_module():
    """Charge 03_predict_daily pour get_ml_prediction et build_feature_row (get_ml_prediction utilise les models dict)."""
    spec = importlib.util.spec_from_file_location("_predict_mod", SCRIPT_DIR / "03_predict_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_true_backtest(days: int = 30) -> Tuple[List[Dict[str, Any]], float]:
    """
    Backtest walk-forward : pour chaque jour J dans les `days` derniers jours,
    entraîne sur données < J, prédit pour J, compare aux résultats réels.
    Retourne (liste de lignes {date, match, pari, cote, resultat, profit}, profit_total).
    """
    supabase = get_client()
    if not supabase:
        return [], 0.0

    try:
        train_mod = _load_training_module()
        predict_mod = _load_predict_module()
    except Exception as e:
        print(f"❌ Chargement modules: {e}")
        return [], 0.0

    from training_engine import build_training_dataset

    today = _today_paris()
    # Inclure aujourd'hui + les N derniers jours pour afficher tous les matchs récents (y compris sans cotes)
    day_list = [(today - timedelta(days=i)).isoformat() for i in range(0, days + 1)]
    results: List[Dict[str, Any]] = []
    total_profit = 0.0

    for day_str in day_list:
        df, err = build_training_dataset(max_date=day_str)
        if err or df is None or df.empty or len(df) < 50:
            continue

        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        train_df = df
        test_df_dummy = df.tail(min(20, len(df)))

        try:
            proba_spread_result = train_mod._train_proba_spread(train_df, test_df_dummy)
            totals_result = train_mod._train_totals(train_df, test_df_dummy, df)
        except Exception as e:
            continue

        totals_reg = totals_result.get("totals_regressor")
        scaler_totals = totals_result.get("scaler_totals")
        models = {
            "classifier": proba_spread_result.get("classifier"),
            "regressor": proba_spread_result.get("regressor"),
            "scaler": proba_spread_result.get("scaler"),
            "feature_names": proba_spread_result.get("feature_names", []),
            "train_means": proba_spread_result.get("train_means", {}),
            "calibration": (proba_spread_result.get("metrics") or {}).get("calibration", []),
            "totals_regressor": totals_reg,
            "scaler_totals": scaler_totals,
            "totals_feature_names": totals_result.get("feature_names_totals", []),
            "totals_train_means": totals_result.get("totals_train_means", {}),
        }

        try:
            r = (
                supabase.table("games_history")
                .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score, home_odd, away_odd")
                .eq("date", day_str)
                .not_.is_("home_score", "null")
                .execute()
            )
        except Exception:
            try:
                r = (
                    supabase.table("games_history")
                    .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score")
                    .eq("date", day_str)
                    .not_.is_("home_score", "null")
                    .execute()
                )
                for row in (r.data or []):
                    row.setdefault("home_odd", None)
                    row.setdefault("away_odd", None)
            except Exception:
                r = type("R", (), {"data": []})()

        games_j = r.data or []
        for g in games_j:
            oh, oa = g.get("home_odd"), g.get("away_odd")
            oh = float(oh) if oh is not None else None
            oa = float(oa) if oa is not None else None
            game_date = str(g.get("date", ""))[:10]
            home_won = (g.get("home_score") or 0) > (g.get("away_score") or 0)
            resultat = "Vic Domicile" if home_won else "Vic Extérieur"
            home_name = _get_team_name(supabase, g["home_id"])
            away_name = _get_team_name(supabase, g["away_id"])
            match_str = f"{home_name} vs {away_name}"
            # Cotes pour information (affichage même si pas de pari)
            cote_home_str = f"{oh:.2f}" if oh is not None else "—"
            cote_away_str = f"{oa:.2f}" if oa is not None else "—"

            pred = predict_mod.get_ml_prediction(
                models, g["home_id"], g["away_id"], game_date, g.get("league_id"), g.get("season") or "", supabase=None
            )
            if pred is None:
                # Match sans prédiction ML : on l'affiche quand même avec cotes et résultat
                results.append({
                    "date": game_date,
                    "match": match_str,
                    "cote_home": cote_home_str,
                    "cote_away": cote_away_str,
                    "pari": "—",
                    "cote": "—",
                    "resultat": resultat,
                    "profit": None,
                })
                continue

            prob = pred.get("prob_home_calibrated", pred.get("prob_home", 0.5))
            edge_h = (prob * (oh or 0) - 1.0) * 100.0 if oh else -999
            edge_a = ((1.0 - prob) * (oa or 0) - 1.0) * 100.0 if oa else -999
            has_odds = oh is not None or oa is not None
            if has_odds and (edge_h >= EDGE_VALUE or edge_a >= EDGE_VALUE):
                stake = 1.0
                if edge_h >= edge_a and edge_h >= EDGE_VALUE and oh is not None:
                    profit = (oh * stake - stake) if home_won else -stake
                    pari = "Vic Domicile"
                    cote = f"{oh:.2f}"
                elif oa is not None:
                    profit = (oa * stake - stake) if not home_won else -stake
                    pari = "Vic Extérieur"
                    cote = f"{oa:.2f}"
                else:
                    profit = 0.0
                    pari = "—"
                    cote = "—"
                total_profit += profit
                results.append({
                    "date": game_date,
                    "match": match_str,
                    "cote_home": cote_home_str,
                    "cote_away": cote_away_str,
                    "pari": pari,
                    "cote": cote,
                    "resultat": resultat,
                    "profit": round(profit, 2),
                })
            else:
                # Match avec prédiction mais sans pari (edge insuffisant ou pas de cotes) : affiché pour info
                results.append({
                    "date": game_date,
                    "match": match_str,
                    "cote_home": cote_home_str,
                    "cote_away": cote_away_str,
                    "pari": "—",
                    "cote": "—",
                    "resultat": resultat,
                    "profit": None,
                })

    return results, total_profit


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Time Machine (10 jours, sans lookahead)")
    parser.add_argument("--days", type=int, default=30, help="Nombre de jours à simuler (défaut 30)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("⏱️ 05_true_backtest — Time Machine")
    print("=" * 50)

    results, total = run_true_backtest(days=args.days)
    n_bets = sum(1 for r in results if r.get("profit") is not None)
    payload = {
        "updated_at": datetime.now(PARIS_TZ).isoformat(),
        "days": args.days,
        "total_profit": round(total, 2),
        "n_bets": n_bets,
        "n_matches": len(results),
        "rows": results,
    }
    with open(BACKTEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(results)} pari(s) simulés → profit total: {total:.2f} u")
    print(f"   Résultats enregistrés dans {BACKTEST_RESULTS_PATH.name}\n")


if __name__ == "__main__":
    main()
