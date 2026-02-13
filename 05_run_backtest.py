#!/usr/bin/env python3
"""
05_run_backtest.py — Backtest persistant (Supabase)
==================================================
Récupère les matchs terminés d'hier (ou N jours),
compare avec les prédictions figées (daily_projections_v2)
et insère le résultat dans backtest_history.

Usage:
  python 05_run_backtest.py
  python 05_run_backtest.py --days 3
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

from database import get_client

PARIS_TZ = pytz.timezone("Europe/Paris")
CHUNK_SIZE = 200


def _today_paris() -> date:
    return datetime.now(PARIS_TZ).date()


def _parse_date(date_str: str) -> Optional[date]:
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _get_team_name(supabase, team_id: int) -> str:
    if not supabase or not team_id:
        return f"Équipe {team_id}"
    try:
        r = supabase.table("teams_metadata").select("nom_equipe, name").eq("team_id", team_id).limit(1).execute()
        if r.data:
            name = (r.data[0].get("nom_equipe") or r.data[0].get("name") or "").strip()
            if name:
                return name
    except Exception:
        pass
    return f"Équipe {team_id}"


def _split_match_name(match_name: str) -> tuple[str, str]:
    if " vs " in match_name:
        home, away = match_name.split(" vs ", 1)
        return home.strip(), away.strip()
    return match_name.strip(), ""


def _bet_from_le_pari(le_pari: Optional[str], home: str, away: str) -> Optional[str]:
    if not le_pari:
        return None
    txt = le_pari.strip().lower()
    if "passer" in txt or "pass" in txt:
        return None
    if "over" in txt:
        return "Over"
    if "under" in txt:
        return "Under"
    if home and home.lower() in txt:
        return "Home"
    if away and away.lower() in txt:
        return "Away"
    return None


def _prediction_winner(prob_home: Optional[float], home: str, away: str) -> Optional[str]:
    if prob_home is None:
        return None
    return home if prob_home >= 0.5 else away


def _status_and_profit(
    bet: Optional[str],
    odds: Optional[float],
    actual_winner: Optional[str],
    line_total: Optional[float],
    actual_total: Optional[float],
) -> tuple[str, float]:
    if bet in ("Home", "Away"):
        if not odds or not actual_winner:
            return "PUSH", 0.0
        win = (bet == "Home" and actual_winner == "HOME") or (bet == "Away" and actual_winner == "AWAY")
        return ("WIN", float(odds) - 1.0) if win else ("LOSS", -1.0)
    if bet in ("Over", "Under"):
        if line_total is None or actual_total is None:
            return "PUSH", 0.0
        if actual_total == line_total:
            return "PUSH", 0.0
        win = (bet == "Over" and actual_total > line_total) or (bet == "Under" and actual_total < line_total)
        if not odds:
            return ("WIN" if win else "LOSS"), 0.0
        return ("WIN", float(odds) - 1.0) if win else ("LOSS", -1.0)
    return "PUSH", 0.0


def _fetch_finished_games(supabase, start_date: str, end_date: str) -> List[dict]:
    if not supabase:
        return []
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, date, home_id, away_id, home_score, away_score")
            .gte("date", start_date[:10])
            .lte("date", end_date[:10])
            .not_.is_("home_score", "null")
            .not_.is_("away_score", "null")
            .order("date", desc=False)
            .execute()
        )
        return r.data or []
    except Exception:
        return []


def _fetch_predictions_map(supabase, game_ids: List[int]) -> Dict[int, dict]:
    if not supabase or not game_ids:
        return {}
    out: Dict[int, dict] = {}
    for i in range(0, len(game_ids), CHUNK_SIZE):
        chunk = game_ids[i : i + CHUNK_SIZE]
        try:
            r = (
                supabase.table("daily_projections_v2")
                .select(
                    "game_id, date_prediction, match_name, proba_ml_calibrated, total_points_projected,"
                    "bookmaker_odds_home, bookmaker_odds_away, bookmaker_line_total, le_pari"
                )
                .in_("game_id", chunk)
                .order("date_prediction", desc=True)
                .execute()
            )
            rows = r.data or []
        except Exception:
            rows = []
        for row in rows:
            gid = row.get("game_id")
            if gid is None:
                continue
            prev = out.get(gid)
            if not prev:
                out[gid] = row
                continue
            prev_date = str(prev.get("date_prediction") or "")[:10]
            curr_date = str(row.get("date_prediction") or "")[:10]
            if curr_date > prev_date:
                out[gid] = row
    return out


def run_backtest(days: int = 1) -> int:
    if days < 1:
        return 0
    supabase = get_client()
    if not supabase:
        print("❌ Connexion Supabase impossible.")
        return 0

    today = _today_paris()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    games = _fetch_finished_games(supabase, start_str, end_str)
    if not games:
        print(f"   Aucun match terminé entre {start_str} et {end_str}.")
        return 0

    game_ids = [int(g.get("game_id")) for g in games if g.get("game_id")]
    preds_map = _fetch_predictions_map(supabase, game_ids)

    upserted = 0
    for g in games:
        gid = g.get("game_id")
        if not gid:
            continue
        pred = preds_map.get(gid)
        if not pred:
            continue

        match_date = str(g.get("date") or "")[:10]
        if len(match_date) != 10:
            continue

        home_id = int(g.get("home_id") or 0)
        away_id = int(g.get("away_id") or 0)
        home_score = g.get("home_score")
        away_score = g.get("away_score")
        if home_score is None or away_score is None:
            continue

        match_name = (pred.get("match_name") or "").strip()
        if not match_name:
            home_name = _get_team_name(supabase, home_id)
            away_name = _get_team_name(supabase, away_id)
            match_name = f"{home_name} vs {away_name}"
        home_name, away_name = _split_match_name(match_name)

        prob = pred.get("proba_ml_calibrated")
        prob = float(prob) if prob is not None else None
        prediction_winner = _prediction_winner(prob, home_name, away_name)

        total_pred = pred.get("total_points_projected")
        total_pred = float(total_pred) if total_pred is not None else None

        actual_winner = "HOME" if float(home_score) > float(away_score) else "AWAY"
        actual_total = float(home_score) + float(away_score)

        le_pari = pred.get("le_pari") or ""
        bet_suggested = _bet_from_le_pari(le_pari, home_name, away_name)
        if not bet_suggested and not le_pari:
            bet_suggested = "Home" if prediction_winner == home_name else "Away"

        odds_taken = None
        if bet_suggested == "Home":
            odds_taken = pred.get("bookmaker_odds_home")
        elif bet_suggested == "Away":
            odds_taken = pred.get("bookmaker_odds_away")

        line_total = pred.get("bookmaker_line_total")
        status, profit = _status_and_profit(
            bet_suggested,
            float(odds_taken) if odds_taken is not None else None,
            actual_winner,
            float(line_total) if line_total is not None else None,
            actual_total,
        )

        payload = {
            "game_id": gid,
            "match_date": match_date,
            "match_name": match_name,
            "prediction_winner": prediction_winner,
            "prediction_proba": prob,
            "prediction_total_points": total_pred,
            "actual_winner": actual_winner,
            "actual_total_points": actual_total,
            "bet_suggested": bet_suggested,
            "odds_taken": float(odds_taken) if odds_taken is not None else None,
            "profit": float(profit),
            "status": status,
        }

        try:
            supabase.table("backtest_history").upsert(payload, on_conflict="game_id").execute()
            upserted += 1
        except Exception as e:
            print(f"   ⚠️ game_id {gid}: {e}")

    return upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest persistant → backtest_history (Supabase)")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Nombre de jours passés à traiter (défaut: 1 = hier)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("📊 05_run_backtest — Backtest persistant")
    print("=" * 55)
    print(f"   Fenêtre : {args.days} jour(s)")

    n = run_backtest(days=args.days)
    print(f"\n✅ {n} ligne(s) upsertées dans backtest_history.\n")


if __name__ == "__main__":
    main()
