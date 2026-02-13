#!/usr/bin/env python3
"""
01_ingest_data.py — L'Aspirateur (Architecture ETL Sniper V1 Pro)
==================================================================
Met à jour games_history (scores d'hier + calendrier J+3), box_scores, et les cotes (Odds).
Aucun appel API dans le front-end : toute la donnée est ingérée ici.

Sources réutilisées :
  - fill_history.py → workflow daily (ingest_recent_games, fetch_future_games, update_team_archetypes)
  - backend_engine.py → ingestion API Basketball + Supabase
  - fetch_historical_odds.py / app_sniper_v27.py → récupération cotes (home/away)

Usage:
  python 01_ingest_data.py              # mode daily (veille + J+3 + cotes)
  python 01_ingest_data.py --days 5     # Deep Fetch : 5 derniers jours (résultats FT + upsert scores)
  python 01_ingest_data.py --no-odds    # sans récupération des cotes
  python 01_ingest_data.py --days-past 2 --days-future 5
  python 01_ingest_data.py --init-leagues 121,16   # Backfill EuroCup + BCL (2023-2024, 2024-2025)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# Connexion Supabase (réutilisation database.py)
from database import get_client

# -----------------------------------------------------------------------------
# CONFIG API (alignée app_sniper_v27 / fetch_historical_odds)
# -----------------------------------------------------------------------------
BASE_URL = "https://v1.basketball.api-sports.io"
BOOKMAKER_IDS = [17, 7, 1]  # Betclic, Unibet, Bwin
BET_HOME_AWAY_ID = 2
API_RETRIES = 3
API_RETRY_DELAY = 1.0
ODDS_SLEEP = 0.6  # rate limit entre deux appels odds


def _api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
    """Appel API Basketball (sans Streamlit). Retourne (data, err)."""
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
        except Exception as e:
            last_err = str(e)
            time.sleep(API_RETRY_DELAY * (attempt + 1))
    return None, last_err


def fetch_odds_moneyline(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Cotes Home/Away pour un match. Priorité Betclic (17), Unibet (7), Bwin (1).
    Retourne (home_odd, away_odd) ou (None, None).
    """
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
    return None, None


def update_future_games_odds(supabase, max_games: int = 200) -> int:
    """
    Pour tous les matchs à venir (home_score IS NULL) dans games_history,
    récupère les cotes via l'API et met à jour home_odd, away_odd.
    Retourne le nombre de matchs mis à jour.
    """
    if not supabase:
        return 0
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, league_id, season")
            .is_("home_score", "null")
            .order("date", desc=False)
            .limit(max_games)
            .execute()
        )
        rows = r.data or []
    except Exception as e:
        print(f"   ⚠️ Lecture games_history: {e}")
        return 0

    updated = 0
    for row in rows:
        gid = row.get("game_id")
        league_id = row.get("league_id")
        season = row.get("season") or ""
        if not gid or league_id is None:
            continue
        oh, oa = fetch_odds_moneyline(gid, league_id, season)
        time.sleep(ODDS_SLEEP)
        if oh is None and oa is None:
            continue
        try:
            supabase.table("games_history").update({
                "home_odd": oh,
                "away_odd": oa,
            }).eq("game_id", gid).execute()
            updated += 1
        except Exception as e:
            # Colonnes home_odd/away_odd absentes si migration non exécutée
            if "home_odd" in str(e) or "PGRST204" in str(e):
                print("   ⚠️ Exécute schema_migration_odds.sql pour activer home_odd/away_odd.")
            break
    return updated


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------------------------


def run_ingest(
    days_past: int = 1,
    days_future: int = 3,
    max_games_past: int = 50,
    max_games_future: int = 150,
    fetch_odds: bool = True,
    skip_archetypes: bool = False,
) -> None:
    """
    Exécute le pipeline d'ingestion :
    1. Résultats de la veille (et J-2 si days_past > 1) + box_scores
    2. Calendrier des N prochains jours (matchs à venir)
    3. Mise à jour des archétypes (optionnel)
    4. Récupération des cotes pour les matchs à venir (optionnel)
    """
    supabase = get_client()
    if not supabase:
        print("❌ Connexion Supabase impossible. Vérifie .env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).")
        sys.exit(1)

    try:
        from backend_engine import (
            ingest_recent_games,
            fetch_future_games,
            update_team_archetypes,
        )
    except ImportError as e:
        print(f"❌ Import backend_engine: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("📥 01_ingest_data — L'Aspirateur")
    print("=" * 50)

    # Étape 1 : Scores passés + box_scores
    print("\n📅 Étape 1 : Résultats des derniers jours + Box Scores")
    n_past = ingest_recent_games(days=days_past, max_games_per_run=max_games_past)
    print(f"   → {n_past} match(s) ingérés (scores + box scores)")

    # Étape 2 : Calendrier à venir (J+1 à J+days_future)
    print(f"\n📅 Étape 2 : Calendrier J+1 à J+{days_future} (matchs à venir)")
    n_future = fetch_future_games(days=days_future, max_games=max_games_future)
    print(f"   → {n_future} match(s) à venir ajoutés au calendrier")

    # Étape 3 : Archétypes
    if not skip_archetypes:
        print("\n🔄 Étape 3 : Mise à jour des archétypes")
        update_team_archetypes()
    else:
        print("\n⏭️ Étape 3 : Archétypes ignorés (--skip-archetypes)")

    # Étape 4 : Cotes pour les matchs à venir
    if fetch_odds:
        print("\n📊 Étape 4 : Récupération des cotes (Odds) pour les matchs à venir")
        n_odds = update_future_games_odds(supabase, max_games=max_games_future)
        print(f"   → {n_odds} match(s) avec cotes mises à jour (games_history.home_odd / away_odd)")
    else:
        print("\n⏭️ Étape 4 : Cotes ignorées (--no-odds)")

    print("\n✅ 01_ingest_data terminé.")
    print("   Look-Ahead : Train = matchs avec scores (passé) | Predict = matchs sans scores (futur)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion données basketball + cotes → Supabase (Sniper ETL)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="Raccourci : nombre de jours passés pour les scores (ex: --days 5 = 5 derniers jours). Prioritaire sur --days-past.",
    )
    parser.add_argument(
        "--days-past",
        type=int,
        default=1,
        help="Jours passés pour récupérer les scores (défaut: 1 = veille)",
    )
    parser.add_argument(
        "--days-future",
        type=int,
        default=3,
        help="Jours à venir pour le calendrier (défaut: 3)",
    )
    parser.add_argument(
        "--max-games-past",
        type=int,
        default=50,
        help="Nombre max de matchs à ingérer dans le passé",
    )
    parser.add_argument(
        "--max-games-future",
        type=int,
        default=150,
        help="Nombre max de matchs à venir à ajouter",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Ne pas récupérer les cotes API pour les matchs à venir",
    )
    parser.add_argument(
        "--skip-archetypes",
        action="store_true",
        help="Ne pas mettre à jour les archétypes équipes",
    )
    parser.add_argument(
        "--init-leagues",
        type=str,
        default=None,
        metavar="ID1,ID2",
        help="Initialisation Ligue : backfill massif (saisons 2023-2024, 2024-2025, FT + box_scores) pour les IDs donnés. Ex: --init-leagues 121,16 (EuroCup + BCL). N'exécute pas le pipeline daily.",
    )
    args = parser.parse_args()

    # Mode backfill (Initialisation Ligue) : uniquement les ligues indiquées, 2 dernières saisons
    if args.init_leagues is not None:
        try:
            league_ids = [int(x.strip()) for x in args.init_leagues.split(",") if x.strip()]
        except ValueError:
            print("❌ --init-leagues : liste d'IDs entiers séparés par des virgules (ex: 121,16)")
            sys.exit(1)
        if not league_ids:
            print("❌ --init-leagues : au moins un league_id requis (ex: 121,16)")
            sys.exit(1)
        supabase = get_client()
        if not supabase:
            print("❌ Connexion Supabase impossible.")
            sys.exit(1)
        try:
            from backend_engine import backfill_league_seasons
        except ImportError as e:
            print(f"❌ Import backend_engine: {e}")
            sys.exit(1)
        print("\n" + "=" * 50)
        print("📥 01_ingest_data — Initialisation Ligue (Backfill)")
        print("=" * 50)
        print(f"   Ligues : {league_ids} | Saisons : 2023-2024, 2024-2025")
        print("   Récupération : tous les matchs FT + box_scores → games_history, box_scores (UPSERT)\n")
        backfill_league_seasons(league_ids=league_ids, seasons=["2023-2024", "2024-2025"])
        print("\n✅ Backfill terminé. Lance 02_train_models.py pour ré-entraîner le modèle.\n")
        return

    days_past = args.days if args.days is not None else args.days_past
    # En mode Deep Fetch (plus d'un jour), augmenter la cap pour ne rien rater
    max_games_past = max(args.max_games_past, days_past * 30) if days_past > 1 else args.max_games_past

    run_ingest(
        days_past=days_past,
        days_future=args.days_future,
        max_games_past=max_games_past,
        max_games_future=args.max_games_future,
        fetch_odds=not args.no_odds,
        skip_archetypes=args.skip_archetypes,
    )


if __name__ == "__main__":
    main()
