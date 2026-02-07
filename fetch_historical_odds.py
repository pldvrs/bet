#!/usr/bin/env python3
"""
Data Engineer API-Basketball — Remplissage des cotes historiques (home_odd, away_odd).
Sélectionne les matchs de games_history où home_odd IS NULL, appelle /odds?game={id},
extrait les cotes Betclic (17) ou Unibet (7), met à jour la base.
Rate limit : time.sleep(1.2) entre chaque appel API.

Usage:
  python3 fetch_historical_odds.py        # uniquement les matchs où home_odd est NULL
  python3 fetch_historical_odds.py --all  # tous les matchs (backfill complet)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from database import get_client

# Charger .env
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

BASE_URL = "https://v1.basketball.api-sports.io"
BOOKMAKER_IDS = [17, 7]  # Betclic, Unibet (priorité)
BET_HOME_AWAY_ID = 2  # Moneyline Home/Away
API_RETRIES = 3
API_RETRY_DELAY = 1.0
SLEEP_BETWEEN_CALLS = 1.2


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


def fetch_odds_for_game(game_id: int, league_id: int, season: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Appelle /odds?game={id}&league=&season= et extrait home/away du bookmaker 17 ou 7.
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
    # Fallback : n'importe quel bookmaker
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


def check_columns_exist(supabase) -> bool:
    """Vérifie que les colonnes home_odd et away_odd existent."""
    try:
        supabase.table("games_history").select("game_id, home_odd, away_odd").limit(1).execute()
        return True
    except Exception:
        return False


def get_matches_missing_odds(supabase, all_games: bool = False) -> List[Dict[str, Any]]:
    """Matchs où home_odd est NULL, ou tous les matchs si all_games=True."""
    try:
        q = supabase.table("games_history").select("game_id, league_id, season")
        if not all_games:
            q = q.is_("home_odd", "null")
        r = q.execute()
        return r.data or []
    except Exception as e:
        print(f"Erreur sélection: {e}")
        return []


def update_game_odds(supabase, game_id: int, home_odd: float, away_odd: float) -> bool:
    """Met à jour home_odd et away_odd pour un game_id."""
    try:
        supabase.table("games_history").update({
            "home_odd": home_odd,
            "away_odd": away_odd,
        }).eq("game_id", game_id).execute()
        return True
    except Exception as e:
        print(f"Erreur update game_id={game_id}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Remplir home_odd/away_odd dans games_history")
    parser.add_argument("--all", action="store_true", help="Traiter tous les matchs (backfill complet)")
    args = parser.parse_args()

    supabase = get_client()
    if not supabase:
        print("Erreur: Supabase indisponible. Vérifiez .env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).")
        return
    if not (os.environ.get("API_BASKETBALL_KEY") or "").strip():
        print("Erreur: API_BASKETBALL_KEY manquante dans .env")
        return

    if not check_columns_exist(supabase):
        print("Les colonnes home_odd et away_odd n'existent pas.")
        print("Exécutez d'abord la migration SQL :")
        print("  schema_migration_odds.sql")
        print("  (Supabase Dashboard → SQL Editor, ou psql)")
        return

    rows = get_matches_missing_odds(supabase, all_games=args.all)
    if not rows:
        if args.all:
            print("Aucun match dans games_history.")
        else:
            print("Aucun match avec home_odd NULL.")
            print("Pour remplir tous les matchs quand même : python3 fetch_historical_odds.py --all")
        return

    mode = "backfill complet" if args.all else "home_odd NULL"
    print(f"Matchs à traiter ({mode}): {len(rows)}. Délai entre appels: {SLEEP_BETWEEN_CALLS}s")
    updated = 0
    failed = 0
    no_odds = 0

    for i, row in enumerate(rows, 1):
        game_id = row.get("game_id")
        league_id = row.get("league_id")
        season = (row.get("season") or "").strip() or "2025"
        if not game_id:
            continue
        home_odd, away_odd = fetch_odds_for_game(game_id, league_id or 0, season)
        time.sleep(SLEEP_BETWEEN_CALLS)
        if home_odd is not None and away_odd is not None:
            if update_game_odds(supabase, game_id, home_odd, away_odd):
                updated += 1
                print(f"[{i}/{len(rows)}] game_id={game_id} → home_odd={home_odd:.2f}, away_odd={away_odd:.2f} ✅")
            else:
                failed += 1
        else:
            no_odds += 1
            print(f"[{i}/{len(rows)}] game_id={game_id} → pas de cotes (Betclic/Unibet) ⚠️")

    print(f"\nTerminé. Mis à jour: {updated}, sans cotes: {no_odds}, erreurs update: {failed}")


if __name__ == "__main__":
    main()
