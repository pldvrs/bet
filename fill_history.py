#!/usr/bin/env python3
"""
fill_history.py — Collecte des résultats et mise à jour de la base
==================================================================
Utilisé par le workflow GitHub Actions pour le pipeline Sniper.

Usage:
  python fill_history.py --mode daily   # Résultats de la veille + Box Scores + Calendrier du jour

Look-Ahead Bias : On ne collecte que les matchs TERMINÉS (hier).
Les matchs du jour (à venir) sont ajoutés au calendrier sans scores.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collecte des résultats basketball → Supabase (pipeline Sniper)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="daily",
        choices=["daily"],
        help="Mode d'exécution (daily = veille + calendrier jour)",
    )
    args = parser.parse_args()

    if args.mode != "daily":
        print(f"Mode inconnu: {args.mode}")
        sys.exit(1)

    # Import du backend après chargement .env
    try:
        from backend_engine import (
            ingest_recent_games,
            fetch_future_games,
            update_team_archetypes,
        )
    except ImportError as e:
        print(f"Erreur import backend_engine: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("📥 FILL HISTORY — Mode daily")
    print("=" * 50)

    # Étape 1 : Résultats de la veille + Box Scores
    # (Train = jusqu'à hier 23:59, jamais les scores d'aujourd'hui)
    print("\n📅 Étape 1 : Collecte des résultats de la veille + Box Scores")
    n_past = ingest_recent_games(days=1, max_games_per_run=50)
    print(f"   → {n_past} match(s) ingérés (scores + box scores)")

    # Étape 2 : Calendrier du jour + cotes fraîches (matchs à venir)
    print("\n📅 Étape 2 : Calendrier du jour (matchs à venir)")
    n_future = fetch_future_games(days=3, max_games=150)
    print(f"   → {n_future} match(s) à venir ajoutés au calendrier")

    # Étape 3 : Mise à jour des archétypes (optionnel, rapide)
    print("\n🔄 Étape 3 : Mise à jour des archétypes")
    update_team_archetypes()

    print("\n✅ fill_history terminé.")
    print("   Règle Look-Ahead : Train = matchs avec scores (passé) | Predict = matchs sans scores (futur)\n")


if __name__ == "__main__":
    main()
