#!/usr/bin/env python3
"""
backfill_past_days.py — Remplir Supabase avec les matchs des N derniers jours
================================================================================
Récupère les matchs terminés (scores + box_scores) sur les 15 derniers jours
(ou un autre nombre via --days) et les insère dans games_history + box_scores.
Idéal pour peupler la base au démarrage ou après une période sans ingestion.

Réutilise backend_engine.ingest_recent_games() et database.get_client().

Usage:
  python backfill_past_days.py              # 15 derniers jours, max 400 matchs
  python backfill_past_days.py --days 7     # 7 derniers jours
  python backfill_past_days.py --days 30 --max-games 1000
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

from database import get_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Récupérer les matchs des N derniers jours et peupler Supabase (games_history + box_scores)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Nombre de jours passés à récupérer (défaut: 15)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=400,
        help="Nombre max de matchs à ingérer au total (défaut: 400)",
    )
    parser.add_argument(
        "--skip-archetypes",
        action="store_true",
        help="Ne pas mettre à jour les archétypes équipes à la fin",
    )
    args = parser.parse_args()

    if get_client() is None:
        print("❌ Connexion Supabase impossible. Vérifiez .env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).")
        sys.exit(1)

    try:
        from backend_engine import ingest_recent_games, update_team_archetypes
    except ImportError as e:
        print(f"❌ Import backend_engine: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"📥 BACKFILL — {args.days} derniers jours (max {args.max_games} matchs)")
    print("=" * 60)

    n = ingest_recent_games(days=args.days, max_games_per_run=args.max_games)
    print(f"\n✅ {n} match(s) ingérés sur les {args.days} derniers jours.")

    if not args.skip_archetypes:
        print("\n🔄 Mise à jour des archétypes équipes...")
        update_team_archetypes()
        print("   → Archétypes à jour.")

    print("\n" + "=" * 60)
    print("   Vous pouvez lancer 02_train_models puis 03_predict_daily.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
