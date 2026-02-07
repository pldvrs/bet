#!/usr/bin/env python3
"""
auto_refresher.py — Fine-tuning / Mise à jour quotidienne des modèles ML
========================================================================
Récupère les derniers matchs terminés, ré-entraîne les modèles avec les données
à jour. Utilise le split temporel strict du training_engine (TimeSeriesSplit).

Look-Ahead Bias : Train = matchs jusqu'à HIER 23:59 uniquement.
                  Pred = à partir d'AUJOURD'HUI 00:01.

Le training_engine trie toujours par date et utilise une coupure nette :
- Passé (Train) : jusqu'à hier soir 23:59
- Futur (Prediction) : à partir d'aujourd'hui 00:01
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


def main() -> None:
    print("\n" + "=" * 50)
    print("🔄 AUTO REFRESHER — Mise à jour des modèles ML")
    print("=" * 50)
    print("   Règle Look-Ahead : Train ≤ hier 23:59 | Predict ≥ aujourd'hui 00:01\n")

    try:
        from training_engine import main_cli
    except ImportError as e:
        print(f"   ❌ Erreur import training_engine: {e}")
        sys.exit(1)

    main_cli()
    print("\n✅ auto_refresher terminé.\n")


if __name__ == "__main__":
    main()
