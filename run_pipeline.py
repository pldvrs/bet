#!/usr/bin/env python3
"""
run_pipeline.py — Le Chef d'Orchestre (Sniper V1 Pro)
=====================================================
Lance séquentiellement : 01_ingest_data → 02_train_models → 03_predict_daily.
Logs clairs avec temps d'exécution par étape.

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-train   # sauter l'entraînement (02)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent


def _run_script(name: str, script: Path, args: list = None) -> bool:
    """Exécute un script Python et retourne True si succès (exit 0)."""
    cmd = [sys.executable, str(script)] + (args or [])
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=False)
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            print(f"✅ Étape terminée en {elapsed:.1f}s\n")
            return True
        print(f"❌ Étape en erreur (code {result.returncode}) après {elapsed:.1f}s\n")
        return False
    except Exception as e:
        print(f"❌ Exception : {e}\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Sniper : 01 → 02 → 03")
    parser.add_argument("--skip-train", action="store_true", help="Ne pas lancer 02_train_models.py")
    parser.add_argument(
        "--days-past",
        type=int,
        default=1,
        help="Jours passés pour 01_ingest_data (défaut: 1 = veille)",
    )
    parser.add_argument(
        "--days-future",
        type=int,
        default=3,
        help="Jours à venir pour 01_ingest_data (défaut: 3)",
    )
    parser.add_argument(
        "--max-games-past",
        type=int,
        default=50,
        help="Nombre max de matchs passés (défaut: 50)",
    )
    parser.add_argument(
        "--max-games-future",
        type=int,
        default=150,
        help="Nombre max de matchs à venir (défaut: 150)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🎯 SNIPER V1 PRO — Chef d'Orchestre")
    print("=" * 60)

    # Étape 1 : Ingestion
    print("\n📥 Étape 1/3 : 01_ingest_data.py (scores + box_scores + cotes)")
    ingest_args = [
        "--days-past", str(args.days_past),
        "--days-future", str(args.days_future),
        "--max-games-past", str(args.max_games_past),
        "--max-games-future", str(args.max_games_future),
    ]
    if not _run_script("01", SCRIPT_DIR / "01_ingest_data.py", ingest_args):
        sys.exit(1)

    # Étape 2 : Entraînement (optionnel)
    if not args.skip_train:
        print("\n🏋️ Étape 2/3 : 02_train_models.py (ré-entraînement)")
        if not _run_script("02", SCRIPT_DIR / "02_train_models.py"):
            sys.exit(1)
    else:
        print("\n⏭️ Étape 2/3 : 02_train_models.py (ignoré --skip-train)")

    # Étape 3 : Prédictions
    step = 3 if not args.skip_train else 2
    print(f"\n🧠 Étape {step}/5 : 03_predict_daily.py (génération des prédictions)")
    if not _run_script("03", SCRIPT_DIR / "03_predict_daily.py"):
        sys.exit(1)

    # Étape 4 : Backfill backtest (prédictions manquantes)
    print("\n🧩 Étape 4/5 : fix_missing_predictions.py --auto (backtest)")
    _run_script("04", SCRIPT_DIR / "fix_missing_predictions.py", ["--auto"])

    # Étape 5 : Backtest persistant (yesterday)
    print("\n📊 Étape 5/5 : 05_run_backtest.py --days 1")
    _run_script("05", SCRIPT_DIR / "05_run_backtest.py", ["--days", "1"])

    print("=" * 60)
    print("✅ Pipeline terminé. Lancez le dashboard : streamlit run 04_app_dashboard.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
