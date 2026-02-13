#!/usr/bin/env python3
"""
02_train_models.py — L'Entraîneur (Architecture ETL Sniper V1 Pro)
=====================================================================
Récupère tout le Feature Engineering de training_engine.py, entraîne les 3 modèles
(Proba, Spread, Totals) et sauvegarde model_proba.pkl, model_spread.pkl, model_totals.pkl.
Charge dynamiquement toutes les ligues présentes en base (games_history + box_scores) :
aucun filtre par league_id — EuroCup, BCL, EuroLeague, etc. sont inclus automatiquement.

Correction CRITIQUE Totals :
  - Aucune constante 150 ni total_est.
  - Features du modèle Totals : Pace_Home, Pace_Away, Def_Rtg_Home, Def_Rtg_Away.
  - Target : home_score + away_score (Total_Points).

Usage:
  python 02_train_models.py
  python 02_train_models.py --test-months 3
"""

import argparse
import json
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

# Feature engineering et dataset depuis training_engine
from training_engine import (
    FEATURE_NAMES,
    build_training_dataset,
)
from training_engine import _get_classifier, _get_regressor

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PROBA_PATH = SCRIPT_DIR / "model_proba.pkl"
MODEL_SPREAD_PATH = SCRIPT_DIR / "model_spread.pkl"
MODEL_TOTALS_PATH = SCRIPT_DIR / "model_totals.pkl"
SCALER_PATH = SCRIPT_DIR / "scaler.pkl"
SCALER_TOTALS_PATH = SCRIPT_DIR / "scaler_totals.pkl"
FEATURES_META_PATH = SCRIPT_DIR / "features_meta.json"
FEATURES_META_TOTALS_PATH = SCRIPT_DIR / "features_meta_totals.json"

# Modèle Totals : uniquement ces 4 features (pas de League_Avg_Total, pas de 150)
FEATURE_NAMES_TOTALS: List[str] = [
    "Pace_Home",
    "Pace_Away",
    "Def_Rtg_Home",
    "Def_Rtg_Away",
]

TEST_MONTHS_DEFAULT = 2


def _train_proba_spread(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Entraîne classifier (proba) et regressor (spread). Retourne result partiel."""
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    X_train = train_df[FEATURE_NAMES].copy()
    y_win_train = train_df["Win_Home"]
    y_diff_train = train_df["Score_Diff"]
    X_test = test_df[FEATURE_NAMES].copy()
    y_win_test = test_df["Win_Home"]
    y_diff_test = test_df["Score_Diff"]

    for col in FEATURE_NAMES:
        if col in X_train.columns and X_train[col].isna().any():
            m = X_train[col].mean()
            X_train[col] = X_train[col].fillna(m)
            X_test[col] = X_test[col].fillna(m)
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_NAMES, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_NAMES, index=X_test.index
    )

    clf = _get_classifier()
    reg = _get_regressor()
    clf.fit(X_train_s, y_win_train)
    reg.fit(X_train_s, y_diff_train)

    y_proba = clf.predict_proba(X_test_s)[:, 1]
    y_pred_diff = reg.predict(X_test_s)
    metrics = {
        "accuracy": float(accuracy_score(y_win_test, clf.predict(X_test_s))),
        "roc_auc": float(roc_auc_score(y_win_test, y_proba)) if len(np.unique(y_win_test)) > 1 else 0.5,
        "log_loss": float(log_loss(y_win_test, y_proba)),
        "mae_spread": float(mean_absolute_error(y_diff_test, y_pred_diff)),
    }
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_win_test, y_proba, n_bins=10)
        metrics["calibration"] = [
            {"predicted_bin": round(float(p), 2), "actual_win_rate": round(float(t), 2)}
            for t, p in zip(prob_true, prob_pred)
        ]
    except Exception:
        metrics["calibration"] = []

    return {
        "classifier": clf,
        "regressor": reg,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "train_means": X_train.mean().to_dict(),
        "train_stds": X_train.std().replace(0, 1).to_dict(),
        "metrics": metrics,
    }


LEAGUE_CALIBRATION_LAST_N = 30  # Nombre de matchs par ligue pour avg_real / avg_pred


def _compute_league_calibration(
    df: pd.DataFrame,
    totals_reg,
    scaler_totals,
) -> Dict[int, float]:
    """
    Calibration par ligue (Reality Check) : pour chaque ligue, sur les N derniers matchs,
    delta = avg_real_score - avg_pred_score. Prediction_Finale = Prediction_Brute + delta.
    """
    if "league_id" not in df.columns or totals_reg is None or scaler_totals is None:
        return {}
    league_calibration = {}
    df_sorted = df.sort_values("date", ascending=True)
    for league_id, grp in df_sorted.groupby("league_id"):
        lid = int(league_id) if league_id is not None else None
        if lid is None:
            continue
        sub = grp.tail(LEAGUE_CALIBRATION_LAST_N)
        if len(sub) < 5:
            continue
        X = sub[FEATURE_NAMES_TOTALS].copy().fillna(0).replace([np.inf, -np.inf], 0)
        X_s = scaler_totals.transform(X)
        preds = totals_reg.predict(X_s)
        avg_real = float(sub["Total_Points"].mean())
        avg_pred = float(np.mean(preds))
        delta = avg_real - avg_pred
        league_calibration[lid] = round(delta, 2)
    return league_calibration


def _train_totals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Entraîne le modèle Total Points : target = home_score + away_score.
    Features : Pace_Home, Pace_Away, Def_Rtg_Home, Def_Rtg_Away uniquement.
    Aucune constante 150. Calcule league_calibration (Reality Check).
    """
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    for col in FEATURE_NAMES_TOTALS:
        if col not in train_df.columns:
            return {"error": f"Colonne manquante pour Totals: {col}"}

    X_train = train_df[FEATURE_NAMES_TOTALS].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df["Total_Points"]
    X_test = test_df[FEATURE_NAMES_TOTALS].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df["Total_Points"]

    scaler_totals = StandardScaler()
    X_train_s = scaler_totals.fit_transform(X_train)
    X_test_s = scaler_totals.transform(X_test)

    totals_reg = _get_regressor()
    totals_reg.fit(X_train_s, y_train)
    y_pred = totals_reg.predict(X_test_s)
    mae_total = float(mean_absolute_error(y_test, y_pred))

    league_calibration = {}
    if "league_id" in full_df.columns:
        league_calibration = _compute_league_calibration(full_df, totals_reg, scaler_totals)

    return {
        "totals_regressor": totals_reg,
        "scaler_totals": scaler_totals,
        "feature_names_totals": FEATURE_NAMES_TOTALS,
        "totals_train_means": train_df[FEATURE_NAMES_TOTALS].mean().to_dict(),
        "mae_total": mae_total,
        "league_calibration": league_calibration,
    }


def save_models(proba_spread_result: Dict[str, Any], totals_result: Dict[str, Any]) -> None:
    """Sauvegarde model_proba.pkl, model_spread.pkl, model_totals.pkl + scalers + meta."""
    clf = proba_spread_result.get("classifier")
    reg = proba_spread_result.get("regressor")
    scaler = proba_spread_result.get("scaler")
    if clf is None or reg is None:
        raise ValueError("Modèles Proba/Spread manquants")

    with open(MODEL_PROBA_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_SPREAD_PATH, "wb") as f:
        pickle.dump(reg, f)
    if scaler is not None:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    meta = {
        "feature_names": proba_spread_result.get("feature_names", FEATURE_NAMES),
        "train_means": proba_spread_result.get("train_means", {}),
        "train_stds": proba_spread_result.get("train_stds", {}),
    }
    if proba_spread_result.get("metrics", {}).get("calibration"):
        meta["calibration"] = proba_spread_result["metrics"]["calibration"]
    with open(FEATURES_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    totals_reg = totals_result.get("totals_regressor")
    scaler_totals = totals_result.get("scaler_totals")
    if totals_reg is not None:
        with open(MODEL_TOTALS_PATH, "wb") as f:
            pickle.dump(totals_reg, f)
    if scaler_totals is not None:
        with open(SCALER_TOTALS_PATH, "wb") as f:
            pickle.dump(scaler_totals, f)
    if totals_result.get("feature_names_totals"):
        meta_totals = {
            "feature_names": totals_result["feature_names_totals"],
            "train_means": totals_result.get("totals_train_means", {}),
            "mae_total": totals_result.get("mae_total"),
            "league_calibration": totals_result.get("league_calibration", {}),
        }
        with open(FEATURES_META_TOTALS_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_totals, f, indent=2)


def run_training(test_months: int = TEST_MONTHS_DEFAULT) -> None:
    """Pipeline : build dataset → split temporel → train proba/spread → train totals → save."""
    print("\n" + "=" * 50)
    print("🏋️ 02_train_models — L'Entraîneur")
    print("=" * 50)

    df, err = build_training_dataset()
    if err or df.empty:
        print(f"❌ Dataset : {err or 'vide'}")
        return

    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    max_date = df["date"].max()
    if pd.isna(max_date):
        print("❌ Dates invalides")
        return

    print(f"\n📂 Chargement : tout l'historique games_history (matchs terminés) → {len(df)} matchs.")

    test_start = max_date - timedelta(days=test_months * 31)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()

    if len(train_df) < 50:
        print("❌ Pas assez de données (min 50 lignes train)")
        return
    if len(test_df) < 5:
        print("❌ Pas assez de matchs en test (2 derniers mois)")
        return

    print(f"\n📊 Train : {len(train_df)} | Test : {len(test_df)} (à partir de {test_start.date()})")

    # Vérifier présence des 4 features Totals (Pace_Home, Pace_Away, Def_Rtg_Home, Def_Rtg_Away)
    for c in FEATURE_NAMES_TOTALS:
        if c not in df.columns:
            print(f"❌ Colonne Totals manquante : {c}. Lance 01_ingest_data puis relance l'entraînement.")
            return

    print("\n📈 Entraînement Proba + Spread...")
    proba_spread_result = _train_proba_spread(train_df, test_df)
    m = proba_spread_result.get("metrics", {})
    print(f"   Accuracy: {m.get('accuracy', 0):.2%} | ROC-AUC: {m.get('roc_auc', 0):.3f} | MAE Spread: {m.get('mae_spread', 0):.2f}")

    print("\n📈 Entraînement Totals (Pace_Home, Pace_Away, Def_Rtg_Home, Def_Rtg_Away → Total_Points)...")
    totals_result = _train_totals(train_df, test_df, df)
    if totals_result.get("error"):
        print(f"   ❌ {totals_result['error']}")
        return
    print(f"   MAE Total: {totals_result.get('mae_total', 0):.2f} pts")
    if totals_result.get("league_calibration"):
        print(f"   Calibration par ligue (Reality Check) : {len(totals_result['league_calibration'])} ligue(s)")

    print("\n💾 Sauvegarde des modèles...")
    save_models(proba_spread_result, totals_result)
    print(f"   → {MODEL_PROBA_PATH.name}, {MODEL_SPREAD_PATH.name}, {MODEL_TOTALS_PATH.name}")
    print(f"   → {SCALER_PATH.name}, {SCALER_TOTALS_PATH.name}, features_meta*.json")

    last_match_date = max_date.date() if hasattr(max_date, "date") else str(max_date)[:10]
    print(f"\n✅ Entraînement terminé sur {len(df)} matchs. Dernier match inclus : {last_match_date}.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement Proba + Spread + Totals (Sniper ETL)")
    parser.add_argument(
        "--test-months",
        type=int,
        default=TEST_MONTHS_DEFAULT,
        help="Nombre de mois pour la période de test (défaut: 2)",
    )
    args = parser.parse_args()
    run_training(test_months=args.test_months)


if __name__ == "__main__":
    main()
