#!/usr/bin/env python3
"""
audit_model.py — Diagnostic pathologique du model_totals.pkl
============================================================
Lead Data Scientist : Validation de Modèle.

Time Machine rigoureuse : 500 derniers matchs, features N-10 recalculées sans triche.
Analyse des résidus, biais par équipe, scatter plot, détection des régimes Pace.

Sortie : Rapport terminal qui te crie dessus si le modèle sous-estime les meilleures
attaques de plus de 5 points en moyenne.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TOTALS_PATH = SCRIPT_DIR / "model_totals.pkl"
SCALER_TOTALS_PATH = SCRIPT_DIR / "scaler_totals.pkl"
FEATURES_META_TOTALS_PATH = SCRIPT_DIR / "features_meta_totals.json"

N_MATCHES: int = 500
BIAS_ALERT_THRESHOLD: float = 5.0  # Biais > 5 pts → ALERTE
BEST_ATTACK_OFFRTG_THRESHOLD: float = 215.0  # Combined_OffRtg > 215 = meilleures attaques
PACE_HIGH_THRESHOLD: float = 75.0  # Pace > 75 = régime rapide (suspect n°1)


def _get_supabase():
    from database import get_client
    return get_client()


def _get_team_name(team_id: int) -> str:
    supabase = _get_supabase()
    if not supabase:
        return f"Équipe {team_id}"
    try:
        r = supabase.table("teams_metadata").select("nom_equipe, name").eq("team_id", team_id).limit(1).execute()
        if r.data and len(r.data) > 0:
            nom = (r.data[0].get("nom_equipe") or r.data[0].get("name") or "").strip()
            if nom:
                return nom
    except Exception:
        pass
    return f"Équipe {team_id}"


def fetch_last_n_games(n: int) -> pd.DataFrame:
    """Récupère les N derniers matchs terminés (order by date DESC, puis reverse pour chrono)."""
    supabase = _get_supabase()
    if not supabase:
        return pd.DataFrame()
    all_data: List[dict] = []
    offset = 0
    chunk = 200
    try:
        while len(all_data) < n:
            r = (
                supabase.table("games_history")
                .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score")
                .not_.is_("home_score", "null")
                .not_.is_("away_score", "null")
                .order("date", desc=True)
                .range(offset, offset + chunk - 1)
                .execute()
            )
            chunk_data = r.data or []
            if not chunk_data:
                break
            all_data.extend(chunk_data)
            if len(chunk_data) < chunk:
                break
            offset += chunk
            if len(all_data) >= n:
                break
        df = pd.DataFrame(all_data[:n])
        # Ordre chronologique (plus ancien en premier) pour cohérence Time Machine
        if not df.empty and "date" in df.columns:
            df = df.sort_values("date", ascending=True).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def run_time_machine_backtest(n: int = N_MATCHES) -> pd.DataFrame:
    """
    Time Machine : pour chaque match, recalcule les features N-10 telles qu'à l'époque,
    prédit avec model_totals.pkl.
    Retourne DataFrame avec : game_id, date, home_id, away_id, home_name, away_name,
    real_total, pred_total, error (real - pred), pace_match, combined_off_rtg, ...
    """
    from training_engine import build_feature_row_for_match, FEATURE_NAMES_TOTALS

    if not MODEL_TOTALS_PATH.exists():
        print("❌ model_totals.pkl introuvable.")
        return pd.DataFrame()

    df_games = fetch_last_n_games(n)
    if df_games.empty:
        print("❌ Aucun match trouvé en base.")
        return pd.DataFrame()

    meta: Dict[str, Any] = {}
    if FEATURES_META_TOTALS_PATH.exists():
        with open(FEATURES_META_TOTALS_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    feature_names = meta.get("feature_names", FEATURE_NAMES_TOTALS)
    train_means = meta.get("train_means", {})

    with open(MODEL_TOTALS_PATH, "rb") as f:
        totals_reg = pickle.load(f)

    scaler = None
    if SCALER_TOTALS_PATH.exists():
        with open(SCALER_TOTALS_PATH, "rb") as f:
            scaler = pickle.load(f)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for _, g in df_games.iterrows():
        game_date = str(g.get("date") or "")[:10]
        if len(game_date) != 10:
            skipped += 1
            continue
        home_id = int(g.get("home_id", 0))
        away_id = int(g.get("away_id", 0))
        league_id = g.get("league_id")
        season = g.get("season", "")
        home_score = g.get("home_score")
        away_score = g.get("away_score")
        if home_score is None or away_score is None or not home_id or not away_id:
            skipped += 1
            continue

        row = build_feature_row_for_match(home_id, away_id, game_date, league_id, season)
        if row is None:
            skipped += 1
            continue

        X = pd.DataFrame([row])
        for col in feature_names:
            if col not in X.columns:
                X[col] = train_means.get(col, 0.0)
        X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        if scaler is not None:
            X = scaler.transform(X)
        X_arr = np.asarray(X, dtype=np.float64).reshape(1, -1)
        pred = float(totals_reg.predict(X_arr)[0])
        real = float(home_score) + float(away_score)
        err = real - pred

        pace_match = (row.get("Combined_Pace") or 0) / 2.0
        combined_off = row.get("Combined_OffRtg") or 0

        rows.append({
            "game_id": g.get("game_id"),
            "date": game_date,
            "home_id": home_id,
            "away_id": away_id,
            "home_name": _get_team_name(home_id),
            "away_name": _get_team_name(away_id),
            "real_total": real,
            "pred_total": pred,
            "error": err,
            "pace_match": pace_match,
            "combined_off_rtg": combined_off,
        })
    print(f"   Time Machine : {len(rows)} matchs traités, {skipped} ignorés (données insuffisantes)")
    return pd.DataFrame(rows)


def analyze_residuals_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Groupe par équipe : Moyenne Score Réel, Moyenne Score Prédit, Biais Moyen (MSE)."""
    if df.empty:
        return pd.DataFrame()

    team_rows: List[Dict[str, Any]] = []
    team_ids = set(df["home_id"].tolist()) | set(df["away_id"].tolist())

    for tid in team_ids:
        mask = (df["home_id"] == tid) | (df["away_id"] == tid)
        sub = df[mask]
        if len(sub) < 3:
            continue
        name = _get_team_name(tid)
        mean_real = sub["real_total"].mean()
        mean_pred = sub["pred_total"].mean()
        bias = sub["error"].mean()
        n = len(sub)
        team_rows.append({
            "team_id": tid,
            "Team": name,
            "N_Matchs": n,
            "Moy_Score_Réel": round(mean_real, 1),
            "Moy_Score_Prédit": round(mean_pred, 1),
            "Biais_Moyen": round(bias, 1),
        })

    team_df = pd.DataFrame(team_rows)
    team_df = team_df.sort_values("Biais_Moyen", ascending=False, key=abs)
    return team_df


def analyze_regime_by_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Performance par tranche de Pace. MAE explose quand Pace > 75 ?"""
    if df.empty or "pace_match" not in df.columns:
        return pd.DataFrame()

    bins = [0, 70, 72, 74, 76, 200]
    labels = ["<70", "70-72", "72-74", "74-76", ">76"]
    df = df.copy()
    df["pace_band"] = pd.cut(df["pace_match"], bins=bins, labels=labels)

    regime_rows = []
    for band in labels:
        sub = df[df["pace_band"] == band]
        if len(sub) < 5:
            continue
        mae = sub["error"].abs().mean()
        bias = sub["error"].mean()
        regime_rows.append({
            "Pace": band,
            "N": len(sub),
            "MAE": round(mae, 1),
            "Biais": round(bias, 1),
        })

    return pd.DataFrame(regime_rows)


def analyze_best_attacks(df: pd.DataFrame) -> Tuple[float, float]:
    """Matches où Combined_OffRtg > seuil. Retourne (biais_moyen, mae)."""
    if df.empty:
        return 0.0, 0.0
    sub = df[df["combined_off_rtg"] >= BEST_ATTACK_OFFRTG_THRESHOLD]
    if len(sub) < 3:
        return 0.0, 0.0
    return float(sub["error"].mean()), float(sub["error"].abs().mean())


def plot_scatter(df: pd.DataFrame, out_path: Optional[Path] = None) -> None:
    """Scatter : X=Score Prédit, Y=Score Réel, diagonale y=x."""
    if df.empty:
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(df["pred_total"], df["real_total"], alpha=0.5, s=20)
        mx = max(df["real_total"].max(), df["pred_total"].max())
        mn = min(df["real_total"].min(), df["pred_total"].min())
        ax.plot([mn, mx], [mn, mx], "r--", label="y=x (parfait)")
        ax.set_xlabel("Score Prédit")
        ax.set_ylabel("Score Réel")
        ax.set_title("Graphique de Vérité — model_totals.pkl\n(Si bon : points sur la ligne)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=120)
            print(f"   Graphique sauvegardé : {out_path}")
        else:
            plt.savefig(SCRIPT_DIR / "audit_scatter.png", dpi=120)
            print(f"   Graphique sauvegardé : audit_scatter.png")
        plt.close()
    except ImportError:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["pred_total"], y=df["real_total"], mode="markers", name="Matchs"))
            mx = max(df["real_total"].max(), df["pred_total"].max())
            mn = min(df["real_total"].min(), df["pred_total"].min())
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="y=x"))
            fig.update_layout(
                xaxis_title="Score Prédit",
                yaxis_title="Score Réel",
                title="Graphique de Vérité — model_totals.pkl",
            )
            out = out_path or (SCRIPT_DIR / "audit_scatter.html")
            fig.write_html(str(out))
            print(f"   Graphique sauvegardé : {out}")
        except ImportError:
            print("   ⚠️ matplotlib et plotly absents — pas de graphique.")


def main() -> None:
    print("\n" + "=" * 70)
    print("🔬 AUDIT MODEL — Diagnostic pathologique model_totals.pkl")
    print("=" * 70)
    print("   Time Machine rigoureuse : features N-10 recalculées sans triche.\n")

    df = run_time_machine_backtest(N_MATCHES)
    if df.empty:
        sys.exit(1)

    # Métriques globales
    mae_global = df["error"].abs().mean()
    bias_global = df["error"].mean()
    print("\n--- MÉTRIQUES GLOBALES ---")
    print(f"   MAE (Erreur absolue moyenne) : {mae_global:.1f} pts")
    print(f"   Biais moyen (Real - Pred)    : {bias_global:+.1f} pts")
    print(f"   Std des résidus              : {df['error'].std():.1f} pts")

    # Analyse par équipe
    print("\n--- BIAIS PAR ÉQUIPE (top 15 sous-estimés / surestimés) ---")
    team_df = analyze_residuals_by_team(df)
    if not team_df.empty:
        display_cols = ["Team", "N_Matchs", "Moy_Score_Réel", "Moy_Score_Prédit", "Biais_Moyen"]
        print(team_df[display_cols].head(15).to_string(index=False))
        print("   ...")
        print(team_df[display_cols].tail(5).to_string(index=False))

    # Détection Paris Basketball (ou équipe avec plus gros biais positif)
    worst_under = team_df[team_df["Biais_Moyen"] > 0].sort_values("Biais_Moyen", ascending=False)
    if not worst_under.empty:
        top = worst_under.iloc[0]
        print(f"\n   ⚠️ Équipe la plus SOUS-ESTIMÉE : {top['Team']} (Biais +{top['Biais_Moyen']:.1f} pts)")

    # Régime Pace
    print("\n--- DÉTECTION RÉGIMES PACE (MAE par tranche) ---")
    regime_df = analyze_regime_by_pace(df)
    if not regime_df.empty:
        print(regime_df.to_string(index=False))
        high_pace = df[df["pace_match"] >= PACE_HIGH_THRESHOLD]
        if len(high_pace) >= 5:
            mae_high = high_pace["error"].abs().mean()
            bias_high = high_pace["error"].mean()
            print(f"\n   Pace >= {PACE_HIGH_THRESHOLD} : N={len(high_pace)}, MAE={mae_high:.1f}, Biais={bias_high:+.1f}")

    # Meilleures attaques
    bias_best, mae_best = analyze_best_attacks(df)
    n_best = len(df[df["combined_off_rtg"] >= BEST_ATTACK_OFFRTG_THRESHOLD])
    print(f"\n--- MEILLEURES ATTAQUES (Combined_OffRtg > {BEST_ATTACK_OFFRTG_THRESHOLD}) ---")
    print(f"   N matchs : {n_best}")
    print(f"   Biais moyen : {bias_best:+.1f} pts")
    print(f"   MAE : {mae_best:.1f} pts")

    # ALERTE
    print("\n" + "=" * 70)
    if abs(bias_best) > BIAS_ALERT_THRESHOLD:
        print("🚨 ALERTE : Le modèle sous-estime les meilleures attaques de plus de 5 pts en moyenne !")
        print(f"   Biais sur matchs OffRtg>{BEST_ATTACK_OFFRTG_THRESHOLD} : {bias_best:+.1f} pts")
        print("   → Corriger par un Scaling Factor ou ré-entraînement ciblé.")
    else:
        print("✅ Pas d'alerte majeure sur les meilleures attaques.")
    print("=" * 70)

    # Graphique
    print("\n--- GRAPHIQUE DE VÉRITÉ ---")
    plot_scatter(df)

    # Export CSV pour analyse
    out_csv = SCRIPT_DIR / "audit_residuals.csv"
    df[["date", "home_name", "away_name", "real_total", "pred_total", "error", "pace_match", "combined_off_rtg"]].to_csv(
        out_csv, index=False
    )
    print(f"\n   Résidus exportés : {out_csv}")

    if not team_df.empty:
        team_csv = SCRIPT_DIR / "audit_by_team.csv"
        team_df.to_csv(team_csv, index=False)
        print(f"   Biais par équipe : {team_csv}")

    print("\n✅ Audit terminé.\n")


if __name__ == "__main__":
    main()
