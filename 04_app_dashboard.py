#!/usr/bin/env python3
"""
04_app_dashboard.py — Terminal de Prise de Décision (Write Once, Read Many)
=============================================================================
Lit UNIQUEMENT daily_projections_v2 (Supabase). Aucun calcul ML dans ce fichier.
Affiche un tableau stable : MATCH | MON PRONO | POURQUOI ? | CONFIANCE | EDGE.
Les prédictions ne changent pas au rechargement : elles sont figées par 03_predict_daily.py.

À lancer quand vous voulez consulter les pronos. Ne jamais lancer 03 depuis ici.

Usage: streamlit run 04_app_dashboard.py
"""

from pathlib import Path
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import streamlit as st

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from database import get_client

SCRIPT_DIR = Path(__file__).resolve().parent
BACKTEST_RESULTS_PATH = SCRIPT_DIR / "backtest_results.json"
PARIS_TZ = pytz.timezone("Europe/Paris")

# Seuils affichage (cohérents avec 03)
EDGE_VALUE = 5.0
EDGE_MAX = 10.0
FIABILITE_FIABLE = 70


def _today_paris() -> date:
    """Date du jour en Europe/Paris (cohérent avec 03_predict_daily)."""
    return datetime.now(PARIS_TZ).date()


def _day_label(game_date_str: str) -> str:
    """Aujourd'hui / Demain / J+2 (calculé à l'affichage, pas figé en base)."""
    if not game_date_str or len(game_date_str) < 10:
        return "—"
    try:
        d = datetime.strptime(game_date_str[:10], "%Y-%m-%d").date()
        today = _today_paris()
        delta = (d - today).days
        if delta == 0:
            return "Aujourd'hui"
        if delta == 1:
            return "Demain"
        if delta >= 2:
            return f"J+{delta}"
        if delta < 0:
            return f"Passé (J{delta})"
    except Exception:
        pass
    return "—"


def _get_supabase():
    return get_client()


@st.cache_data(ttl=120)
def _fetch_daily_projections_v2() -> List[dict]:
    """SELECT simple sur daily_projections_v2. Aucun calcul ML."""
    supabase = _get_supabase()
    if not supabase:
        return []
    try:
        r = (
            supabase.table("daily_projections_v2")
            .select("*")
            .order("edge_ml", desc=True)
            .execute()
        )
        return r.data or []
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Tableau principal — Lecture seule depuis daily_projections_v2
# -----------------------------------------------------------------------------


def _confiance_badge(edge_ml: float, confidence_score: float) -> str:
    """Badge CONFIANCE à partir des champs figés edge_ml et confidence_score."""
    if confidence_score < 30 or edge_ml < 0:
        return "🛑 PASS"
    if edge_ml >= EDGE_MAX and confidence_score >= FIABILITE_FIABLE:
        return "🔥 MAX BET"
    if edge_ml >= EDGE_VALUE:
        return "✅ VALUE"
    return "🛑 PASS"


def build_v2_action_table(rows: List[dict]) -> pd.DataFrame:
    """
    Tableau stable : MATCH | DATE | MON PRONO | POURQUOI ? | CONFIANCE | EDGE.
    DATE et JOUR sont calculés à l'affichage (date du match, pas figés en base).
    """
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        match_name = (r.get("match_name") or "").strip()
        game_date_str = (str(r.get("start_time") or "")[:10]) or ""
        jour = _day_label(game_date_str)
        if " vs " in match_name:
            home_name, away_name = match_name.split(" vs ", 1)[0].strip(), match_name.split(" vs ", 1)[1].strip()
        else:
            home_name, away_name = match_name, ""

        proj_h = float(r.get("projected_score_home") or 0)
        proj_a = float(r.get("projected_score_away") or 0)
        spread = proj_h - proj_a
        edge_ml = float(r.get("edge_ml") or 0)
        confidence_score = float(r.get("confidence_score") or 50)
        reasoning = (r.get("reasoning_text") or "").strip() or "—"

        if edge_ml < 0:
            mon_prono = "PASSER"
        elif spread >= 0:
            mon_prono = f"{home_name} {spread:+.1f}"
        else:
            mon_prono = f"{away_name} {-spread:+.1f}"

        confiance = _confiance_badge(edge_ml, confidence_score)
        edge_str = f"{edge_ml:+.1f}%"
        date_display = f"{game_date_str} · {jour}" if game_date_str else jour

        out.append({
            "MATCH": match_name,
            "DATE": date_display,
            "MON PRONO": mon_prono,
            "POURQUOI ?": reasoning,
            "CONFIANCE": confiance,
            "EDGE": edge_str,
            "_edge_num": edge_ml,
        })
    out.sort(key=lambda x: x["_edge_num"], reverse=True)
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "_edge_num"} for row in out])
    return df


def style_v2_table(df: pd.DataFrame) -> Any:
    """Badge couleur CONFIANCE ; EDGE vert ≥ 5%, rouge < 0. Colonnes : MATCH, DATE, MON PRONO, POURQUOI ?, CONFIANCE, EDGE."""
    import re
    ncols = len(df.columns)
    def _color_row(row):
        edge_str = str(row.get("EDGE", ""))
        e = 0.0
        m = re.search(r"([+-]?[\d.]+)", edge_str)
        if m:
            try:
                e = float(m.group(1))
            except ValueError:
                pass
        confiance = row.get("CONFIANCE", "")
        styles = [""] * ncols
        idx_confiance = list(row.index).index("CONFIANCE") if "CONFIANCE" in row.index else ncols - 2
        idx_edge = list(row.index).index("EDGE") if "EDGE" in row.index else ncols - 1
        if "PASS" in confiance:
            styles[idx_confiance] = "background-color: #374151; color: #9ca3af"
        if e < 0:
            styles[idx_edge] = "background-color: rgba(239,68,68,0.25); color: #b91c1c; font-weight: 600"
        elif e >= EDGE_VALUE:
            styles[idx_edge] = "background-color: rgba(34,197,94,0.3); color: #166534; font-weight: 600"
        return styles

    return df.style.apply(_color_row, axis=1)


# -----------------------------------------------------------------------------
# Backtest Time Machine (05_true_backtest) — lecture du JSON uniquement
# -----------------------------------------------------------------------------


def _load_true_backtest_results() -> Optional[Dict[str, Any]]:
    """Charge backtest_results.json généré par 05_true_backtest.py (sans lookahead)."""
    if not BACKTEST_RESULTS_PATH.exists():
        return None
    try:
        with open(BACKTEST_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Sniper — Terminal", layout="wide", page_icon="🎯", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
    .stApp { background: #0f172a; }
    .header-bar { display: flex; gap: 1rem; margin-bottom: 1rem; font-family: monospace; }
    .badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; }
    .badge-db { background: #166534; color: white; }
    .badge-offline { background: #1e40af; color: white; }
    .roi-big { font-size: 2rem; font-weight: 700; color: #22c55e; }
    .roi-big.negative { color: #ef4444; }
    </style>
    """, unsafe_allow_html=True)

    db_ok = _get_supabase() is not None
    today_str = _today_paris().isoformat()
    st.markdown(
        f'<div class="header-bar">'
        f'<span class="badge badge-db">{"🟢 DB" if db_ok else "🔴 DB Off"}</span>'
        f'<span class="badge badge-offline">📦 daily_projections_v2 (lecture seule)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.title("🎯 Terminal de Prise de Décision")
    st.caption("Tableau stable — Les pronos sont figés par 03_predict_daily.py (aucun recalcul au rechargement).")

    if not db_ok:
        st.error("❌ Supabase indisponible. Vérifiez .env")
        st.stop()

    all_rows = _fetch_daily_projections_v2()
    rows = [r for r in all_rows if (r.get("date_prediction") or "")[:10] == today_str]

    # Ne garder que les matchs à venir (date du match >= aujourd'hui)
    def _game_date(r: dict) -> str:
        t = r.get("start_time") or ""
        return (str(t)[:10] if t else "") or ""

    rows = [r for r in rows if _game_date(r) >= today_str]

    # Si rien pour aujourd'hui : afficher la dernière batch disponible (ex. hier)
    if not rows and all_rows:
        dates = sorted(set((r.get("date_prediction") or "")[:10] for r in all_rows if r.get("date_prediction")), reverse=True)
        if dates:
            last_date = dates[0]
            rows = [r for r in all_rows if (r.get("date_prediction") or "")[:10] == last_date]
            rows = [r for r in rows if _game_date(r) >= today_str]
            st.info(
                f"📅 Aucune projection pour **aujourd'hui** ({today_str}). "
                f"Affichage de la dernière batch : **{last_date}** (matchs à venir uniquement). "
                "Le workflow GitHub (08h00) lance 01 → 02 → 03 ; si besoin, déclenchez-le à la main : **Actions → Sniper Bot Daily → Run workflow**."
            )

    if not rows:
        st.warning(f"Aucune projection en base pour la date du jour ({today_str}).")
        st.info(
            "Le **workflow GitHub** (08h00 Paris) lance **01 → 02 → 03** et remplit `daily_projections_v2`. "
            "Pour lancer à la main : **Actions** → **Sniper Bot Daily** → **Run workflow** (ou en local : `python run_pipeline.py`)."
        )
        with st.expander("🔧 Le job a tourné mais toujours rien ?"):
            st.markdown("""
1. **Table `daily_projections_v2`** : exécuter une fois dans Supabase (SQL) le fichier `schema_migration_daily_projections_v2.sql`.
2. **Logs du job** : dans l’onglet Actions, ouvrir la dernière exécution et vérifier l’étape **03_predict_daily** — le script affiche le nombre de matchs trouvés (aujourd’hui + J+1 à J+3). Si « 0 match », l’étape 01 n’a peut‑être pas inséré de matchs à venir.
3. **Timezone** : le workflow utilise `TZ=Europe/Paris` pour que 01 et 03 utilisent la même date du jour.
            """)
        return

    # --- Tableau principal : MATCH | MON PRONO | POURQUOI ? | CONFIANCE | EDGE ---
    st.subheader("📋 L'Action — Tri par Edge décroissant")
    df_action = build_v2_action_table(rows)
    if not df_action.empty:
        st.dataframe(
            style_v2_table(df_action),
            use_container_width=True,
            hide_index=True,
            column_config={
                "MATCH": st.column_config.TextColumn("MATCH", width="large"),
                "DATE": st.column_config.TextColumn("DATE", help="Date du match · Aujourd'hui / Demain / J+2 (calculé à l'affichage)"),
                "MON PRONO": st.column_config.TextColumn("MON PRONO", width="medium"),
                "POURQUOI ?": st.column_config.TextColumn("POURQUOI ?", width="large"),
                "CONFIANCE": st.column_config.TextColumn("CONFIANCE"),
                "EDGE": st.column_config.TextColumn("EDGE"),
            },
        )
        st.caption("Colonne DATE = date du match + jour (Aujourd'hui / Demain / J+2) calculé à l'affichage. Seuls les matchs à venir sont affichés.")

    # --- Over/Under (données figées v2) ---
    rows_ou = [r for r in rows if r.get("total_points_projected") is not None]
    if rows_ou:
        st.subheader("📊 Over/Under — Total projeté (figé)")
        ou_data = []
        for r in rows_ou:
            match_name = (r.get("match_name") or "").strip()
            total_proj = r.get("total_points_projected")
            line_book = r.get("bookmaker_line_total")
            ou_data.append({
                "MATCH": match_name,
                "LIGNE BOOK": f"{line_book:.1f}" if line_book is not None else "—",
                "MA PROJECTION": f"{total_proj:.1f}" if total_proj is not None else "—",
            })
        if ou_data:
            st.dataframe(pd.DataFrame(ou_data), use_container_width=True, hide_index=True)

    # --- Tabs Deep Dive & Backtest ---
    tab_deep, tab_backtest = st.tabs(["🔬 Deep Dive", "📈 Backtest"])

    with tab_deep:
        st.subheader("Détail par match (données figées)")
        opts = [(r.get("match_name") or "").strip() for r in rows if r.get("match_name")]
        opts = list(dict.fromkeys(opts))
        if not opts:
            st.info("Aucun match.")
        else:
            if "deep_match_sel" not in st.session_state:
                st.session_state["deep_match_sel"] = opts[0]
            idx = opts.index(st.session_state["deep_match_sel"]) if st.session_state["deep_match_sel"] in opts else 0
            sel = st.selectbox("Choisir un match", opts, index=idx, key="deep_select")
            st.session_state["deep_match_sel"] = sel
            row = next((r for r in rows if (r.get("match_name") or "").strip() == sel), None)
            if row:
                st.markdown("**POURQUOI ?**")
                st.write((row.get("reasoning_text") or "—"))
                c1, c2, c3 = st.columns(3)
                with c1:
                    proba_pct = float(row.get("proba_ml_calibrated") or 0.5) * 100
                    st.metric("Proba ML (calibrée)", f"{proba_pct:.0f}%", "")
                with c2:
                    edge_val = float(row.get("edge_ml") or 0)
                    st.metric("Edge", f"{edge_val:+.1f}%", "")
                with c3:
                    total_proj = row.get("total_points_projected")
                    st.metric("Total projeté", f"{total_proj:.1f}" if total_proj is not None else "—", "pts")
                st.metric("Confiance (0-100)", f"{float(row.get('confidence_score') or 0):.0f}", "")

    with tab_backtest:
        st.subheader("Backtest Time Machine — Sans lookahead")
        backtest_data = _load_true_backtest_results()
        if backtest_data and backtest_data.get("rows"):
            total_profit = backtest_data.get("total_profit", 0)
            n_bets = backtest_data.get("n_bets", 0)
            days = backtest_data.get("days", 10)
            roi = (total_profit / n_bets * 100.0) if n_bets > 0 else 0.0
            st.caption(f"Si tu avais utilisé l'outil sur les {days} derniers jours (modèles ré-entraînés chaque jour sans données futures) :")
            st.metric("Profit total (unités)", f"{total_profit:.2f}", f"ROI moyen {roi:.1f}% · {n_bets} paris")
            df_bt = pd.DataFrame(backtest_data["rows"])
            df_bt_display = df_bt.rename(columns={
                "date": "Date",
                "match": "Match",
                "pari": "Pari",
                "cote": "Cote",
                "resultat": "Résultat",
                "profit": "Profit",
            })
            st.dataframe(df_bt_display, use_container_width=True, hide_index=True, column_config={
                "Date": st.column_config.TextColumn("Date"),
                "Match": st.column_config.TextColumn("Match", width="large"),
                "Pari": st.column_config.TextColumn("Pari"),
                "Cote": st.column_config.TextColumn("Cote"),
                "Résultat": st.column_config.TextColumn("Résultat"),
                "Profit": st.column_config.NumberColumn("Profit", format="%.2f"),
            })
            if len(df_bt) > 0 and "profit" in df_bt.columns:
                cumul = df_bt["profit"].cumsum()
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_bt["date"], y=cumul, mode="lines+markers", name="Cumul (u)"))
                    fig.update_layout(xaxis_title="Date", yaxis_title="Gains cumulés", template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
            st.caption("Généré par 05_true_backtest.py — Relancer pour mettre à jour.")
        else:
            st.info("Aucun backtest Time Machine. Lancez : python 05_true_backtest.py — Puis rafraîchissez cette page.")


if __name__ == "__main__":
    main()
