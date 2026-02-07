#!/usr/bin/env python3
"""
04_app_dashboard.py — L'Interface (Architecture ETL Sniper V1 Pro)
====================================================================
Aucun calcul ni appel API : lit uniquement daily_projections (Supabase) et affiche
les tableaux Streamlit (couleurs, badges, styles).

Pipeline : 01_ingest_data → 02_train_models → 03_predict_daily → 04_app_dashboard (lecture).

Usage:
  streamlit run 04_app_dashboard.py
"""

from pathlib import Path
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from database import get_client

# Style (aligné app_sniper_v27)
COLOR_MAX_BET = "#22c55e"
COLOR_MAX_BET_BG = "#14532d"
COLOR_VALUE = "#22c55e"
COLOR_PASS = "#6b7280"
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_META_TOTALS_PATH = SCRIPT_DIR / "features_meta_totals.json"

# Seuils UX Moneyline
EDGE_MIN_BET = 5.0       # Edge > 5% pour parier
EDGE_VALUE_HIGH = 10.0   # Edge > 10% = VALUE (outsider)
PROBA_SECURITE = 0.75    # Proba > 75% = SÉCURITÉ
FIABILITE_SECURITE = 70  # Fiabilité min pour SÉCURITÉ
KELLY_CAP = 0.05         # Max 5% bankroll par pari


def _get_supabase():
    return get_client()


@st.cache_data(ttl=60)
def _fetch_daily_projections() -> List[dict]:
    """SELECT * FROM daily_projections (ordre edge décroissant)."""
    supabase = _get_supabase()
    if not supabase:
        return []
    try:
        r = supabase.table("daily_projections").select("*").order("edge_percent", desc=True).execute()
        return r.data or []
    except Exception:
        return []


def _get_team_name(supabase, team_id: int) -> str:
    if not supabase:
        return f"Équipe {team_id}"
    try:
        r = supabase.table("teams_metadata").select("nom_equipe, name").eq("team_id", team_id).limit(1).execute()
        if r.data:
            nom = (r.data[0].get("nom_equipe") or r.data[0].get("name") or "").strip()
            if nom:
                return nom
    except Exception:
        pass
    return f"Équipe {team_id}"


def _last_n_matches_with_box(supabase, team_id: int, n: int = 3) -> List[dict]:
    """Derniers n matchs avec box_scores + scores (Supabase uniquement)."""
    if not supabase:
        return []
    try:
        r = (
            supabase.table("box_scores")
            .select("game_id, opponent_id, is_home, date, pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate")
            .eq("team_id", team_id)
            .order("date", desc=True)
            .limit(n)
            .execute()
        )
        rows = r.data or []
        if not rows:
            return []
        game_ids = list({x["game_id"] for x in rows})
        games_by_id = {}
        for gid in game_ids:
            gr = supabase.table("games_history").select("home_id, away_id, home_score, away_score").eq("game_id", gid).limit(1).execute()
            if gr.data:
                games_by_id[gid] = gr.data[0]
        out = []
        for b in rows:
            g = games_by_id.get(b["game_id"], {})
            opp_id = b.get("opponent_id")
            is_home = b.get("is_home", True)
            score_us = g.get("home_score") if is_home else g.get("away_score")
            score_opp = g.get("away_score") if is_home else g.get("home_score")
            opp_name = _get_team_name(supabase, opp_id) if opp_id else "?"
            res = "V" if (score_us is not None and score_opp is not None and score_us > score_opp) else ("D" if (score_us is not None and score_opp is not None) else "?")
            score_str = f"{score_us} — {score_opp}" if (score_us is not None and score_opp is not None) else "—"
            out.append({
                "date": (b.get("date") or "")[:10],
                "opponent": opp_name,
                "score": score_str,
                "result": res,
                "pace": b.get("pace"),
                "off_rtg": b.get("off_rtg"),
                "def_rtg": b.get("def_rtg"),
                "efg_pct": b.get("efg_pct"),
                "orb_pct": b.get("orb_pct"),
                "tov_pct": b.get("tov_pct"),
                "ft_rate": b.get("ft_rate"),
            })
        return out
    except Exception:
        return []


def build_df_from_projections() -> pd.DataFrame:
    """Construit le DataFrame affichage à partir de daily_projections."""
    rows = _fetch_daily_projections()
    if not rows:
        return pd.DataFrame()

    out = []
    for r in rows:
        oh, oa = r.get("odds_home"), r.get("odds_away")
        cotes_str = f"{oh:.2f} | {oa:.2f}" if (oh is not None and oa is not None) else "—"
        proba_book_h = (100.0 / oh) if oh else None
        proba_book_a = (100.0 / oa) if oa else None
        proba_book_str = f"{proba_book_h:.0f}% | {proba_book_a:.0f}%" if (proba_book_h and proba_book_a) else "—"
        prob_cal = r.get("proba_calibree")
        prob_cal_str = f"{prob_cal*100:.1f}%" if prob_cal is not None else "—"
        edge = r.get("edge_percent") or 0
        edge_str = f"{edge:+.0f}%" if edge != 0 else "—"
        line_book = r.get("line_bookmaker")
        ligne_book_str = f"{line_book:.1f}" if line_book is not None else "En attente"
        diff = r.get("diff_total")
        diff_str = f"{diff:+.1f} pts" if diff is not None else "—"
        pred_total = r.get("predicted_total")
        proj_sniper = round(pred_total or 165.0, 1)  # pas 150

        out.append({
            "_game_id": r.get("game_id"),
            "_date": str(r.get("date", ""))[:10] if r.get("date") else None,
            "_league_id": r.get("league_id"),
            "_season": r.get("season"),
            "_home_id": r.get("home_id"),
            "_away_id": r.get("away_id"),
            "_home_name": (r.get("match_name") or "").split(" vs ")[0] if r.get("match_name") else "",
            "_away_name": (r.get("match_name") or "").split(" vs ")[1] if r.get("match_name") and " vs " in (r.get("match_name") or "") else "",
            "Jour": r.get("jour", "—"),
            "MATCH": r.get("match_name", ""),
            "Match": r.get("match_name", ""),
            "COTES (H/A)": cotes_str,
            "PROBA BOOK (%)": proba_book_str,
            "PROBA SNIPER (%)": prob_cal_str,
            "EDGE (%)": edge_str,
            "LE PARI": r.get("le_pari", ""),
            "Cerveau utilisé": r.get("brain_used", ""),
            "Proba ML": prob_cal_str,
            "Proba calibrée": prob_cal_str,
            "Style de Match": r.get("style_match", "—"),
            "🚨 ALERTE TRAPPE": r.get("alerte_trappe", "—"),
            "🎯 PARI OUTSIDER": r.get("pari_outsider", "—"),
            "Message de Contexte": r.get("message_contexte", ""),
            "Confiance": r.get("confiance_label", ""),
            "Fiabilité": f"{r.get('fiabilite', 50):.0f}%",
            "_edge": edge,
            "_prob_home": prob_cal or 0.5,
            "odds_home": r.get("odds_home"),
            "odds_away": r.get("odds_away"),
            "_ml_total_predicted": pred_total,
            "LIGNE BOOK": ligne_book_str,
            "PROJETÉ SNIPER": proj_sniper,
            "DIFF": diff_str,
            "PARI TOTAL": r.get("pari_total", ""),
            "CONFIANCE": r.get("confiance_ou", "—"),
            "_diff_total": r.get("diff_total"),
        })

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("_edge", ascending=False, ignore_index=True)


def _style_confiance(row):
    c = row.get("Confiance", "")
    trap = row.get("🚨 ALERTE TRAPPE", "")
    brain = row.get("Cerveau utilisé", "")
    if "SNIPER" in str(c):
        return ["background-color: #7c3aed; color: white; font-weight: 700"] * len(row)
    if "MAX" in str(c):
        return [f"background-color: {COLOR_MAX_BET_BG}; color: {COLOR_MAX_BET}; font-weight: 700"] * len(row)
    if "VALUE" in str(c):
        return [f"background-color: rgba(34,197,94,0.2); color: {COLOR_VALUE}"] * len(row)
    if "Chasseur" in str(brain):
        return ["background-color: rgba(234,88,12,0.2); color: #ea580c; font-weight: 600"] * len(row)
    if "Trap" in str(trap):
        return ["background-color: rgba(249,115,22,0.25); color: #ea580c"] * len(row)
    return [f"color: {COLOR_PASS}"] * len(row)


def _style_ou_confiance(row):
    c = row.get("CONFIANCE", "")
    if "MAX" in str(c):
        return [f"background-color: {COLOR_MAX_BET_BG}; color: {COLOR_MAX_BET}; font-weight: 700"] * len(row)
    if "VALUE" in str(c):
        return [f"background-color: rgba(34,197,94,0.2); color: {COLOR_VALUE}"] * len(row)
    return [f"color: {COLOR_PASS}"] * len(row)


def _kelly_fraction(prob: float, odd: float) -> float:
    """Kelly Criterion : (p*b - 1) / (b - 1). Cap à KELLY_CAP."""
    if odd is None or odd <= 1:
        return 0.0
    kelly = (prob * odd - 1.0) / (odd - 1.0)
    return max(0.0, min(KELLY_CAP, kelly))


def build_moneyline_simple_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tableau simplifié « Est-ce que je parie ou pas ? »
    Colonnes : MATCH, COTE VALUE, SIGNAL, MISE CONSEILLÉE (% bankroll).
    """
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        prob = float(r.get("_prob_home") or 0.5)
        edge = float(r.get("_edge") or 0)
        fiabilite = float(str(r.get("Fiabilité", "50")).replace("%", ""))
        oh = r.get("odds_home")
        oa = r.get("odds_away")
        odds_home = float(oh) if oh is not None and isinstance(oh, (int, float)) else None
        odds_away = float(oa) if oa is not None and isinstance(oa, (int, float)) else None
        if odds_home is None and "COTES (H/A)" in r:
            raw = str(r.get("COTES (H/A)", ""))
            if " | " in raw:
                try:
                    a, b = raw.split(" | ")
                    odds_home, odds_away = float(a.strip()), float(b.strip())
                except ValueError:
                    pass
        if odds_home is None:
            odds_home = odds_away = None
        edge_home = (prob * (odds_home or 0) - 1.0) * 100.0 if odds_home else 0.0
        edge_away = ((1.0 - prob) * (odds_away or 0) - 1.0) * 100.0 if odds_away else 0.0
        has_value_home = edge_home >= EDGE_MIN_BET
        has_value_away = edge_away >= EDGE_MIN_BET
        if has_value_home and (edge_home >= edge_away or not has_value_away):
            cote_value = f"{odds_home:.2f}" if odds_home else "—"
            kelly = _kelly_fraction(prob, odds_home)
            bet_side = "home"
            edge_bet = edge_home
        elif has_value_away:
            cote_value = f"{odds_away:.2f}" if odds_away else "—"
            kelly = _kelly_fraction(1.0 - prob, odds_away)
            bet_side = "away"
            edge_bet = edge_away
        else:
            cote_value = "—"
            kelly = 0.0
            bet_side = None
            edge_bet = 0.0
        if bet_side and edge_bet >= EDGE_VALUE_HIGH:
            signal = "💰 VALUE"
        elif bet_side and prob >= PROBA_SECURITE and fiabilite >= FIABILITE_SECURITE:
            signal = "🛡️ SÉCURITÉ"
        elif bet_side:
            signal = "💰 VALUE" if edge_bet >= EDGE_VALUE_HIGH else "✅ VALUE"
        else:
            signal = "⚠️ PASSER"
        mise = f"{kelly * 100:.1f}%" if kelly > 0 else "—"
        match_str = r.get("Match", "") or r.get("MATCH", "")
        jour = r.get("Jour", "—")
        match_affichage = f"{match_str} · {jour}" if jour and jour != "—" else match_str
        rows.append({
            "MATCH": match_affichage,
            "COTE VALUE": cote_value,
            "SIGNAL": signal,
            "MISE CONSEILLÉE": mise,
        })
    return pd.DataFrame(rows)


def _style_signal(row):
    s = row.get("SIGNAL", "")
    if "SÉCURITÉ" in str(s):
        return ["background-color: rgba(34,197,94,0.25); color: #16a34a; font-weight: 600"] * len(row)
    if "VALUE" in str(s) and "PASSER" not in str(s):
        return ["background-color: rgba(34,197,94,0.2); color: " + COLOR_VALUE] * len(row)
    return [f"color: {COLOR_PASS}"] * len(row)


def backtest_profitability(days_list: List[int] = (1, 7, 30)) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Backtest sur matchs terminés (games_history avec scores + cotes).
    Pour chaque période (1j, 7j, 30j) : mises simulées (edge > 5%), gains, ROI.
    Retourne (tableau Période | Mises | Gains | ROI %, série cumulée des gains pour graphique).
    """
    supabase = _get_supabase()
    if not supabase:
        return pd.DataFrame(), None
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("predict_daily", SCRIPT_DIR / "03_predict_daily.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        load_models = getattr(mod, "load_models")
        get_ml_prediction = getattr(mod, "get_ml_prediction")
    except Exception:
        return pd.DataFrame(), None
    models = load_models()
    if not models:
        return pd.DataFrame(), None
    today = date.today().isoformat()
    max_days = max(days_list) if days_list else 30
    start_all = (date.today() - timedelta(days=max_days)).isoformat()
    try:
        r = (
            supabase.table("games_history")
            .select("game_id, date, league_id, season, home_id, away_id, home_score, away_score, home_odd, away_odd")
            .not_.is_("home_score", "null")
            .gte("date", start_all)
            .lte("date", today)
            .order("date", desc=False)
            .execute()
        )
    except Exception:
        r = type("R", (), {"data": []})()
    games = r.data or []
    # Liste des profits par (date, période) pour table + cumul
    cumul = 0.0
    cumulative = []
    profits_by_date = []
    for g in games:
        oh, oa = g.get("home_odd"), g.get("away_odd")
        if oh is None or oa is None:
            continue
        game_date = str(g.get("date", ""))[:10]
        pred = get_ml_prediction(models, g["home_id"], g["away_id"], game_date, g.get("league_id"), g.get("season") or "")
        if pred is None:
            continue
        prob = pred.get("prob_home_calibrated", pred.get("prob_home", 0.5))
        edge_h = (prob * oh - 1.0) * 100.0
        edge_a = ((1.0 - prob) * oa - 1.0) * 100.0
        if edge_h < EDGE_MIN_BET and edge_a < EDGE_MIN_BET:
            continue
        stake = 1.0
        home_won = (g.get("home_score") or 0) > (g.get("away_score") or 0)
        if edge_h >= edge_a and edge_h >= EDGE_MIN_BET:
            profit = (oh * stake - stake) if home_won else -stake
        else:
            profit = (oa * stake - stake) if not home_won else -stake
        cumul += profit
        cumulative.append({"date": game_date, "cumul": cumul})
        profits_by_date.append({"date": game_date, "profit": profit})
    # Tableau par période
    all_results = []
    for days in days_list:
        start = (date.today() - timedelta(days=days)).isoformat()
        sub = [x for x in profits_by_date if start <= x["date"] <= today]
        mises = len(sub)
        gains = sum(x["profit"] for x in sub)
        roi = (gains / mises * 100.0) if mises > 0 else 0.0
        all_results.append({
            "Période": f"{days} jour(s)",
            "Mises": int(mises),
            "Gains": round(gains, 2),
            "ROI %": round(roi, 1),
        })
    df_bt = pd.DataFrame(all_results)
    df_cumul = pd.DataFrame(cumulative) if cumulative else None
    return df_bt, df_cumul


def main() -> None:
    st.set_page_config(page_title="Sniper V1 Pro", layout="wide", page_icon="🎯", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
    .stApp { background: #0f172a; }
    .bloomberg-header { display: flex; gap: 1rem; margin-bottom: 1rem; font-family: monospace; }
    .status { padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; }
    .status-db { background: #166534; color: white; }
    .status-offline { background: #1e40af; color: white; }
    </style>
    """, unsafe_allow_html=True)

    db_ok = _get_supabase() is not None
    st.markdown(
        f'<div class="bloomberg-header">'
        f'<span class="status status-db">{"🟢 DB: Connected" if db_ok else "🔴 DB: Off"}</span>'
        f'<span class="status status-offline">📦 Offline-First · Lecture daily_projections</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.title("🎯 Sniper Scanner V1 Pro — Dashboard")
    st.caption("Pipeline ETL : 01_ingest_data → 02_train_models → 03_predict_daily. Aucun appel API ici.")

    if not db_ok:
        st.error("❌ Supabase indisponible. Vérifiez .env")
        st.stop()

    df = build_df_from_projections()
    if df.empty:
        st.warning("Aucune projection. Lancez 01_ingest_data, 02_train_models, puis 03_predict_daily.")
        return

    st.success(f"✅ {len(df)} match(s) (daily_projections)")

    tab_scanner, tab_rentabilite, tab_deep = st.tabs(["📊 Sniper Scanner", "📈 Rentabilité", "🔬 Deep Dive"])

    with tab_scanner:
        st.subheader("🎯 Vainqueur (Moneyline) — Est-ce que je parie ou pas ?")
        df_simple = build_moneyline_simple_df(df)
        if not df_simple.empty:
            st.dataframe(
                df_simple.style.apply(_style_signal, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MATCH": st.column_config.TextColumn("MATCH", width="large", help="Équipes · Jour"),
                    "COTE VALUE": st.column_config.TextColumn("COTE VALUE", help="Cote bookmaker si valeur détectée"),
                    "SIGNAL": st.column_config.TextColumn("SIGNAL", help="🛡️ SÉCURITÉ (Proba > 75%, fiable) · 💰 VALUE (Edge > 10%) · ⚠️ PASSER"),
                    "MISE CONSEILLÉE": st.column_config.TextColumn("MISE CONSEILLÉE", help="% bankroll (Kelly Criterion, plafond 5%)"),
                },
            )
        st.caption("SÉCURITÉ = proba > 75% et historique fiable · VALUE = edge > 5% · Mise = Kelly (max 5%).")

        st.subheader("Détail (colonnes complètes)")
        display_cols = [
            "MATCH", "COTES (H/A)", "PROBA SNIPER (%)", "EDGE (%)",
            "LE PARI", "Confiance", "Cerveau utilisé", "Style de Match",
            "🚨 ALERTE TRAPPE", "🎯 PARI OUTSIDER", "Fiabilité",
        ]
        display_df = df[[c for c in display_cols if c in df.columns]]
        if not display_df.empty:
            st.dataframe(
                display_df.style.apply(_style_confiance, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("📊 Total Points (Over/Under)")
        mae_total = None
        if FEATURES_META_TOTALS_PATH.exists():
            try:
                import json
                with open(FEATURES_META_TOTALS_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                mae_total = meta.get("mae_total")
            except Exception:
                pass
        if mae_total is not None:
            st.caption(f"Total : model_totals.pkl (MAE **{mae_total:.1f}** pts).")
        else:
            st.caption("Total : model_totals.pkl (pas de constante 150).")

        totals_cols = ["Match", "LIGNE BOOK", "PROJETÉ SNIPER", "DIFF", "Style de Match", "PARI TOTAL", "CONFIANCE"]
        df_totals = df[[c for c in totals_cols if c in df.columns]].copy()
        if "Style de Match" in df_totals.columns:
            df_totals = df_totals.rename(columns={"Style de Match": "STYLE"})
        if not df_totals.empty:
            want = ["Match", "LIGNE BOOK", "PROJETÉ SNIPER", "DIFF", "STYLE", "PARI TOTAL", "CONFIANCE"]
            display_totals = df_totals[[c for c in want if c in df_totals.columns]]
            st.dataframe(
                display_totals.style.apply(_style_ou_confiance, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Match": st.column_config.TextColumn("MATCH", width="large"),
                    "LIGNE BOOK": st.column_config.TextColumn("LIGNE BOOK"),
                    "PROJETÉ SNIPER": st.column_config.NumberColumn("PROJETÉ SNIPER", format="%.1f"),
                    "DIFF": st.column_config.TextColumn("DIFF"),
                    "STYLE": st.column_config.TextColumn("STYLE"),
                    "PARI TOTAL": st.column_config.TextColumn("🎯 PARI TOTAL"),
                    "CONFIANCE": st.column_config.TextColumn("CONFIANCE"),
                },
            )
        else:
            st.info("Aucune donnée Over/Under.")

    with tab_rentabilite:
        st.subheader("📈 Rentabilité (Backtest)")
        st.caption("Simulation : pari si Edge > 5%, mise 1 unité. Données : matchs terminés avec cotes (games_history).")
        days_list = [1, 7, 30]
        df_bt, df_cumul = backtest_profitability(days_list)
        if df_bt is not None and not df_bt.empty:
            st.dataframe(df_bt, use_container_width=True, hide_index=True, column_config={
                "Période": st.column_config.TextColumn("Période"),
                "Mises": st.column_config.NumberColumn("Mises", format="%d"),
                "Gains": st.column_config.NumberColumn("Gains", format="%.2f"),
                "ROI %": st.column_config.NumberColumn("ROI %", format="%.1f"),
            })
            if df_cumul is not None and not df_cumul.empty:
                st.subheader("Gains cumulés")
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_cumul["date"], y=df_cumul["cumul"], mode="lines+markers", name="Cumul (unités)"))
                fig.update_layout(xaxis_title="Date", yaxis_title="Gains cumulés", template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Impossible de lancer le backtest (modèles absents ou pas de matchs avec cotes). Lancez 02_train_models et 01_ingest_data avec cotes.")

    with tab_deep:
        st.subheader("Analyse Détaillée")
        opts = df["Match"].tolist()
        if not opts:
            st.info("Aucun match.")
        else:
            sel = st.selectbox("Match", opts, key="deep_match")
            if sel:
                row = df[df["Match"] == sel].iloc[0]
                home_id = int(row["_home_id"])
                away_id = int(row["_away_id"])
                home_name = str(row["_home_name"])
                away_name = str(row["_away_name"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    pred = row.get("_ml_total_predicted")
                    st.metric("Total projeté", f"{pred:.1f} pts" if pred is not None else "—", "model_totals")
                with col2:
                    st.metric("Proba Domicile", f"{row['_prob_home']*100:.1f}%", "calibrée")
                with col3:
                    st.metric("Edge", f"{row.get('_edge', 0):+.1f}%", "daily_projections")

                msg = row.get("Message de Contexte", "")
                if msg:
                    st.markdown("**Contexte**")
                    st.caption(msg)

                st.markdown("**3 derniers matchs — Box Scores**")
                supabase = _get_supabase()
                col_home, col_away = st.columns(2)
                with col_home:
                    st.caption(f"🏠 {home_name}")
                    matches_h = _last_n_matches_with_box(supabase, home_id, 3)
                    if matches_h:
                        df_h = pd.DataFrame(matches_h)
                        df_h = df_h.rename(columns={
                            "date": "Date", "opponent": "Adversaire", "score": "Score", "result": "R",
                            "pace": "Pace", "off_rtg": "Off", "def_rtg": "Def",
                            "efg_pct": "eFG%", "orb_pct": "ORB%", "tov_pct": "TOV%", "ft_rate": "FT Rate",
                        })
                        st.dataframe(df_h, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun box score récent.")
                with col_away:
                    st.caption(f"✈️ {away_name}")
                    matches_a = _last_n_matches_with_box(supabase, away_id, 3)
                    if matches_a:
                        df_a = pd.DataFrame(matches_a)
                        df_a = df_a.rename(columns={
                            "date": "Date", "opponent": "Adversaire", "score": "Score", "result": "R",
                            "pace": "Pace", "off_rtg": "Off", "def_rtg": "Def",
                            "efg_pct": "eFG%", "orb_pct": "ORB%", "tov_pct": "TOV%", "ft_rate": "FT Rate",
                        })
                        st.dataframe(df_a, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun box score récent.")


if __name__ == "__main__":
    main()
