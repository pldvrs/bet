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
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import streamlit as st

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

from database import get_client

SCRIPT_DIR = Path(__file__).resolve().parent
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


def _fetch_game_teams(supabase, game_id: int) -> Optional[dict]:
    """Récupère home_id, away_id, league_id, season, date pour un game_id (games_history)."""
    if not supabase or not game_id:
        return None
    try:
        r = (
            supabase.table("games_history")
            .select("home_id, away_id, league_id, season, date")
            .eq("game_id", game_id)
            .limit(1)
            .execute()
        )
        data = r.data or []
        return data[0] if data else None
    except Exception:
        return None


def _fetch_standings_before_date(
    supabase, league_id: int, season: str, before_date: str, limit: int = 500
) -> dict:
    """
    Classement saison (W-L) par équipe pour league_id + season, matchs avant before_date.
    Retourne {team_id: {"wins": int, "losses": int, "played": int, "rank": int}}.
    """
    if not supabase or not before_date or len(before_date) < 10:
        return {}
    try:
        r = (
            supabase.table("games_history")
            .select("home_id, away_id, home_score, away_score")
            .eq("league_id", league_id)
            .eq("season", season)
            .lt("date", before_date[:10])
            .not_.is_("home_score", "null")
            .not_.is_("away_score", "null")
            .limit(limit)
            .execute()
        )
        rows = r.data or []
    except Exception:
        return {}
    wins: dict = {}
    for g in rows:
        h, a = g.get("home_id"), g.get("away_id")
        sh, sa = g.get("home_score"), g.get("away_score")
        if h is None or a is None or sh is None or sa is None:
            continue
        wins.setdefault(h, {"wins": 0, "losses": 0})
        wins.setdefault(a, {"wins": 0, "losses": 0})
        if sh > sa:
            wins[h]["wins"] += 1
            wins[a]["losses"] += 1
        else:
            wins[a]["wins"] += 1
            wins[h]["losses"] += 1
    for v in wins.values():
        v["played"] = v["wins"] + v["losses"]
    sorted_teams = sorted(wins.keys(), key=lambda tid: wins[tid]["wins"], reverse=True)
    for rank, tid in enumerate(sorted_teams, 1):
        wins[tid]["rank"] = rank
    return wins


def _fetch_team_box_stats_last_n(
    supabase, team_id: int, before_date: str, n: int = 10
) -> Optional[dict]:
    """
    Moyennes des stats (pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate, three_rate)
    sur les n derniers matchs de l'équipe avant before_date (box_scores).
    """
    if not supabase or not team_id or not before_date or len(before_date) < 10:
        return None
    try:
        r = (
            supabase.table("box_scores")
            .select("pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate, three_rate")
            .eq("team_id", team_id)
            .lt("date", before_date[:10])
            .order("date", desc=True)
            .limit(n)
            .execute()
        )
        rows = r.data or []
    except Exception:
        return None
    if not rows:
        return None
    cols = ["pace", "off_rtg", "def_rtg", "efg_pct", "orb_pct", "tov_pct", "ft_rate", "three_rate"]
    out = {}
    for c in cols:
        vals = [float(x[c]) for x in rows if x.get(c) is not None]
        out[c] = sum(vals) / len(vals) if vals else None
    return out


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

        edge_ml = float(r.get("edge_ml") or 0)
        confidence_score = float(r.get("confidence_score") or 50)
        reasoning = (r.get("reasoning_text") or "").strip() or "—"
        # Prono ML figé en base (aligné Deep Dive) ; sinon fallback spread
        mon_prono = (r.get("le_pari") or "").strip()
        if not mon_prono:
            proj_h = float(r.get("projected_score_home") or 0)
            proj_a = float(r.get("projected_score_away") or 0)
            spread = proj_h - proj_a
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
# Backtest persistant — lecture Supabase (backtest_history)
# -----------------------------------------------------------------------------


@st.cache_data(ttl=120)
def _fetch_backtest_history(limit: int = 2000) -> List[dict]:
    supabase = _get_supabase()
    if not supabase:
        return []
    try:
        r = (
            supabase.table("backtest_history")
            .select("*")
            .order("match_date", desc=True)
            .limit(limit)
            .execute()
        )
        return r.data or []
    except Exception:
        return []


def _style_backtest_table(df: pd.DataFrame) -> Any:
    if df.empty:
        return df
    def _color_row(row):
        styles = [""] * len(row)
        if "Status" in row.index:
            idx = list(row.index).index("Status")
            status = str(row.get("Status", ""))
            if status == "WIN":
                styles[idx] = "background-color: rgba(34,197,94,0.25); color: #166534; font-weight: 600"
            elif status == "LOSS":
                styles[idx] = "background-color: rgba(239,68,68,0.25); color: #b91c1c; font-weight: 600"
            else:
                styles[idx] = "background-color: rgba(148,163,184,0.2); color: #cbd5f5; font-weight: 600"
        return styles
    return df.style.apply(_color_row, axis=1)


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
                supabase = _get_supabase()
                game_id = row.get("game_id")
                game_date_str = (str(row.get("start_time") or "")[:10]) or ""

                st.markdown("**POURQUOI ?**")
                st.write((row.get("reasoning_text") or "—"))

                # Métriques principales
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    proba_pct = float(row.get("proba_ml_calibrated") or 0.5) * 100
                    st.metric("Proba ML (calibrée)", f"{proba_pct:.0f}%", "")
                with c2:
                    edge_val = float(row.get("edge_ml") or 0)
                    st.metric("Edge", f"{edge_val:+.1f}%", "")
                with c3:
                    total_proj = row.get("total_points_projected")
                    st.metric("Total projeté", f"{total_proj:.1f}" if total_proj is not None else "—", "pts")
                with c4:
                    st.metric("Confiance (0-100)", f"{float(row.get('confidence_score') or 0):.0f}", "")

                # Score projeté (détail)
                st.markdown("**Score projeté**")
                ph = float(row.get("projected_score_home") or 0)
                pa = float(row.get("projected_score_away") or 0)
                parts = (row.get("match_name") or "").split(" vs ", 1)
                home_label = parts[0].strip() if len(parts) > 0 else "Domicile"
                away_label = parts[1].strip() if len(parts) > 1 else "Extérieur"
                col_h, col_vs, col_a = st.columns([2, 1, 2])
                with col_h:
                    st.metric(home_label, f"{ph:.1f}", "pts")
                with col_vs:
                    st.write("**—**")
                with col_a:
                    st.metric(away_label, f"{pa:.1f}", "pts")

                # Style de jeu (figé en v2)
                style_match = (row.get("style_match") or "").strip()
                if style_match:
                    st.markdown("**Style de match**")
                    st.caption(style_match)

                # Classement + Stats à jour (lecture Supabase)
                if supabase and game_id and game_date_str:
                    game_info = _fetch_game_teams(supabase, game_id)
                    if game_info:
                        home_id = game_info.get("home_id")
                        away_id = game_info.get("away_id")
                        league_id = game_info.get("league_id")
                        season = game_info.get("season") or ""

                        # Classement saison (W-L + rang)
                        standings = _fetch_standings_before_date(supabase, league_id, season, game_date_str) if league_id and season else {}
                        if standings:
                            st.markdown("**Classement saison (avant ce match)**")
                            col_h, col_a = st.columns(2)
                            with col_h:
                                if home_id and home_id in standings:
                                    s = standings[home_id]
                                    st.caption(f"**{_get_team_name(supabase, home_id)}** — {s.get('wins', 0)}V-{s.get('losses', 0)}D · Rang {s.get('rank', '—')}")
                                else:
                                    st.caption(f"**{home_label}** — —")
                            with col_a:
                                if away_id and away_id in standings:
                                    s = standings[away_id]
                                    st.caption(f"**{_get_team_name(supabase, away_id)}** — {s.get('wins', 0)}V-{s.get('losses', 0)}D · Rang {s.get('rank', '—')}")
                                else:
                                    st.caption(f"**{away_label}** — —")

                        # Stats à jour (derniers 10 matchs : Pace, OffRtg, DefRtg, eFG%, 3pt rate, etc.)
                        stats_h = _fetch_team_box_stats_last_n(supabase, home_id, game_date_str, 10) if home_id else None
                        stats_a = _fetch_team_box_stats_last_n(supabase, away_id, game_date_str, 10) if away_id else None
                        if stats_h or stats_a:
                            st.markdown("**Statistiques à jour (moy. 10 derniers matchs)**")
                            stat_cols = [
                                ("pace", "Pace", "Possessions / 48 min"),
                                ("off_rtg", "OffRtg", "Pts / 100 poss. (attaque)"),
                                ("def_rtg", "DefRtg", "Pts concédés / 100 poss. (défense)"),
                                ("efg_pct", "eFG%", "Effective FG% (2pt + 1.5×3pt)"),
                                ("three_rate", "3pt rate", "Part des tirs tentés à 3pts"),
                                ("orb_pct", "ORB%", "Rebonds off. récupérés"),
                                ("tov_pct", "TOV%", "Balles perdues / poss."),
                                ("ft_rate", "FT rate", "Tirs francs / tirs tentés"),
                            ]
                            table_data = []
                            for key, label, _ in stat_cols:
                                vh = (stats_h or {}).get(key)
                                va = (stats_a or {}).get(key)
                                if vh is None and va is None:
                                    continue
                                table_data.append({
                                    "Indicateur": label,
                                    home_label: f"{vh:.2f}" if vh is not None else "—",
                                    away_label: f"{va:.2f}" if va is not None else "—",
                                })
                            if table_data:
                                df_stats = pd.DataFrame(table_data)
                                st.dataframe(
                                    df_stats,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Indicateur": st.column_config.TextColumn("Indicateur", width="medium"),
                                        home_label: st.column_config.TextColumn(home_label),
                                        away_label: st.column_config.TextColumn(away_label),
                                    },
                                )
                            st.caption(
                                "Pace = possessions/48min · OffRtg = pts marqués/100 poss. · DefRtg = pts concédés/100 poss. · "
                                "eFG% = effective FG% · ORB% = rebonds off. · TOV% = turnovers · FT rate = ratio tirs francs."
                            )

    with tab_backtest:
        st.subheader("Backtest persistant — Lecture Supabase")
        rows_bt = _fetch_backtest_history()
        if rows_bt:
            df_bt = pd.DataFrame(rows_bt)
            df_bt["match_date"] = pd.to_datetime(df_bt["match_date"], errors="coerce")
            df_bt = df_bt.sort_values("match_date", ascending=False)

            wins = (df_bt["status"] == "WIN").sum()
            losses = (df_bt["status"] == "LOSS").sum()
            n_bets = int(wins + losses)
            total_profit = float(df_bt["profit"].fillna(0).sum())
            roi = (total_profit / n_bets * 100.0) if n_bets > 0 else 0.0
            win_rate = (wins / n_bets * 100.0) if n_bets > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("ROI Total", f"{roi:.1f}%")
            c2.metric("Win Rate", f"{win_rate:.1f}%")
            c3.metric("Profit Net", f"{total_profit:.2f} u")

            # Courbe profit cumulé (30 derniers jours)
            today = _today_paris()
            start_30 = today - timedelta(days=29)
            df_recent = df_bt[df_bt["match_date"] >= pd.Timestamp(start_30)]
            if not df_recent.empty:
                daily = (
                    df_recent.groupby(df_recent["match_date"].dt.date)["profit"]
                    .sum()
                    .reindex(pd.date_range(start_30, today), fill_value=0.0)
                )
                cumul = daily.cumsum()
                st.line_chart(cumul, height=250)

            df_display = df_bt[[
                "match_date",
                "match_name",
                "bet_suggested",
                "odds_taken",
                "profit",
                "status",
            ]].copy()
            df_display.rename(columns={
                "match_date": "Date",
                "match_name": "Match",
                "bet_suggested": "Pari",
                "odds_taken": "Cote",
                "profit": "Profit",
                "status": "Status",
            }, inplace=True)
            df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")

            st.dataframe(
                _style_backtest_table(df_display),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn("Date"),
                    "Match": st.column_config.TextColumn("Match", width="large"),
                    "Pari": st.column_config.TextColumn("Pari"),
                    "Cote": st.column_config.NumberColumn("Cote", format="%.2f"),
                    "Profit": st.column_config.NumberColumn("Profit", format="%.2f"),
                    "Status": st.column_config.TextColumn("Status"),
                },
            )
            st.caption("Données figées en base (backtest_history).")
        else:
            st.info("Aucun backtest persistant trouvé. Lancez : python 05_run_backtest.py")


if __name__ == "__main__":
    main()
