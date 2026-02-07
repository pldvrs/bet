#!/usr/bin/env python3
"""
Terminal V25 — Read-Only Dashboard (CQRS)
==========================================
Interface de lecture pure. Zéro appel API. Données exclusivement depuis Supabase.
Prédictions calculées à la volée (Pythagorean Expectation) depuis box_scores.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import get_client

try:
    from archetype_engine import get_tactical_clash
except Exception:
    get_tactical_clash = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================

LEAGUES: Dict[str, int] = {
    "Betclic Élite (FR)": 2,
    "Pro B (FR)": 8,
    "EuroLeague": 120,
    "LBA Italie": 4,
    "Lega A (Italie)": 52,
    "Liga ACB (ESP)": 5,
    "Turquie BSL": 194,
    "Grèce HEBA": 198,
    "Greek Basket League": 45,
    "ABA League (ADR)": 206,
}

LEAGUE_IDS: set = set(LEAGUES.values())
CACHE_TTL: int = 300
HOME_ADVANTAGE_PTS: float = 3.0
PYTHAGOREAN_EXP: float = 10.2
PACE_DEFAULT: float = 72.0
RADAR_KEYS: List[str] = ["volume_exterieur", "pression_interieure", "controle_tempo", "identite_defensive"]
RADAR_CATEGORIES: List[str] = ["Volume Extérieur", "Pression Intérieure", "Contrôle Tempo", "Identité Défensive"]

# ==============================================================================
# DATABASE LAYER (Supabase uniquement)
# ==============================================================================


def _get_supabase():
    return get_client()


@st.cache_data(ttl=CACHE_TTL)
def get_future_games() -> List[dict]:
    """
    Récupère les matchs à venir (date >= today) depuis games_history.
    """
    supabase = _get_supabase()
    if not supabase:
        return []

    today_str = date.today().strftime("%Y-%m-%d")
    try:
        r = (
            supabase.table("games_history")
            .select("*")
            .gte("date", today_str)
            .order("date", desc=False)
            .execute()
        )
        rows = r.data or []
        return [g for g in rows if g.get("league_id") in LEAGUE_IDS]
    except Exception as e:
        st.error(f"Erreur Supabase: {e}")
        return []


@st.cache_data(ttl=CACHE_TTL)
def get_team_profile(team_id: int) -> Optional[dict]:
    supabase = _get_supabase()
    if not supabase:
        return None
    try:
        r = supabase.table("teams_metadata").select("*").eq("team_id", team_id).limit(1).execute()
        data = r.data or []
        return data[0] if data else None
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL)
def get_team_name(team_id: int) -> str:
    meta = get_team_profile(team_id)
    if meta:
        nom = (meta.get("nom_equipe") or meta.get("name") or "").strip()
        if nom:
            return nom
    return f"Équipe {team_id}"


@st.cache_data(ttl=CACHE_TTL)
def get_team_stats_from_box_scores(team_id: int, n_games: int = 10) -> Optional[dict]:
    """
    Moyennes (Pace, Off Rtg, Def Rtg, etc.) depuis box_scores.
    """
    supabase = _get_supabase()
    if not supabase:
        return None
    try:
        r = (
            supabase.table("box_scores")
            .select("pace, off_rtg, def_rtg, efg_pct, orb_pct, tov_pct, ft_rate")
            .eq("team_id", team_id)
            .order("game_id", desc=True)
            .limit(n_games)
            .execute()
        )
        rows = r.data or []
        if len(rows) < 2:
            return None

        n = len(rows)
        pace = sum(x.get("pace") or PACE_DEFAULT for x in rows) / n
        off_rtg = sum(x.get("off_rtg") or 100 for x in rows) / n
        def_rtg = sum(x.get("def_rtg") or 100 for x in rows) / n

        return {
            "pace": round(float(pace), 1),
            "off_rtg": round(float(off_rtg), 1),
            "def_rtg": round(float(def_rtg), 1),
            "n_games": n,
        }
    except Exception:
        return None


# ==============================================================================
# CALCULS PURS (Pythagorean Expectation)
# ==============================================================================


def calculate_win_probability(
    home_stats: Optional[dict],
    away_stats: Optional[dict],
    home_advantage_pts: float = HOME_ADVANTAGE_PTS,
    exp: float = PYTHAGOREAN_EXP,
) -> Tuple[float, float, float]:
    """
    Calcule la probabilité de victoire domicile et le Fair Spread.
    Formule : proj_home = (off_home + def_away)/2 * pace/100 + home_adv
    Retourne (prob_home, proj_home, proj_away).
    """
    pace_h = (home_stats or {}).get("pace") or PACE_DEFAULT
    pace_a = (away_stats or {}).get("pace") or PACE_DEFAULT
    off_h = (home_stats or {}).get("off_rtg") or 100.0
    def_h = (home_stats or {}).get("def_rtg") or 100.0
    off_a = (away_stats or {}).get("off_rtg") or 100.0
    def_a = (away_stats or {}).get("def_rtg") or 100.0

    pace_avg = (pace_h + pace_a) / 2.0
    proj_home = (off_h / 100.0 * pace_avg) + home_advantage_pts
    proj_away = off_a / 100.0 * pace_avg

    denom = (proj_home**exp) + (proj_away**exp)
    prob_home = float((proj_home**exp) / denom) if denom > 0 else 0.5

    return prob_home, proj_home, proj_away


def calculate_fair_spread(proj_home: float, proj_away: float, home_name: str, away_name: str) -> str:
    spread = proj_home - proj_away
    if abs(spread) < 0.1:
        return "Égalité"
    if spread > 0:
        return f"{home_name} -{spread:.1f}"
    return f"{away_name} -{abs(spread):.1f}"


# ==============================================================================
# UI HELPERS
# ==============================================================================


def _badge_style(archetype: str) -> str:
    if not archetype:
        return "—"
    arch = str(archetype)
    if "Pace & Space" in arch:
        return "🚀 Pace & Space"
    if "Grit & Grind" in arch:
        return "🛡️ Grit & Grind"
    if "Paint Beast" in arch:
        return "💪 Paint Beast"
    if "Snipers" in arch:
        return "🎯 Snipers"
    if "Run & Gun" in arch:
        return "⚡ Run & Gun"
    return arch


def _format_game_time(d: str) -> str:
    if not d or len(d) < 10:
        return d or "—"
    try:
        dt = datetime.strptime(d[:10], "%Y-%m-%d")
        return dt.strftime("%d/%m")
    except Exception:
        return d[:10]


def plot_tactical_radar(pv_home: dict, pv_away: dict, home_name: str, away_name: str) -> go.Figure:
    def _vals(pv: dict) -> List[float]:
        out = []
        for k in RADAR_KEYS:
            v = (pv or {}).get(k, 50)
            try:
                out.append(min(100, max(0, float(v))))
            except (TypeError, ValueError):
                out.append(50)
        return out + [out[0]]

    theta = RADAR_CATEGORIES + [RADAR_CATEGORIES[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=_vals(pv_home), theta=theta, fill="toself", name=home_name, line=dict(color="#3b82f6"))
    )
    fig.add_trace(
        go.Scatterpolar(r=_vals(pv_away), theta=theta, fill="toself", name=away_name, line=dict(color="#ef4444"))
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Radar Tactique",
        height=450,
    )
    return fig


# ==============================================================================
# SNIPER SCANNER (Onglet 1)
# ==============================================================================


@st.cache_data(ttl=CACHE_TTL)
def build_sniper_table() -> pd.DataFrame:
    """
    Tableau des matchs à venir avec Probabilité, Fair Spread, Badges de Style.
    """
    games = get_future_games()
    if not games:
        return pd.DataFrame()

    league_id_to_name = {v: k for k, v in LEAGUES.items()}
    rows: List[dict] = []

    for g in games:
        game_id = g.get("game_id")
        date_str = (g.get("date") or "")[:10]
        league_id = g.get("league_id")
        home_id = g.get("home_id")
        away_id = g.get("away_id")
        if not home_id or not away_id:
            continue

        home_name = get_team_name(home_id)
        away_name = get_team_name(away_id)
        home_profile = get_team_profile(home_id)
        away_profile = get_team_profile(away_id)
        home_arch = (home_profile or {}).get("current_archetype") or "—"
        away_arch = (away_profile or {}).get("current_archetype") or "—"

        home_stats = get_team_stats_from_box_scores(home_id)
        away_stats = get_team_stats_from_box_scores(away_id)
        prob_home, proj_home, proj_away = calculate_win_probability(home_stats, away_stats)
        fair_spread = calculate_fair_spread(proj_home, proj_away, home_name, away_name)

        # Edge simulé (pour tri) : écart à 50%
        edge = abs(prob_home - 0.5) * 100 if prob_home else 0

        league_name = league_id_to_name.get(league_id, str(league_id))

        rows.append({
            "_game_id": game_id,
            "_home_id": home_id,
            "_away_id": away_id,
            "_home_name": home_name,
            "_away_name": away_name,
            "Heure": _format_game_time(date_str),
            "Date": date_str,
            "Ligue": league_name,
            "Match": f"{home_name} vs {away_name}",
            "Prob. Domicile %": round(prob_home * 100, 1),
            "Fair Spread": fair_spread,
            "Style Domicile": _badge_style(home_arch),
            "Style Extérieur": _badge_style(away_arch),
            "_edge": edge,
            "_prob_home": prob_home,
            "_proj_home": proj_home,
            "_proj_away": proj_away,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["Date", "_edge"], ascending=[True, False], ignore_index=True)


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    st.set_page_config(page_title="Terminal V25 — Sniper", layout="wide", page_icon="🏀")
    st.title("🏀 Terminal V25 — Read-Only Dashboard")
    st.caption("Données Supabase · Prédictions à la volée · Zéro appel API")

    supabase = _get_supabase()
    if not supabase:
        st.error("❌ Connexion Supabase impossible. Vérifiez SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY dans .env")
        st.stop()

    st.markdown("---")

    with st.spinner("Chargement des matchs à venir…"):
        df_sniper = build_sniper_table()

    if df_sniper.empty:
        st.warning("⚠️ Aucun match à venir trouvé.")
        st.info("**Lancer le backend pour récupérer le calendrier** : `python3 backend_engine.py --future-only`")
        st.caption("Le backend appelle l'API pour récupérer les matchs des 7 prochains jours et les insère dans games_history.")
        return

    st.success(f"✅ {len(df_sniper)} match(s) à venir chargé(s).")

    tab_sniper, tab_lab = st.tabs(["🎯 Le Sniper Scanner", "🔬 Le Laboratoire"])

    # --- Onglet 1 : Sniper Scanner ---
    with tab_sniper:
        st.subheader("Matchs à venir — Probabilité, Fair Spread, Styles")
        sort_by = st.radio("Trier par", ["Heure (chrono)", "Edge"], horizontal=True, key="sort_sniper")
        if "Edge" in sort_by:
            df_display = df_sniper.sort_values("_edge", ascending=False, ignore_index=True)
        else:
            df_display = df_sniper

        display_cols = ["Heure", "Date", "Ligue", "Match", "Prob. Domicile %", "Fair Spread", "Style Domicile", "Style Extérieur"]
        st.dataframe(
            df_display[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Heure": st.column_config.TextColumn("Heure", width="small"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Ligue": st.column_config.TextColumn("Ligue", width="small"),
                "Match": st.column_config.TextColumn("Match", width="large"),
                "Prob. Domicile %": st.column_config.NumberColumn("Prob. Domicile %", format="%.1f"),
                "Fair Spread": st.column_config.TextColumn("Fair Spread", width="medium"),
                "Style Domicile": st.column_config.TextColumn("Style D.", width="small"),
                "Style Extérieur": st.column_config.TextColumn("Style E.", width="small"),
            },
        )

    # --- Onglet 2 : Laboratoire ---
    with tab_lab:
        st.subheader("Deep Dive — Radar Tactique & Alerte Kryptonite")
        match_options = df_sniper["Match"].tolist()
        if not match_options:
            st.info("Aucun match à analyser.")
        else:
            sel = st.selectbox("Choisir un match", match_options, key="match_lab")
            if sel:
                row = df_sniper[df_sniper["Match"] == sel].iloc[0]
                home_id = int(row["_home_id"])
                away_id = int(row["_away_id"])
                home_name = str(row.get("_home_name", row["Match"].split(" vs ")[0] if " vs " in str(row["Match"]) else "Domicile"))
                away_name = str(row.get("_away_name", row["Match"].split(" vs ")[1] if " vs " in str(row["Match"]) else "Extérieur"))

                home_profile = get_team_profile(home_id)
                away_profile = get_team_profile(away_id)
                pv_h = (home_profile or {}).get("profile_vector") or {}
                pv_a = (away_profile or {}).get("profile_vector") or {}

                st.markdown("**Radar Tactique**")
                fig = plot_tactical_radar(pv_h, pv_a, home_name, away_name)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Alerte Kryptonite**")
                if get_tactical_clash:
                    clash = get_tactical_clash(home_id, away_id)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Domicile", clash.get("home_archetype") or "—")
                    with col2:
                        st.metric("Extérieur", clash.get("away_archetype") or "—")
                    if clash.get("kryptonite"):
                        st.error(f"⚠️ KRYPTONITE : {clash.get('kryptonite_msg', '')}")
                    else:
                        st.success("Pas de clash tactique majeur.")
                    if clash.get("alert"):
                        st.warning(clash["alert"])
                else:
                    st.info("Module archetype_engine non disponible.")


if __name__ == "__main__":
    main()
