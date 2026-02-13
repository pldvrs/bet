#!/usr/bin/env python3
"""
Upset Analyzer — Analyse des profils statistiques des vainqueurs
================================================================
Script autonome pour analyser le passé (30–50 derniers matchs) et identifier
les profils des équipes qui gagnent, notamment les outsiders (victoires avec
eFG% faible = rebond / défense).
- API: basketball.api-sports.io
- Four Factors par match du VAINQUEUR : eFG%, ORB%, TOV%, FT Rate.
"""

import time
from datetime import date, timedelta
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURATION API
# ==============================================================================

API_KEY: str = "84077b8a5366ab2bbb14943e583d0ada"
BASE_URL: str = "https://v1.basketball.api-sports.io"
HEADER_KEY: str = "x-apisports-key"

LEAGUES: Dict[str, int] = {
    "Betclic Élite (FR)": 2,
    "Pro B (FR)": 8,
    "EuroLeague": 120,
    "EuroCup": 121,
    "Champions League (BCL)": 16,
    "Liga ACB (ESP)": 5,
}

SEASONS_TO_TRY: List[str] = ["2024-2025", "2025-2026", "2025", "2024"]
API_RETRIES: int = 3
API_RETRY_DELAY: float = 1.0
ORB_DENOM_APPROX: float = 25.0  # ORB% = ORB / (ORB + 25) si DRB adverse indispo

# ==============================================================================
# API
# ==============================================================================


def _api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{BASE_URL}/{endpoint}"
    headers = {HEADER_KEY: API_KEY.strip()}
    last_err: Optional[str] = None
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, headers=headers, params=params or {}, timeout=15)
            data = r.json() if r.text else {}
            if r.status_code != 200:
                last_err = data.get("errors") or f"HTTP {r.status_code}"
                if r.status_code == 429:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                continue
            errs = data.get("errors")
            if errs:
                msg = errs[0] if isinstance(errs, list) else str(errs)
                if isinstance(msg, dict):
                    msg = "; ".join(f"{k}: {v}" for k, v in msg.items())
                last_err = msg
                continue
            return data, None
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(API_RETRY_DELAY * (attempt + 1))
    return None, last_err


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, float):
        return max(0, int(val))
    if isinstance(val, dict):
        return _safe_int(val.get("total") or val.get("all"), default)
    if isinstance(val, str):
        try:
            return max(0, int(float(val)))
        except (TypeError, ValueError):
            return default
    return default


def fetch_games(league_id: int, season: str, date_str: str) -> List[dict]:
    data, err = _api_get("games", {"date": date_str, "league": league_id, "season": season, "timezone": "Europe/Paris"})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def fetch_games_league_season(league_id: int, season: str) -> List[dict]:
    """Tous les matchs d'une ligue/saison en un appel (évite le scan jour par jour)."""
    data, err = _api_get("games", {"league": league_id, "season": season, "timezone": "Europe/Paris"})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def fetch_games_by_date_only(league_id: int, date_str: str) -> Tuple[List[dict], str]:
    data, err = _api_get("games", {"date": date_str, "league": league_id, "timezone": "Europe/Paris"})
    if err or not data:
        return [], ""
    resp = data.get("response")
    games = resp if isinstance(resp, list) else []
    used_season = "2024-2025"
    if games:
        g0 = games[0]
        league = g0.get("league") if isinstance(g0.get("league"), dict) else {}
        if league and league.get("season") is not None:
            used_season = str(league.get("season", used_season))
    return games, used_season


def fetch_game_statistics_teams(game_id: int) -> List[dict]:
    data, err = _api_get("games/statistics/teams", {"id": game_id})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


# ==============================================================================
# BOX SCORE EXTRACTION & FOUR FACTORS (VAINQUEUR, UN MATCH)
# ==============================================================================


def _extract_raw_stats_from_team_game(team_stats_item: dict) -> Optional[dict]:
    """Extrait FGA, FGM, 3PM, FTA, FTM, ORB, TOV, Possessions, pts_for depuis games/statistics/teams."""
    raw = team_stats_item
    try:
        fg = raw.get("field_goals") or {}
        fg_for = (fg.get("for") if isinstance(fg, dict) else {}) or {}
        fga = _safe_int(fg_for.get("attempted"))
        fgm = _safe_int(fg_for.get("made")) or _safe_int(fg_for.get("total"))
        th = raw.get("three_points") or {}
        th_for = (th.get("for") if isinstance(th, dict) else {}) or {}
        thm = _safe_int(th_for.get("made")) or _safe_int(th_for.get("total"))
        ft = raw.get("free_throws") or {}
        ft_for = (ft.get("for") if isinstance(ft, dict) else {}) or {}
        fta = _safe_int(ft_for.get("attempted"))
        ftm = _safe_int(ft_for.get("made")) or _safe_int(ft_for.get("total"))
        reb = raw.get("rebounds") or {}
        reb_for = (reb.get("for") if isinstance(reb, dict) else {}) or {}
        orb = _safe_int(reb_for.get("offensive"))
        tov = raw.get("turnovers") or {}
        tov_for = (tov.get("for") if isinstance(tov, dict) else {}) or {}
        tov_val = _safe_int(tov_for.get("total")) if isinstance(tov_for, dict) else _safe_int(tov)
        pts = raw.get("points") or {}
        pts_for = _safe_int((pts.get("for") or {}).get("total") if isinstance(pts.get("for"), dict) else pts.get("for"))
    except Exception:
        return None
    if fga <= 0 and fta <= 0:
        return None
    poss = fga + int(0.44 * fta) + tov_val - orb
    if poss <= 0:
        poss = max(1, fga + fta)
    return {
        "FGA": fga, "FGM": fgm, "3PM": thm, "FTA": fta, "FTM": ftm,
        "ORB": orb, "TOV": tov_val, "Possessions": poss, "pts_for": pts_for,
    }


def four_factors_single_match(raw: dict) -> Optional[dict]:
    """
    Four Factors du vainqueur sur UN match.
    - eFG% = (FGM + 0.5 * 3PM) / FGA
    - ORB% = ORB / (ORB + 25) (approximation)
    - TOV% = TOV / Possessions
    - FT Rate = FTA / FGA
    Retourne dict avec efg_pct, orb_pct, tov_pct, ft_rate, threes (3PM) ou None.
    """
    if not raw:
        return None
    fga = raw.get("FGA", 0)
    fgm = raw.get("FGM", 0)
    thm = raw.get("3PM", 0)
    fta = raw.get("FTA", 0)
    orb = raw.get("ORB", 0)
    tov = raw.get("TOV", 0)
    poss = raw.get("Possessions", 1)
    if fga <= 0:
        return None
    efg_pct = (fgm + 0.5 * thm) / fga
    orb_pct = orb / (orb + ORB_DENOM_APPROX) if (orb + ORB_DENOM_APPROX) > 0 else 0.0
    tov_pct = tov / poss if poss > 0 else 0.0
    ft_rate = fta / fga if fga > 0 else 0.0
    return {
        "efg_pct": efg_pct,
        "orb_pct": orb_pct,
        "tov_pct": tov_pct,
        "ft_rate": ft_rate,
        "threes": thm,
    }


def _game_total_scores(g: dict) -> Tuple[int, int]:
    scores = g.get("scores") or {}
    home = scores.get("home")
    away = scores.get("away")
    if isinstance(home, dict):
        pts_h = _safe_int(home.get("total") or home.get("points"))
    else:
        pts_h = _safe_int(home)
    if isinstance(away, dict):
        pts_a = _safe_int(away.get("total") or away.get("points"))
    else:
        pts_a = _safe_int(away)
    return pts_h, pts_a


# ==============================================================================
# DATA FETCHING — N derniers matchs terminés (scan jour par jour)
# ==============================================================================


def _collect_finished_games_list(league_id: int, n: int) -> List[dict]:
    """
    Récupère la liste des N derniers matchs terminés (FT) sans box score.
    Priorité : 1 appel par saison (league+season) puis tri par date ; secours : scan jour par jour.
    """
    all_finished: List[dict] = []
    seen_gid: set = set()

    # 1) Essai rapide : tous les matchs ligue/saison en un appel par saison
    for season in SEASONS_TO_TRY:
        games = fetch_games_league_season(league_id, season)
        for g in games:
            status = g.get("status") or {}
            status_short = status.get("short") if isinstance(status, dict) else str(status)
            if status_short != "FT":
                continue
            gid = g.get("id")
            if gid is None or gid in seen_gid:
                continue
            seen_gid.add(gid)
            pts_h, pts_a = _game_total_scores(g)
            if pts_h == 0 and pts_a == 0:
                continue
            all_finished.append(g)
        if len(all_finished) >= n * 2:  # assez pour prendre les N plus récents
            break

    if not all_finished:
        # 2) Secours : scan jour par jour (limité à 45 jours pour éviter temps excessif)
        today = date.today()
        for days_back in range(45):
            d = today - timedelta(days=days_back)
            date_str = d.strftime("%Y-%m-%d")
            games, _ = fetch_games_by_date_only(league_id, date_str)
            for g in games:
                status = g.get("status") or {}
                status_short = status.get("short") if isinstance(status, dict) else str(status)
                if status_short != "FT":
                    continue
                gid = g.get("id")
                if gid is None or gid in seen_gid:
                    continue
                seen_gid.add(gid)
                pts_h, pts_a = _game_total_scores(g)
                if pts_h == 0 and pts_a == 0:
                    continue
                all_finished.append(g)
            if len(all_finished) >= n:
                break

    # Tri par date décroissante, prendre les N premiers
    def _date_key(gg):
        d = gg.get("date") or ""
        return (d[:10] if len(d) >= 10 else d, gg.get("id", 0))

    all_finished.sort(key=_date_key, reverse=True)
    return all_finished[:n]


@st.cache_data(ttl=600)
def fetch_last_n_finished_games_meta(league_id: int, n: int) -> List[Dict[str, Any]]:
    """
    Liste des N derniers matchs terminés (FT) : id, scores, équipes, winner_id.
    En 1–4 appels API (ligue/saison). Pas de box score (pour affichage progress).
    """
    games_list = _collect_finished_games_list(league_id, n)
    out: List[Dict[str, Any]] = []
    for g in games_list:
        gid = g.get("id")
        pts_h, pts_a = _game_total_scores(g)
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
        away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
        home_name = home.get("name", "Domicile") if isinstance(home, dict) else "Domicile"
        away_name = away.get("name", "Extérieur") if isinstance(away, dict) else "Extérieur"
        if not home_id or not away_id:
            continue
        winner_id = home_id if pts_h > pts_a else away_id
        out.append({
            "game_id": gid,
            "date": (g.get("date") or "")[:10],
            "home_id": home_id, "away_id": away_id,
            "home_name": home_name, "away_name": away_name,
            "pts_h": pts_h, "pts_a": pts_a,
            "winner_id": winner_id,
        })
    return out


def enrich_matches_with_box_scores(metas: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
    """
    Pour chaque match meta, récupère le box score et calcule les Four Factors du vainqueur.
    progress_callback(i, total) optionnel pour afficher la progression.
    """
    collected: List[Dict[str, Any]] = []
    total = len(metas)
    for i, m in enumerate(metas):
        if progress_callback:
            progress_callback(i, total)
        gid = m.get("game_id")
        winner_id = m.get("winner_id")
        home_name = m.get("home_name", "Domicile")
        away_name = m.get("away_name", "Extérieur")
        pts_h, pts_a = m.get("pts_h", 0), m.get("pts_a", 0)
        score_str = f"{pts_h}-{pts_a}"
        winner_name = home_name if pts_h > pts_a else away_name

        team_stats_list = fetch_game_statistics_teams(gid)
        time.sleep(0.1)
        winner_raw = None
        for item in team_stats_list:
            tid = item.get("team_id")
            if tid is None and isinstance(item.get("team"), dict):
                tid = item.get("team", {}).get("id")
            if tid is not None and int(tid) == winner_id:
                winner_raw = _extract_raw_stats_from_team_game(item)
                break

        factors = four_factors_single_match(winner_raw) if winner_raw else None
        if factors is None:
            continue

        collected.append({
            "match": f"{home_name} vs {away_name}",
            "winner": winner_name,
            "score": score_str,
            "date": m.get("date", ""),
            "efg_pct": factors["efg_pct"],
            "orb_pct": factors["orb_pct"],
            "tov_pct": factors["tov_pct"],
            "ft_rate": factors["ft_rate"],
            "threes": factors["threes"],
        })
    if progress_callback and total:
        progress_callback(total, total)
    return collected


@st.cache_data(ttl=600)
def fetch_last_n_finished_matches(league_id: int, n: int) -> List[Dict[str, Any]]:
    """
    Récupère les N derniers matchs terminés (FT) avec Four Factors du vainqueur.
    Liste des matchs en 1–4 appels (ligue/saison) ; puis N appels box score.
    """
    metas = fetch_last_n_finished_games_meta(league_id, n)
    if not metas:
        return []
    return enrich_matches_with_box_scores(metas, progress_callback=None)


# ==============================================================================
# STREAMLIT — UI
# ==============================================================================


def plot_winner_profile_radar(rows: List[Dict[str, Any]]) -> go.Figure:
    """Radar : profil type du vainqueur (moyennes eFG%, ORB%, 1-TOV%, FT Rate)."""
    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Profil Type du Vainqueur (aucune donnée)", height=400)
        return fig
    mean_efg = sum(r["efg_pct"] for r in rows) / len(rows)
    mean_orb = sum(r["orb_pct"] for r in rows) / len(rows)
    mean_tov = sum(r["tov_pct"] for r in rows) / len(rows)
    mean_ft = sum(r["ft_rate"] for r in rows) / len(rows)
    categories = ["Efficacité Tir (eFG%)", "Rebond Off (ORB%)", "Propreté (1 - TOV%)", "Lancers (FT Rate)"]
    vals = [mean_efg, mean_orb, 1.0 - mean_tov, mean_ft]
    r = vals + [vals[0]]
    theta = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself", name="Profil Type Vainqueur", line=dict(color="darkgreen")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.7])),
        showlegend=True,
        title="Profil Type du Vainqueur (moyennes)",
        height=400,
    )
    return fig


def style_efg_orb_gradient(df: pd.DataFrame):
    """Dégradé vert (haut) → rouge (bas) sur eFG% et ORB% (valeurs en 0–100)."""
    subset = [c for c in ["eFG%", "ORB%"] if c in df.columns]
    if not subset:
        return df.style
    return df.style.background_gradient(subset=subset, cmap="RdYlGn", axis=None, vmin=0, vmax=60)


def main() -> None:
    st.set_page_config(page_title="Upset Analyzer", layout="wide", page_icon="📊")
    st.title("📊 Upset Analyzer — Profils des Vainqueurs")
    st.caption("Analyse des 30–50 derniers matchs terminés : Four Factors du vainqueur (eFG%, ORB%, TOV%, FT Rate).")

    league_name = st.selectbox("Ligue", list(LEAGUES.keys()), key="league_upset")
    league_id = LEAGUES[league_name]
    n_matches = st.slider("Nombre de matchs à analyser", min_value=10, max_value=50, value=30, step=5, key="n_upset")

    st.caption("Premier chargement : liste en 1–4 appels puis box scores (~5–15 s). Résultat mis en cache 10 min.")
    with st.spinner("Récupération des matchs terminés et box scores…"):
        rows = fetch_last_n_finished_matches(league_id, n_matches)

    if not rows:
        st.warning("Aucun match terminé avec box score trouvé pour cette ligue. Essayez une autre ligue ou augmentez la période.")
        return

    st.success(f"{len(rows)} match(s) analysé(s).")

    # KPIs globaux
    st.subheader("📈 KPIs globaux — Moyennes des vainqueurs")
    mean_efg = sum(r["efg_pct"] for r in rows) / len(rows)
    mean_orb = sum(r["orb_pct"] for r in rows) / len(rows)
    mean_tov = sum(r["tov_pct"] for r in rows) / len(rows)
    mean_ft = sum(r["ft_rate"] for r in rows) / len(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("eFG% moyen", f"{mean_efg * 100:.1f}%", help="Efficacité tir (2P + 3P pondéré)")
    c2.metric("ORB% moyen", f"{mean_orb * 100:.1f}%", help="Rebonds offensifs (approx ORB/(ORB+25))")
    c3.metric("TOV% moyen", f"{mean_tov * 100:.1f}%", help="Pertes de balle / possessions")
    c4.metric("FT Rate moyen", f"{mean_ft * 100:.1f}%", help="FTA / FGA")

    # Radar
    st.subheader("🎯 Profil type du vainqueur (radar)")
    fig_radar = plot_winner_profile_radar(rows)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Tableau des surprises
    st.subheader("📋 Tableau des matchs — Four Factors du vainqueur")
    df = pd.DataFrame([
        {
            "Match": r["match"],
            "Vainqueur": r["winner"],
            "Score": r["score"],
            "eFG%": round(r["efg_pct"] * 100, 1),
            "ORB%": round(r["orb_pct"] * 100, 1),
            "TOV%": round(r["tov_pct"] * 100, 1),
            "3PM": r["threes"],
        }
        for r in rows
    ])
    # Colorimétrie : dégradé vert (haut) / rouge (bas) sur eFG% et ORB%
    styled = style_efg_orb_gradient(df)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Vert = valeur haute, Rouge = valeur basse. Une victoire avec eFG% faible (rouge) et ORB% élevé (vert) signale un profil « outsider » (rebond/défense).")

    # Conseil Quants
    st.markdown("---")
    st.subheader("💡 Comment interpréter les données (Quants)")
    st.markdown("""
    - **eFG% faible mais Victoire** → Regarder **ORB%** : une équipe qui gagne en étant moins efficace au tir compense souvent par les rebonds offensifs (secondes chances). Profil typique d’outsider ou match physique.
    - **eFG% élevé + ORB% élevé** → Vainqueur dominant (efficacité + contrôle du panneau). Les favoris qui confirment.
    - **TOV% élevé chez le vainqueur** → Soit match chaotique, soit l’adversaire a encore plus perdu de balles. Croiser avec le TOV% du perdant si disponible.
    - **FT Rate** → Indique si la victoire s’appuie sur les lancers francs (agressivité, fin de match). Utile pour les Totaux (Over/Under) et le style de jeu.
    - **3PM** → Nombre de 3 pts du vainqueur sur le match. Repérer les équipes qui gagnent « à la 3 » vs « dans la raquette » pour adapter les paris (totaux, handicaps).
    """)


if __name__ == "__main__":
    main()
