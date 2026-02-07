#!/usr/bin/env python3
"""
Sniffer Diagnostic — Structure réelle du JSON Box Scores (API Basketball).
Objectif : voir de nos yeux la forme des données (field_goals, rebounds) pour
adapter le Terminal V24. Cible EuroLeague (120), ACB (5), et ABA League (206)
avec fallback Reconstructed from Players.

Usage:
  python debug_api_basketball.py
  GAME_ID=470712 python debug_api_basketball.py   # Sniffer un Match ID précis
"""
import json
import re
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

# ==============================================================================
# CONFIG (aligné Terminal V24)
# ==============================================================================
API_KEY: str = os.environ.get("API_BASKETBALL_KEY", "84077b8a5366ab2bbb14943e583d0ada").strip()
BASE_URL: str = "https://v1.basketball.api-sports.io"
HEADER_KEY: str = "x-apisports-key"

EUROLEAGUE_ID: int = 120
ACB_ID: int = 5
ABA_LEAGUE_ID: int = 206

SEASONS_TO_TRY: List[str] = ["2025", "2024", "2024-2025", "2025-2026"]


def _api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{BASE_URL}/{endpoint}"
    headers = {HEADER_KEY: API_KEY}
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=15)
        data = r.json() if r.text else {}
        if r.status_code != 200:
            return None, data.get("errors") or f"HTTP {r.status_code}"
        if data.get("errors"):
            errs = data["errors"]
            msg = errs[0] if isinstance(errs, list) else str(errs)
            if isinstance(msg, dict):
                msg = "; ".join(f"{k}: {v}" for k, v in msg.items())
            return None, msg
        return data, None
    except requests.RequestException as e:
        return None, str(e)


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
        # "45%", "12", " 7 "
        s = val.strip().replace("%", "").replace(",", ".")
        try:
            return max(0, int(float(s)))
        except (TypeError, ValueError):
            return default
    return default


# ==============================================================================
# BRUTE FORCE EXTRACTION — teste toutes les clés possibles
# ==============================================================================


def brute_force_extract(data: Any, keys_to_try: List[str]) -> Optional[int]:
    """
    Teste récursivement : data['total'], data['all'], data['for']['total'], etc.
    Si data est un int/float direct (ex: turnovers=11), retourne l'int.
    Si la donnée est une string (ex: "45%"), conversion propre en int.
    Retourne la première valeur entière trouvée, sinon None.
    """
    if data is None:
        return None
    # Doc API: turnovers peut être un int direct
    if isinstance(data, (int, float)):
        return max(0, int(data))
    # Accès direct
    for key in keys_to_try:
        if not isinstance(data, dict):
            break
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return max(0, int(val))
        if isinstance(val, str):
            return _safe_int(val, 0) or None
        if isinstance(val, dict):
            inner = brute_force_extract(val, ["total", "all", "made", "attempted", "attempts", "offensive", "offence"])
            if inner is not None:
                return inner
    # Sous-clé "for" (format API nested)
    if isinstance(data, dict) and "for" in data:
        inner = brute_force_extract(data["for"], keys_to_try)
        if inner is not None:
            return inner
    return None


def _deep_scan_section(raw: dict, section_name: str, key_variants: List[str]) -> Dict[str, Any]:
    """
    Pour une section (field_goals, rebounds, ...), extrait toutes les sous-valeurs
    en testant key_variants + chemins 'for'. Retourne un dict {clé: valeur ou "N/A"}.
    """
    section = raw.get(section_name)
    # Aliases (Euroleague, etc.)
    if section is None and section_name == "field_goals":
        section = raw.get("field_goals_goals")
    if section is None and section_name == "rebounds":
        section = raw.get("rebounds_goals")
    if section is None and section_name == "three_points":
        section = raw.get("three_points_goals") or raw.get("threepoint_goals")
    if section is None and section_name == "free_throws":
        section = raw.get("free_throws_goals") or raw.get("freethrows_goals")
    if section is None and section_name == "turnovers":
        section = raw.get("turnovers_goals")
    if section is None and section_name == "points":
        section = raw.get("points_goals")

    result: Dict[str, Any] = {}
    if not isinstance(section, dict):
        result["_raw"] = section
        result["_keys_tried"] = list(raw.keys()) if isinstance(raw, dict) else []
        return result

    # Clés à chercher selon la section
    if section_name == "field_goals":
        for label, keys in [("attempted", ["attempted", "attempts"]), ("made", ["made", "total", "all"])]:
            v = brute_force_extract(section, keys)
            result[label] = v if v is not None else "N/A"
    elif section_name == "rebounds":
        for label, keys in [
            ("total", ["total", "all"]),
            ("offensive", ["offensive", "offence"]),
        ]:
            v = brute_force_extract(section, keys)
            result[label] = v if v is not None else "N/A"
    elif section_name == "three_points":
        v = brute_force_extract(section, ["made", "total", "all"])
        result["made"] = v if v is not None else "N/A"
    elif section_name == "free_throws":
        for label, keys in [("attempted", ["attempted", "attempts"]), ("made", ["made", "total", "all"])]:
            v = brute_force_extract(section, keys)
            result[label] = v if v is not None else "N/A"
    elif section_name == "turnovers":
        v = brute_force_extract(section, ["total", "all"])
        result["total"] = v if v is not None else "N/A"
    elif section_name == "points":
        v = brute_force_extract(section, ["total", "all"])
        result["total"] = v if v is not None else "N/A"
    result["_raw_section"] = section
    return result


def deep_scan_team_item(team_item: dict) -> Dict[str, Any]:
    """
    Brute force sur tout l'item équipe : field_goals, rebounds, three_points,
    free_throws, turnovers, points. Retourne un dict par section avec valeurs ou N/A.
    """
    out: Dict[str, Any] = {"game_id": team_item.get("game_id") or (team_item.get("game") or {}).get("id")}
    out["team_id"] = team_item.get("team_id") or (team_item.get("team") or {}).get("id")
    for section in ["field_goals", "rebounds", "three_points", "free_throws", "turnovers", "points"]:
        out[section] = _deep_scan_section(team_item, section, ["total", "all", "made", "attempted", "attempts", "offensive", "offence"])
    return out


# ==============================================================================
# Équipe — tous les matchs de la saison (pour sniffer par équipe)
# ==============================================================================


def fetch_teams_by_league_season(league_id: int, season: str) -> List[Dict[str, Any]]:
    """
    Liste les équipes de la ligue pour la saison (API teams?league=&season=).
    Si vide ou erreur, fallback : extraire les équipes des matchs des 60 derniers jours.
    Retourne [ {"id": int, "name": str}, ... ] trié par nom.
    """
    data, err = _api_get("teams", {"league": league_id, "season": season})
    if not err and data:
        resp = data.get("response")
        if isinstance(resp, list) and resp:
            out = []
            for t in resp:
                tid = t.get("id")
                name = (t.get("name") or "?").strip()
                if tid is not None and name:
                    out.append({"id": int(tid), "name": name})
            if out:
                out.sort(key=lambda x: x["name"].lower())
                return out
    # Fallback : matchs des 60 derniers jours, extraire équipes uniques
    from datetime import date, timedelta
    seen: Dict[int, str] = {}
    for days_back in range(60):
        d_str = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for try_season in [season] + [s for s in SEASONS_TO_TRY if s != season]:
            data, err = _api_get("games", {"league": league_id, "season": try_season, "date": d_str, "timezone": "Europe/Paris"})
            if err or not data:
                continue
            resp = data.get("response")
            if not isinstance(resp, list):
                continue
            for g in resp:
                for side in ("home", "away"):
                    team = (g.get("teams") or {}).get(side)
                    if isinstance(team, dict):
                        tid = team.get("id")
                        name = (team.get("name") or "?").strip()
                        if tid is not None and tid not in seen:
                            seen[int(tid)] = name
            time.sleep(0.1)
    out = [{"id": k, "name": v} for k, v in sorted(seen.items(), key=lambda x: x[1].lower())]
    return out


def fetch_team_games_full(league_id: int, season: str, team_id: int, max_games: int = 80) -> List[dict]:
    """
    Tous les matchs de l'équipe sur la saison (game_id, date, adversaire, etc.).
    Retourne liste de {game_id, date, opponent_name, opponent_id, home_away, pts_for, pts_against}.
    """
    for try_season in [season] + [s for s in SEASONS_TO_TRY if s != season]:
        data, err = _api_get("games", {"league": league_id, "season": try_season, "team": team_id})
        if err or not data:
            continue
        resp = data.get("response")
        if not isinstance(resp, list):
            continue
        out: List[dict] = []
        for g in resp:
            gid = g.get("id")
            teams_g = g.get("teams") or {}
            home = teams_g.get("home") or {}
            away = teams_g.get("away") or {}
            home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
            away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
            home_name = home.get("name", "?") if isinstance(home, dict) else "?"
            away_name = away.get("name", "?") if isinstance(away, dict) else "?"
            scores = g.get("scores") or {}
            sh = scores.get("home")
            sa = scores.get("away")
            pts_h = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
            pts_a = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)
            if home_id == team_id:
                out.append({
                    "game_id": gid, "date": g.get("date"),
                    "opponent_name": away_name, "opponent_id": away_id, "home_away": "Domicile",
                    "pts_for": pts_h, "pts_against": pts_a,
                })
            elif away_id == team_id:
                out.append({
                    "game_id": gid, "date": g.get("date"),
                    "opponent_name": home_name, "opponent_id": home_id, "home_away": "Extérieur",
                    "pts_for": pts_a, "pts_against": pts_h,
                })
        out.sort(key=lambda x: x.get("date") or "", reverse=True)
        return out[:max_games]
    return []


def fetch_game_statistics_teams_batch(game_ids: List[int], max_per_call: int = 20) -> List[dict]:
    """Box scores pour plusieurs matchs en un appel (param ids, max 20)."""
    if not game_ids:
        return []
    ids_str = "-".join(str(gid) for gid in game_ids[:max_per_call])
    data, err = _api_get("games/statistics/teams", {"ids": ids_str})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


# ==============================================================================
# Récupération des 3 derniers matchs terminés (par ligue)
# ==============================================================================


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


def get_last_n_finished_games(league_id: int, n: int = 3) -> List[Tuple[dict, str]]:
    """
    Parcourt les jours passés, récupère les matchs par date+ligue, garde les terminés (FT ou scores > 0).
    Retourne [(game_dict, used_season), ...] (au plus n).
    """
    from datetime import date, timedelta

    collected: List[Tuple[dict, str]] = []
    seen_gid: set = set()
    for days_back in range(1, 60):
        d_str = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for season in SEASONS_TO_TRY:
            data, err = _api_get("games", {"date": d_str, "league": league_id, "season": season, "timezone": "Europe/Paris"})
            if err or not data:
                continue
            resp = data.get("response")
            if not isinstance(resp, list):
                continue
            for g in resp:
                gid = g.get("id")
                if not gid or gid in seen_gid:
                    continue
                status = g.get("status") or {}
                status_short = status.get("short") if isinstance(status, dict) else str(status)
                pts_h, pts_a = _game_total_scores(g)
                finished = (status_short == "FT") or (pts_h > 0 or pts_a > 0)
                if not finished or (pts_h == 0 and pts_a == 0):
                    continue
                seen_gid.add(gid)
                collected.append((g, season))
                if len(collected) >= n:
                    return collected
        if len(collected) >= n:
            break
    return collected


# ==============================================================================
# Diagnostic 1 — EuroLeague / ACB : dump JSON field_goals + rebounds
# ==============================================================================


def run_diagnostic_euroleague_acb(league_id: int, league_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"[LIGUE] {league_name} (ID {league_id}) — 3 derniers matchs terminés")
    print("=" * 60)

    games_with_season = get_last_n_finished_games(league_id, n=3)
    if not games_with_season:
        print("Aucun match terminé trouvé.")
        return

    for g, used_season in games_with_season:
        gid = g.get("id")
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_name = home.get("name", "?") if isinstance(home, dict) else "?"
        away_name = away.get("name", "?") if isinstance(away, dict) else "?"
        match_label = f"{home_name} vs {away_name}"

        print(f"\n[MATCH] {match_label} (Game ID {gid}, season {used_season})")
        time.sleep(0.3)
        data, err = _api_get("games/statistics/teams", {"id": gid})
        if err or not data:
            print(f"  [API teams] Erreur: {err}")
            continue
        resp = data.get("response")
        if not isinstance(resp, list):
            print(f"  [API teams] response n'est pas une liste: {type(resp)}")
            continue

        for idx, team_item in enumerate(resp):
            team = team_item.get("team") or {}
            tname = team.get("name", "?") if isinstance(team, dict) else f"Team_{idx}"
            print(f"\n  --- Équipe: {tname} ---")
            fg = team_item.get("field_goals") or team_item.get("field_goals_goals")
            reb = team_item.get("rebounds") or team_item.get("rebounds_goals")
            print("  [RAW JSON] field_goals:")
            print(json.dumps(fg, indent=2, default=str))
            print("  [RAW JSON] rebounds:")
            print(json.dumps(reb, indent=2, default=str))

            # Brute force extraction
            scanned = deep_scan_team_item(team_item)
            print("  [DEEP SCAN] extrait:")
            print(json.dumps(scanned, indent=2, default=str))


# ==============================================================================
# ABA League — Team Stats API + Reconstructed from Players
# ==============================================================================


def fetch_players(team_id: int, season: str) -> List[dict]:
    for s in [season] + [x for x in SEASONS_TO_TRY if x != season]:
        data, err = _api_get("players", {"team": team_id, "season": s})
        if err or not data:
            continue
        resp = data.get("response")
        if isinstance(resp, list) and resp:
            return resp
    return []


def fetch_game_players_stats(game_id: int) -> List[dict]:
    """1 seul appel API : tous les joueurs du match. Doc: /games/statistics/players?id=game_id."""
    data, err = _api_get("games/statistics/players", {"id": game_id})
    if err or not data:
        return []
    resp = data.get("response")
    return resp if isinstance(resp, list) else []


def _int_or_total(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return max(0, val)
    if isinstance(val, dict):
        return max(0, int(val.get("total", default)))
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return default


def aggregate_team_stats_from_players(
    game_id: int, team_id: int, season: str
) -> Optional[Dict[str, Any]]:
    """
    1 appel API: /games/statistics/players?id=game_id.
    Filtre par team_id, agrège FGA, FGM, 3PM, FTA, FTM, ORB, TRB, TOV, PTS.
    """
    raw_list = fetch_game_players_stats(game_id)
    if not raw_list:
        return None

    tot_fga = tot_fgm = tot_3pm = tot_fta = tot_ftm = tot_orb = tot_trb = tot_tov = tot_pts = 0
    found_any = False

    for row in raw_list:
            t_obj = row.get("team") or {}
            row_tid = t_obj.get("id") if isinstance(t_obj, dict) else (t_obj if isinstance(t_obj, int) else None)
            if row_tid != team_id:
                continue
            found_any = True
            # Extraction brute force par stat
            pts = _int_or_total(row.get("points"), 0)
            reb = row.get("rebounds")
            if isinstance(reb, dict):
                tot_trb += _int_or_total(reb.get("total"), 0)
                tot_orb += _int_or_total(reb.get("offence") or reb.get("offensive"), 0)
            else:
                tot_trb += _int_or_total(reb, 0)
            tot_tov += _int_or_total(row.get("turnovers"), 0)
            tot_pts += pts

            fg = row.get("field_goals") or row.get("goals")
            if isinstance(fg, dict):
                tot_fga += _int_or_total(fg.get("attempted") or fg.get("attempts"), 0)
                tot_fgm += _int_or_total(fg.get("made") or fg.get("total"), 0)
            th = row.get("threepoint_goals") or row.get("three_points")
            if isinstance(th, dict):
                tot_3pm += _int_or_total(th.get("made") or th.get("total"), 0)
            ft = row.get("freethrows_goals") or row.get("free_throws") or row.get("freethrows")
            if isinstance(ft, dict):
                tot_fta += _int_or_total(ft.get("attempted") or ft.get("attempts"), 0)
                tot_ftm += _int_or_total(ft.get("made") or ft.get("total"), 0)

    if not found_any:
        return None
    poss = tot_fga + int(0.44 * tot_fta) + tot_tov - tot_orb
    if poss <= 0:
        poss = max(1, tot_fga + tot_fta)
    return {
        "FGA": tot_fga, "FGM": tot_fgm, "3PM": tot_3pm, "FTA": tot_fta, "FTM": tot_ftm,
        "ORB": tot_orb, "TRB": tot_trb, "TOV": tot_tov, "Possessions": poss, "pts_for": tot_pts,
    }


def compute_four_factors_from_raw(raw: Dict[str, Any]) -> Dict[str, str]:
    """Calcule eFG%, ORB%, Pace à afficher (ou 'N/A')."""
    out: Dict[str, str] = {"eFG%": "N/A", "ORB%": "N/A", "Pace": "N/A"}
    fga = raw.get("FGA", 0) or 0
    fgm = raw.get("FGM", 0) or 0
    thm = raw.get("3PM", 0) or 0
    orb = raw.get("ORB", 0) or 0
    poss = raw.get("Possessions", 0) or 0
    n = 1  # 1 match
    if fga > 0:
        out["eFG%"] = f"{100.0 * (fgm + 0.5 * thm) / fga:.1f}%"
    if (orb + 30.0 * n) > 0 and n:
        out["ORB%"] = f"{100.0 * orb / (orb + 30.0 * n):.1f}%"
    if poss > 0:
        out["Pace"] = f"{poss:.1f}"
    return out


def raw_team_from_team_item(team_item: dict) -> Optional[Dict[str, Any]]:
    """
    Construit {FGA, FGM, 3PM, FTA, FTM, ORB, TRB, TOV, Possessions, pts_for}
    à partir d'un item games/statistics/teams via brute_force_extract.
    """
    # Doc API: freethrows_goals, threepoint_goals, rebounds.offence
    fg = team_item.get("field_goals") or team_item.get("field_goals_goals")
    reb = team_item.get("rebounds") or team_item.get("rebounds_goals")
    th = team_item.get("threepoint_goals") or team_item.get("three_points") or team_item.get("three_points_goals")
    ft = team_item.get("freethrows_goals") or team_item.get("free_throws") or team_item.get("free_throws_goals")
    tov = team_item.get("turnovers") or team_item.get("turnovers_goals")
    pts = team_item.get("points") or team_item.get("points_goals")

    fga = brute_force_extract(fg, ["attempted", "attempts"]) if isinstance(fg, dict) else None
    fgm = brute_force_extract(fg, ["made", "total", "all"]) if isinstance(fg, dict) else None
    fta = brute_force_extract(ft, ["attempted", "attempts"]) if isinstance(ft, dict) else None
    ftm = brute_force_extract(ft, ["made", "total", "all"]) if isinstance(ft, dict) else None
    thm = brute_force_extract(th, ["made", "total", "all"]) if isinstance(th, dict) else None
    orb = brute_force_extract(reb, ["offensive", "offence"]) if isinstance(reb, dict) else None
    trb = brute_force_extract(reb, ["total", "all"]) if isinstance(reb, dict) else None
    tov_val = brute_force_extract(tov, ["total", "all"]) if isinstance(tov, dict) else _safe_int(tov)
    pts_for = brute_force_extract(pts, ["total", "all"]) if isinstance(pts, dict) else None
    if pts_for is None and (fgm or thm or ftm):
        pts_for = 2 * (fgm or 0) + (thm or 0) + (ftm or 0)

    fga = fga or 0
    fgm = fgm or 0
    fta = fta or 0
    ftm = ftm or 0
    thm = thm or 0
    orb = orb or 0
    trb = trb or 0
    tov_val = tov_val or 0
    pts_for = pts_for or 0

    if fga <= 0 and fgm <= 0 and fta <= 0:
        return None
    poss = fga + int(0.44 * fta) + tov_val - orb
    if poss <= 0:
        poss = max(1, fga + fta)
    return {
        "FGA": fga, "FGM": fgm, "3PM": thm, "FTA": fta, "FTM": ftm,
        "ORB": orb, "TRB": trb, "TOV": tov_val, "Possessions": poss, "pts_for": pts_for,
    }


def run_diagnostic_aba_poc() -> None:
    """
    PoC ABA League : dernier match (ou Dubai vs Igokea si trouvé).
    [SOURCE] Team Stats API → [SOURCE] Reconstructed from Players → [FINAL DATA].
    """
    print(f"\n{'='*60}")
    print("[ABA LEAGUE] PoC — Team Stats API vs Reconstructed from Players")
    print("=" * 60)

    games_with_season = get_last_n_finished_games(ABA_LEAGUE_ID, n=5)
    # Préférer un match contenant "Dubai" et "Igokea" si présent
    target = None
    for g, season in games_with_season:
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_name = (home.get("name") or "") if isinstance(home, dict) else ""
        away_name = (away.get("name") or "") if isinstance(away, dict) else ""
        if "dubai" in home_name.lower() or "dubai" in away_name.lower():
            if "igokea" in home_name.lower() or "igokea" in away_name.lower():
                target = (g, season)
                break
    if not target:
        target = (games_with_season[0][0], games_with_season[0][1]) if games_with_season else None
    if not target:
        print("Aucun match ABA terminé trouvé.")
        return

    g, used_season = target
    gid = g.get("id")
    teams = g.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
    away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
    home_name = home.get("name", "?") if isinstance(home, dict) else "?"
    away_name = away.get("name", "?") if isinstance(away, dict) else "?"

    print(f"\n[MATCH] {home_name} vs {away_name} (Game ID {gid})")

    # Source 1 — Team Stats API (games/statistics/teams)
    team_stats_ok = False
    raw_team: Optional[Dict[str, Any]] = None
    data, err = _api_get("games/statistics/teams", {"id": gid})
    if not err and data:
        resp = data.get("response")
        if isinstance(resp, list):
            for team_item in resp:
                raw_team = raw_team_from_team_item(team_item)
                if raw_team and (raw_team.get("FGA") or raw_team.get("pts_for")):
                    team_stats_ok = True
                    break

    print("[SOURCE] Team Stats API (games/statistics/teams): Succès" if team_stats_ok else "[SOURCE] Team Stats API (games/statistics/teams): Echec")

    # Source 2 — Reconstructed from Players
    recon_ok = False
    raw_recon: Optional[Dict[str, Any]] = None
    for tid, tname in [(home_id, home_name), (away_id, away_name)]:
        if not tid:
            continue
        raw_recon = aggregate_team_stats_from_players(gid, tid, used_season)
        if raw_recon and (raw_recon.get("FGA") or raw_recon.get("pts_for")):
            recon_ok = True
            break

    print("[SOURCE] Reconstructed from Players (games/statistics/players): Succès" if recon_ok else "[SOURCE] Reconstructed from Players (games/statistics/players): Echec")

    # Final data (priorité: Team API, sinon Reconstructed)
    final_raw = raw_team if raw_team else raw_recon
    if final_raw:
        final_display = compute_four_factors_from_raw(final_raw)
        print("[FINAL DATA] " + " | ".join(f"{k}: {v}" for k, v in final_display.items()))
    else:
        print("[FINAL DATA] Aucune donnée exploitable (eFG%: N/A, ORB%: N/A, Pace: N/A)")


# ==============================================================================
# Résultats structurés pour l'app Streamlit (Sniffer UI)
# ==============================================================================


def get_single_game_result(game_id: int) -> Dict[str, Any]:
    """
    Retourne un dict pour l'UI : game_id, error?, teams: [{ team_name, raw_field_goals,
    raw_rebounds, deep_scan, raw_team, final_data }].
    """
    out: Dict[str, Any] = {"game_id": game_id, "error": None, "match_label": None, "teams": []}
    data, err = _api_get("games/statistics/teams", {"id": game_id})
    if err:
        out["error"] = err
        return out
    if not data:
        out["error"] = "Réponse API vide."
        return out
    resp = data.get("response")
    if not isinstance(resp, list):
        out["error"] = f"response n'est pas une liste (type={type(resp).__name__})"
        return out
    if not resp:
        out["error"] = "response = [] (Box Score absent pour ce Match ID)."
        return out

    for idx, team_item in enumerate(resp):
        team = team_item.get("team") or {}
        tname = team.get("name", "?") if isinstance(team, dict) else f"Team_{idx}"
        fg = team_item.get("field_goals") or team_item.get("field_goals_goals")
        reb = team_item.get("rebounds") or team_item.get("rebounds_goals")
        raw_team = raw_team_from_team_item(team_item)
        final_data = compute_four_factors_from_raw(raw_team) if raw_team else {"eFG%": "N/A", "ORB%": "N/A", "Pace": "N/A"}
        out["teams"].append({
            "team_name": tname,
            "raw_field_goals": fg,
            "raw_rebounds": reb,
            "deep_scan": deep_scan_team_item(team_item),
            "raw_team": raw_team,
            "final_data": final_data,
        })
    return out


def get_league_sniffer_result(league_id: int, league_name: str, n: int = 3) -> Dict[str, Any]:
    """Retourne { league_name, league_id, matches: [ { match_label, game_id, season, teams: [...] } ] }."""
    out: Dict[str, Any] = {"league_name": league_name, "league_id": league_id, "matches": []}
    games_with_season = get_last_n_finished_games(league_id, n=n)
    for g, used_season in games_with_season:
        gid = g.get("id")
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_name = home.get("name", "?") if isinstance(home, dict) else "?"
        away_name = away.get("name", "?") if isinstance(away, dict) else "?"
        match_label = f"{home_name} vs {away_name}"
        time.sleep(0.3)
        data, err = _api_get("games/statistics/teams", {"id": gid})
        team_list: List[Dict[str, Any]] = []
        if not err and data:
            resp = data.get("response")
            if isinstance(resp, list):
                for team_item in resp:
                    team = team_item.get("team") or {}
                    tname = team.get("name", "?") if isinstance(team, dict) else "?"
                    fg = team_item.get("field_goals") or team_item.get("field_goals_goals")
                    reb = team_item.get("rebounds") or team_item.get("rebounds_goals")
                    raw_team = raw_team_from_team_item(team_item)
                    final_data = compute_four_factors_from_raw(raw_team) if raw_team else {"eFG%": "N/A", "ORB%": "N/A", "Pace": "N/A"}
                    team_list.append({
                        "team_name": tname,
                        "raw_field_goals": fg,
                        "raw_rebounds": reb,
                        "deep_scan": deep_scan_team_item(team_item),
                        "raw_team": raw_team,
                        "final_data": final_data,
                    })
        out["matches"].append({
            "match_label": match_label,
            "game_id": gid,
            "season": used_season,
            "teams": team_list,
            "api_error": err,
        })
    return out


def get_aba_poc_result() -> Dict[str, Any]:
    """
    Retourne { match_label, game_id, source_team_ok, source_players_ok, final_data, error? }
    pour l'affichage PoC ABA (Dubai vs Igokea ou dernier match).
    """
    out: Dict[str, Any] = {
        "match_label": None,
        "game_id": None,
        "source_team_ok": False,
        "source_players_ok": False,
        "final_data": {"eFG%": "N/A", "ORB%": "N/A", "Pace": "N/A"},
        "error": None,
    }
    games_with_season = get_last_n_finished_games(ABA_LEAGUE_ID, n=5)
    target = None
    for g, season in games_with_season:
        teams = g.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        home_name = (home.get("name") or "") if isinstance(home, dict) else ""
        away_name = (away.get("name") or "") if isinstance(away, dict) else ""
        if "dubai" in home_name.lower() or "dubai" in away_name.lower():
            if "igokea" in home_name.lower() or "igokea" in away_name.lower():
                target = (g, season)
                break
    if not target and games_with_season:
        target = (games_with_season[0][0], games_with_season[0][1])
    if not target:
        out["error"] = "Aucun match ABA terminé trouvé."
        return out

    g, used_season = target
    gid = g.get("id")
    teams = g.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_id = int(home.get("id", 0)) if isinstance(home, dict) else 0
    away_id = int(away.get("id", 0)) if isinstance(away, dict) else 0
    home_name = home.get("name", "?") if isinstance(home, dict) else "?"
    away_name = away.get("name", "?") if isinstance(away, dict) else "?"

    out["match_label"] = f"{home_name} vs {away_name}"
    out["game_id"] = gid

    raw_team: Optional[Dict[str, Any]] = None
    data, err = _api_get("games/statistics/teams", {"id": gid})
    if not err and data:
        resp = data.get("response")
        if isinstance(resp, list):
            for team_item in resp:
                raw_team = raw_team_from_team_item(team_item)
                if raw_team and (raw_team.get("FGA") or raw_team.get("pts_for")):
                    out["source_team_ok"] = True
                    break

    raw_recon: Optional[Dict[str, Any]] = None
    for tid in (home_id, away_id):
        if not tid:
            continue
        raw_recon = aggregate_team_stats_from_players(gid, tid, used_season)
        if raw_recon and (raw_recon.get("FGA") or raw_recon.get("pts_for")):
            out["source_players_ok"] = True
            break

    final_raw = raw_team if raw_team else raw_recon
    if final_raw:
        out["final_data"] = compute_four_factors_from_raw(final_raw)
    return out


def get_team_season_sniffer_result(
    team_id: int, league_id: int, season: str, team_name: Optional[str] = None, max_games: int = 50
) -> Dict[str, Any]:
    """
    Sniffer pour une équipe sur toute sa saison : tous les matchs + box score (raw + extrait).
    Retourne { team_name, team_id, league_id, season, matches: [ { game_id, date, opponent_name,
    home_away, raw_field_goals, raw_rebounds, deep_scan, raw_team, final_data, status } ] }.
    """
    out: Dict[str, Any] = {
        "team_name": team_name or "?",
        "team_id": team_id,
        "league_id": league_id,
        "season": season,
        "matches": [],
    }
    games = fetch_team_games_full(league_id, season, team_id, max_games=max_games)
    if not games:
        return out
    if team_name is None:
        # Récupérer le nom d'équipe depuis le premier match (optionnel, on peut le passer en param)
        pass  # on garde "?" ou on pourrait faire un appel teams?team=team_id
    game_ids = [g["game_id"] for g in games if g.get("game_id")]
    if not game_ids:
        return out

    # Batch : appels de 20 en 20
    batch_by_game_team: Dict[Tuple[int, int], dict] = {}
    for i in range(0, len(game_ids), 20):
        chunk = game_ids[i : i + 20]
        time.sleep(0.25)
        items = fetch_game_statistics_teams_batch(chunk, max_per_call=20)
        for item in items:
            gid = item.get("game_id") or (item.get("game") or {}).get("id")
            tid = item.get("team_id") or (item.get("team") or {}).get("id")
            if gid is not None and tid is not None:
                batch_by_game_team[(int(gid), int(tid))] = item

    for g in games:
        gid = g.get("game_id")
        if not gid:
            continue
        team_item = batch_by_game_team.get((int(gid), int(team_id)))
        raw_fg = raw_reb = None
        deep_scan = {}
        raw_team = None
        final_data = {"eFG%": "N/A", "ORB%": "N/A", "Pace": "N/A"}
        status = "vide"
        if team_item:
            raw_fg = team_item.get("field_goals") or team_item.get("field_goals_goals")
            raw_reb = team_item.get("rebounds") or team_item.get("rebounds_goals")
            deep_scan = deep_scan_team_item(team_item)
            raw_team = raw_team_from_team_item(team_item)
            if raw_team:
                final_data = compute_four_factors_from_raw(raw_team)
                status = "OK"
            else:
                status = "extraction_échouée"
        out["matches"].append({
            "game_id": gid,
            "date": (g.get("date") or "")[:10],
            "opponent_name": g.get("opponent_name", "?"),
            "home_away": g.get("home_away", "?"),
            "pts_for": g.get("pts_for"),
            "pts_against": g.get("pts_against"),
            "raw_field_goals": raw_fg,
            "raw_rebounds": raw_reb,
            "deep_scan": deep_scan,
            "raw_team": raw_team,
            "final_data": final_data,
            "status": status,
        })
    return out


# ==============================================================================
# Diagnostic Match ID unique (ex: 470712) — structure exacte du JSON
# ==============================================================================


def run_diagnostic_single_game(game_id: int) -> None:
    """
    Cible un Match ID précis. Appelle games/statistics/teams puis affiche
    le JSON brut de field_goals et rebounds + deep_scan. Utile pour 470712.
    """
    print(f"\n{'='*60}")
    print(f"[MATCH ID] {game_id} — Sniffer structure JSON")
    print("=" * 60)

    data, err = _api_get("games/statistics/teams", {"id": game_id})
    if err:
        print(f"[API] Erreur: {err}")
        return
    if not data:
        print("[API] Réponse vide.")
        return
    resp = data.get("response")
    if not isinstance(resp, list):
        print(f"[API] response type = {type(resp)}")
        print(json.dumps(data, indent=2, default=str)[:2000])
        return
    if not resp:
        print("[API] response = [] (liste vide — Box Score absent pour ce Match ID).")
        return

    for idx, team_item in enumerate(resp):
        team = team_item.get("team") or {}
        tname = team.get("name", "?") if isinstance(team, dict) else f"Team_{idx}"
        print(f"\n  --- Équipe: {tname} ---")
        fg = team_item.get("field_goals") or team_item.get("field_goals_goals")
        reb = team_item.get("rebounds") or team_item.get("rebounds_goals")
        print("  [RAW JSON] field_goals:")
        print(json.dumps(fg, indent=2, default=str))
        print("  [RAW JSON] rebounds:")
        print(json.dumps(reb, indent=2, default=str))
        scanned = deep_scan_team_item(team_item)
        print("  [DEEP SCAN] extrait:")
        print(json.dumps(scanned, indent=2, default=str))
        raw = raw_team_from_team_item(team_item)
        if raw:
            final = compute_four_factors_from_raw(raw)
            print("  [FINAL DATA] " + " | ".join(f"{k}: {v}" for k, v in final.items()))
        else:
            print("  [FINAL DATA] Extraction brute force → échec (données manquantes/invalides).")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    print("Sniffer Diagnostic — Box Scores API Basketball")
    game_id_env = os.environ.get("GAME_ID", "").strip()
    if game_id_env:
        try:
            gid = int(game_id_env)
            run_diagnostic_single_game(gid)
            return
        except ValueError:
            pass

    print("Ligues ciblées: EuroLeague (120), ACB (5), ABA (206)")

    # 1) EuroLeague — dump brut field_goals + rebounds
    run_diagnostic_euroleague_acb(EUROLEAGUE_ID, "EuroLeague")

    # 2) ACB
    run_diagnostic_euroleague_acb(ACB_ID, "Liga ACB")

    # 3) ABA — PoC Dubai vs Igokea (ou dernier match)
    run_diagnostic_aba_poc()


if __name__ == "__main__":
    main()
