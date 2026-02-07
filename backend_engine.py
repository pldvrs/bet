#!/usr/bin/env python3
"""
Smart Ingestion Engine V25 FINAL
================================
- Configuration blindée (override .env)
- Smart Fetch : skip si match déjà en base (économie quota)
- Sniffer : Brute Force + Fallback Joueurs
- Logs clairs : ✅ ⏭️ ❌
"""

import os
import sys
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# --- CHARGEMENT CONFIGURATION BLINDÉ (AVANT TOUT) ---
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

API_KEY = (os.environ.get("API_BASKETBALL_KEY") or "").strip()
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

# Vérification critique : clé API
DEMO_KEY = "84077b8a5366ab2bbb14943e583d0ada"
if not API_KEY:
    print("\033[91m" + "❌ ERREUR : API_BASKETBALL_KEY vide ou absente du .env" + "\033[0m")
    print("   Remplis ton fichier .env avec ta clé personnelle.")
    sys.exit(1)
if API_KEY == DEMO_KEY or API_KEY.startswith("84077"):
    print("\033[91m" + "❌ ERREUR : Clé API de démo détectée (84077...). Utilise ta clé personnelle." + "\033[0m")
    sys.exit(1)

key_preview = f"{API_KEY[:4]}....{API_KEY[-4:]}" if len(API_KEY) >= 8 else "****"
print(f"   🔑 API Key chargée : {key_preview}")

BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

from database import get_client

supabase = get_client()
if not supabase:
    print("\033[91m" + "❌ ERREUR : Connexion Supabase impossible. Vérifie SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY." + "\033[0m")
    sys.exit(1)

# --- CONFIG ---
LEAGUES = {
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
SEASONS_TO_TRY = ["2025-2026", "2025", "2024-2025", "2024"]
STATS_RATE_LIMIT_SLEEP = 0.5


def get_seasons_for_date(year: int, month: int) -> list[str]:
    """
    Saisons basket à essayer pour une date (saison = sept à juin).
    Jan 2024 → 2023-2024 ; Sept 2024 → 2024-2025.
    """
    if month >= 9:
        return [f"{year}-{year+1}", str(year), str(year + 1)]
    elif month <= 6:
        return [f"{year-1}-{year}", str(year - 1), str(year)]
    else:
        return [f"{year-1}-{year}", f"{year}-{year+1}", str(year)]


# --- 1. SNIFFER (Brute Force + Fallback Joueurs) ---


def _safe_int(val, default=0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val.strip().replace("%", "")))
        except Exception:
            return default
    return default


def brute_force_extract(data, keys_to_try):
    """Extrait une valeur depuis un dict ou un nombre direct (API renvoie parfois un int)."""
    if data is None:
        return None
    if isinstance(data, (int, float)):
        return int(data)
    if isinstance(data, dict):
        for k in keys_to_try:
            if k in data and data[k] is not None:
                val = data[k]
                if isinstance(val, str):
                    try:
                        return int(float(val.strip().replace("%", "")))
                    except Exception:
                        pass
                if isinstance(val, (int, float)):
                    return int(val)
        for v in data.values():
            if isinstance(v, dict):
                res = brute_force_extract(v, keys_to_try)
                if res is not None:
                    return res
    return None


def fetch_players_stats_fallback(game_id, team_id, season, headers):
    """Reconstruction via somme des stats joueurs. Doc API: 1 appel /games/statistics/players?id=game_id."""
    print(f"      ☢️  FALLBACK JOUEURS: Team {team_id}...")
    try:
        r = requests.get(
            f"{BASE_URL}/games/statistics/players",
            headers=headers,
            params={"id": game_id},
            timeout=15,
        )
        time.sleep(STATS_RATE_LIMIT_SLEEP)
        all_players = r.json().get("response", [])
        if not all_players:
            return None

        total = {"pts": 0, "tov": 0, "orb": 0, "fga": 0, "fgm": 0, "fta": 0, "3pm": 0}
        found_any = False

        for stats in all_players:
            try:
                t_obj = stats.get("team") or {}
                if isinstance(t_obj, dict) and t_obj.get("id") != team_id:
                    continue
                if isinstance(t_obj, int) and t_obj != team_id:
                    continue
                found_any = True
                total["pts"] += _safe_int(stats.get("points"))
                total["tov"] += _safe_int(stats.get("turnovers"))
                reb = stats.get("rebounds")
                if isinstance(reb, dict):
                    total["orb"] += _safe_int(reb.get("offence") or reb.get("offensive") or (reb.get("total", 0) * 0.25))
                elif isinstance(reb, (int, float)):
                    total["orb"] += int(reb) * 25 // 100
                fg = stats.get("field_goals", {})
                if isinstance(fg, dict):
                    total["fga"] += _safe_int(fg.get("attempts") or fg.get("attempted"))
                    total["fgm"] += _safe_int(fg.get("total") or fg.get("made"))
                ft = stats.get("freethrows_goals") or stats.get("free_throws", {})
                if isinstance(ft, dict):
                    total["fta"] += _safe_int(ft.get("attempts") or ft.get("attempted"))
                t3 = stats.get("threepoint_goals") or stats.get("three_points", {})
                if isinstance(t3, dict):
                    total["3pm"] += _safe_int(t3.get("total") or t3.get("made"))
            except Exception as e:
                print(f"         ⚠️  Joueur ignoré: {e}")
                continue

        if not found_any:
            return None

        poss = total["fga"] + (0.44 * total["fta"]) + total["tov"] - total["orb"]
        return {
            "points": total["pts"],
            "possessions": max(1, poss),
            "orb": total["orb"],
            "tov": total["tov"],
            "fga": total["fga"],
            "fta": total["fta"],
            "fgm": total["fgm"],
            "3pm": total["3pm"],
        }
    except Exception as e:
        print(f"      ❌ Erreur Fallback: {e}")
        return None


# --- 2. VÉRIFICATION "DÉJÀ EN BASE" ---


def game_already_in_base(game_id: int) -> bool:
    """Vérifie si le match a déjà des box_scores en base (économie API)."""
    if not supabase:
        return False
    try:
        r = supabase.table("box_scores").select("game_id").eq("game_id", game_id).limit(1).execute()
        return bool(r.data and len(r.data) > 0)
    except Exception:
        return False


# --- 3. INGESTION ---


def ingest_game(game_data: dict, league_id: int, season: str) -> bool:
    """Ingère un match. Utilise game_data (pas d'appel /games)."""
    if not supabase:
        return False

    game_id = game_data.get("id")
    if not game_id:
        return False

    raw_date = game_data.get("date") or ""
    date_str = str(raw_date)[:10] if raw_date else ""

    teams = game_data.get("teams") or {}
    home_obj = teams.get("home")
    away_obj = teams.get("away")
    home_id = int(home_obj.get("id", 0)) if isinstance(home_obj, dict) else 0
    away_id = int(away_obj.get("id", 0)) if isinstance(away_obj, dict) else 0

    scores = game_data.get("scores") or {}
    sh = scores.get("home")
    sa = scores.get("away")
    home_score = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
    away_score = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)

    if not home_id or not away_id:
        print(f"      ❌ Match {game_id}: IDs home/away manquants")
        return False

    if len(date_str) != 10:
        print(f"      ❌ Match {game_id}: date invalide '{date_str}'")
        return False

    # Mise à jour des noms d'équipes depuis l'API (teams_metadata.nom_equipe) — sans écraser archétype
    home_name = (home_obj.get("name") or "").strip() if isinstance(home_obj, dict) else ""
    away_name = (away_obj.get("name") or "").strip() if isinstance(away_obj, dict) else ""
    for tid, nom in [(home_id, home_name), (away_id, away_name)]:
        if nom:
            try:
                r = supabase.table("teams_metadata").select("team_id").eq("team_id", tid).limit(1).execute()
                if r.data and len(r.data) > 0:
                    supabase.table("teams_metadata").update({"nom_equipe": nom, "name": nom}).eq("team_id", tid).execute()
                else:
                    supabase.table("teams_metadata").insert(
                        {"team_id": tid, "nom_equipe": nom, "name": nom, "league_id": league_id}
                    ).execute()
            except Exception as e:
                err = str(e) if hasattr(e, "__str__") else repr(e)
                if "nom_equipe" in err and ("PGRST204" in err or "schema" in err.lower()):
                    print("\033[93m" + "⚠️  Exécute schema_migration_nom_equipe.sql pour activer les noms d'équipes." + "\033[0m")
                pass

    supabase.table("games_history").upsert(
        {
            "game_id": game_id,
            "date": date_str,
            "league_id": league_id,
            "season": season,
            "home_id": home_id,
            "away_id": away_id,
            "home_score": home_score,
            "away_score": away_score,
        },
        on_conflict="game_id",
    ).execute()

    r_stats = requests.get(
        f"{BASE_URL}/games/statistics/teams",
        headers=HEADERS,
        params={"id": game_id},
        timeout=15,
    )
    time.sleep(STATS_RATE_LIMIT_SLEEP)
    s_resp = r_stats.json().get("response", [])

    if not s_resp:
        s_resp = [{"team": {"id": home_id}}, {"team": {"id": away_id}}]

    for item in s_resp:
        team_obj = item.get("team") or {}
        tid = team_obj.get("id") if isinstance(team_obj, dict) else None
        if not tid:
            continue
        is_home = tid == home_id
        opp_id = away_id if is_home else home_id

        fg = item.get("field_goals")
        fga = brute_force_extract(fg, ["attempts", "attempted"])
        fgm = brute_force_extract(fg, ["made", "total"])
        ft = item.get("freethrows_goals") or item.get("free_throws")
        fta = brute_force_extract(ft, ["attempts", "attempted"])
        tov = brute_force_extract(item.get("turnovers"), ["total"])
        orb = brute_force_extract(item.get("rebounds"), ["offence", "offensive"])
        pts = home_score if is_home else away_score
        t3pm = brute_force_extract(item.get("threepoint_goals") or item.get("three_points"), ["total", "made"])

        poss = 0
        valid = True

        if not fga or fga == 0:
            recon = fetch_players_stats_fallback(game_id, tid, season, HEADERS)
            if recon:
                fga = recon["fga"]
                fgm = recon["fgm"]
                fta = recon["fta"]
                tov = recon["tov"]
                orb = recon["orb"]
                pts = recon["points"]
                t3pm = recon["3pm"]
                poss = recon["possessions"]
            else:
                valid = False

        if valid:
            fga = fga or 0
            fgm = fgm or 0
            fta = fta or 0
            tov = tov or 0
            orb = orb or 0
            t3pm = t3pm or 0
            pts = pts or 0
            if poss == 0:
                poss = (fga or 0) + (0.44 * (fta or 0)) + (tov or 0) - (orb or 0)
                if poss <= 0:
                    poss = 70.0

            poss = poss or 70.0
            off_rtg = ((pts or 0) / poss) * 100
            opp_score = away_score if is_home else home_score
            def_rtg = ((opp_score or 0) / poss) * 100

            efg = ((fgm or 0) + 0.5 * (t3pm or 0)) / max(1, fga or 1) if (fga or 0) > 0 else 0.5
            orb_pct_val = (orb or 0) / max(1, (orb or 0) + 30) if orb else 0.25
            tov_pct_val = (tov or 0) / max(1, poss) if tov else 0.15
            ft_rate_val = (fta or 0) / max(1, fga or 1) if fga else 0.25
            three_rate_val = (t3pm or 0) / max(1, fga or 1) if fga else 0.0

            box_row = {
                "game_id": game_id,
                "team_id": tid,
                "opponent_id": opp_id,
                "is_home": is_home,
                "date": date_str,
                "pace": poss,
                "off_rtg": off_rtg,
                "def_rtg": def_rtg,
                "efg_pct": efg,
                "orb_pct": orb_pct_val,
                "tov_pct": tov_pct_val,
                "ft_rate": ft_rate_val,
                "three_rate": three_rate_val,
            }
            try:
                supabase.table("box_scores").upsert(box_row, on_conflict="game_id,team_id").execute()
                print(f"      ✅ Stats Team {tid} sauvegardées")
            except Exception as e:
                err_msg = str(e) if hasattr(e, "__str__") else repr(e)
                if "three_rate" in err_msg and ("PGRST204" in err_msg or "schema cache" in err_msg.lower()):
                    print("\033[91m" + "❌ ERREUR : Colonne 'three_rate' manquante en base." + "\033[0m")
                    print("   Exécute dans Supabase → SQL Editor :")
                    print("   ALTER TABLE box_scores ADD COLUMN IF NOT EXISTS three_rate FLOAT;")
                    raise SystemExit(1) from e
                if "date" in err_msg and ("PGRST204" in err_msg or "schema cache" in err_msg.lower()):
                    print("\033[91m" + "❌ ERREUR : Colonne 'date' manquante dans box_scores." + "\033[0m")
                    print("   Exécute schema_migration_date_reset.sql dans Supabase.")
                    raise SystemExit(1) from e
                raise

    return True


# --- 4. CALENDRIER FUTUR ---


def fetch_future_games(days: int = 7, max_games: int = 200) -> int:
    """
    Récupère le calendrier des N prochains jours via l'API /games.
    Insère les matchs à venir dans games_history (home_score=null, away_score=null).
    Ignore les matchs déjà en base avec des scores (terminés).
    """
    if not supabase:
        return 0

    today = datetime.now().date()
    ingested = 0
    skipped = 0
    seen_gids = set()

    print(f"\n📅 CALENDRIER FUTUR — {days} prochains jours\n")

    for day_offset in range(days):
        d = today + timedelta(days=day_offset)
        date_str = d.strftime("%Y-%m-%d")
        print(f"   Date : {date_str}")

        for league_name, league_id in LEAGUES.items():
            if ingested >= max_games:
                break
            for season in get_seasons_for_date(d.year, d.month):
                try:
                    r = requests.get(
                        f"{BASE_URL}/games",
                        headers=HEADERS,
                        params={"date": date_str, "league": league_id, "season": season},
                        timeout=15,
                    )
                    time.sleep(STATS_RATE_LIMIT_SLEEP * 0.5)
                    data = r.json()

                    if data.get("errors"):
                        errs = data["errors"]
                        if "rate" in str(errs).lower() or "limit" in str(errs).lower():
                            print(f"   🛑 Rate Limit API atteint.")
                            return ingested
                        continue

                    games = data.get("response", [])
                    if not games:
                        continue

                    for g in games:
                        if ingested >= max_games:
                            break
                        gid = g.get("id")
                        if not gid or gid in seen_gids:
                            continue

                        status = g.get("status") or {}
                        status_short = (status.get("short") or "") if isinstance(status, dict) else str(status)
                        # Ne pas écraser un match terminé (FT, AOT...)
                        if status_short in ("FT", "AOT", "AOT1", "AOT2"):
                            continue

                        seen_gids.add(gid)

                        teams = g.get("teams") or {}
                        home_obj = teams.get("home") or {}
                        away_obj = teams.get("away") or {}
                        home_id = int(home_obj.get("id", 0)) if isinstance(home_obj, dict) else 0
                        away_id = int(away_obj.get("id", 0)) if isinstance(away_obj, dict) else 0
                        if not home_id or not away_id:
                            continue

                        # Vérifier si match déjà en base avec scores (terminé)
                        try:
                            r2 = supabase.table("games_history").select("home_score, away_score").eq("game_id", gid).limit(1).execute()
                            if r2.data and len(r2.data) > 0:
                                row = r2.data[0]
                                if row.get("home_score") is not None or row.get("away_score") is not None:
                                    skipped += 1
                                    continue
                        except Exception:
                            pass

                        home_name = (home_obj.get("name") or "").strip() if isinstance(home_obj, dict) else ""
                        away_name = (away_obj.get("name") or "").strip() if isinstance(away_obj, dict) else ""

                        try:
                            supabase.table("games_history").upsert(
                                {
                                    "game_id": gid,
                                    "date": date_str,
                                    "league_id": league_id,
                                    "season": season,
                                    "home_id": home_id,
                                    "away_id": away_id,
                                    "home_score": None,
                                    "away_score": None,
                                },
                                on_conflict="game_id",
                            ).execute()

                            for tid, nom in [(home_id, home_name), (away_id, away_name)]:
                                if nom:
                                    try:
                                        r3 = supabase.table("teams_metadata").select("team_id").eq("team_id", tid).limit(1).execute()
                                        if r3.data and len(r3.data) > 0:
                                            supabase.table("teams_metadata").update({"nom_equipe": nom, "name": nom}).eq("team_id", tid).execute()
                                        else:
                                            supabase.table("teams_metadata").insert(
                                                {"team_id": tid, "nom_equipe": nom, "name": nom, "league_id": league_id}
                                            ).execute()
                                    except Exception:
                                        pass

                            ingested += 1
                            print(f"      ✅ Match {gid} : {home_name or home_id} vs {away_name or away_id}")
                        except Exception as e:
                            print(f"      ⚠️ Match {gid} : {e}")

                except Exception as e:
                    print(f"   ❌ Erreur {league_name}: {e}")
                    continue
                break

    print(f"\n📊 Calendrier : {ingested} matchs à venir ajoutés | {skipped} déjà terminés")
    return ingested


# --- 5. SMART FETCH (Passé) ---


def ingest_recent_games(days: int = 3, max_games_per_run: int = 30) -> int:
    """Récupère les matchs des X derniers jours. Skip si déjà en base."""
    if not supabase:
        return 0

    today = datetime.now().date()
    ingested = 0
    skipped = 0
    seen_gids = set()

    print(f"\n🔎 Recherche sur {days} jours...\n")

    for day_offset in range(days):
        d = today - timedelta(days=day_offset)
        date_str = d.strftime("%Y-%m-%d")
        print(f"📅 DATE : {date_str}")

        for league_name, league_id in LEAGUES.items():
            if ingested >= max_games_per_run:
                break
            for season in get_seasons_for_date(d.year, d.month):
                try:
                    r = requests.get(
                        f"{BASE_URL}/games",
                        headers=HEADERS,
                        params={"date": date_str, "league": league_id, "season": season},
                        timeout=15,
                    )
                    data = r.json()

                    if data.get("errors"):
                        errs = data["errors"]
                        if "rate" in str(errs).lower() or "limit" in str(errs).lower():
                            print(f"   🛑 Rate Limit API atteint.")
                            return ingested
                        print(f"   ⚠️  API: {errs}")

                    games = data.get("response", [])
                    if not games:
                        continue

                    print(f"   🎯 {league_name} ({season}): {len(games)} matchs")

                    for g in games:
                        if ingested >= max_games_per_run:
                            break
                        gid = g.get("id")
                        if not gid or gid in seen_gids:
                            continue
                        status = g.get("status") or {}
                        status_short = status.get("short", "") if isinstance(status, dict) else str(status)
                        if status_short not in ("FT", "AOT", "AOT1", "AOT2"):
                            continue
                        scores = g.get("scores") or {}
                        sh, sa = scores.get("home"), scores.get("away")
                        pts_h = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
                        pts_a = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)
                        if pts_h == 0 and pts_a == 0:
                            continue
                        seen_gids.add(gid)

                        if game_already_in_base(gid):
                            print(f"   ⏭️  Match {gid} déjà en base, on passe.")
                            skipped += 1
                            continue

                        print(f"   📥 Match {gid} → ingestion...")
                        if ingest_game(g, league_id, season):
                            ingested += 1

                except Exception as e:
                    print(f"   ❌ Erreur {league_name}: {e}")
                    continue
                break

    print(f"\n📊 Bilan : {ingested} ingérés | {skipped} passés (déjà en base)")
    return ingested


def _season_start_date(season: str) -> datetime | None:
    """Retourne la date de début de saison."""
    if not season:
        return None
    s = str(season).strip()
    if "-" in s:
        parts = s.split("-")
        try:
            y1 = int(parts[0])
            return datetime(y1, 9, 1)
        except (ValueError, IndexError):
            return datetime(2025, 9, 1)
    try:
        return datetime(int(s), 1, 1)
    except ValueError:
        return datetime(2025, 9, 1)


def ingest_from_season_start(season: str = "2025-2026", max_games: int = 500) -> int:
    """Ingère depuis le début de la saison jusqu'à aujourd'hui."""
    if not supabase:
        return 0
    start = _season_start_date(season)
    if not start:
        print("   ❌ Saison invalide.")
        return 0
    today = datetime.now().date()
    start_date = start.date()
    if start_date >= today:
        print(f"   ⚠️  Saison {season} pas encore commencée.")
        return 0
    delta = (today - start_date).days
    total_days = min(delta, 200)
    print(f"\n🔎 Saison {season} : du {start_date} à aujourd'hui ({total_days} jours)\n")
    ingested = 0
    skipped = 0
    seen_gids = set()
    for day_offset in range(total_days - 1, -1, -1):
        d = today - timedelta(days=day_offset)
        date_str = d.strftime("%Y-%m-%d")
        if ingested >= max_games:
            break
        print(f"📅 DATE : {date_str}")
        for league_name, league_id in LEAGUES.items():
            if ingested >= max_games:
                break
            for s in get_seasons_for_date(d.year, d.month):
                try:
                    r = requests.get(
                        f"{BASE_URL}/games",
                        headers=HEADERS,
                        params={"date": date_str, "league": league_id, "season": s},
                        timeout=15,
                    )
                    data = r.json()
                    if data.get("errors"):
                        errs = data["errors"]
                        if "rate" in str(errs).lower() or "limit" in str(errs).lower():
                            print(f"   🛑 Rate Limit API atteint.")
                            return ingested
                    games = data.get("response", [])
                    if not games:
                        continue
                    print(f"   🎯 {league_name} ({s}): {len(games)} matchs")
                    for g in games:
                        if ingested >= max_games:
                            break
                        gid = g.get("id")
                        if not gid or gid in seen_gids:
                            continue
                        status = g.get("status") or {}
                        status_short = status.get("short", "") if isinstance(status, dict) else str(status)
                        if status_short not in ("FT", "AOT", "AOT1", "AOT2"):
                            continue
                        scores = g.get("scores") or {}
                        sh, sa = scores.get("home"), scores.get("away")
                        pts_h = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
                        pts_a = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)
                        if pts_h == 0 and pts_a == 0:
                            continue
                        seen_gids.add(gid)
                        if game_already_in_base(gid):
                            skipped += 1
                            continue
                        print(f"   📥 Match {gid} → ingestion...")
                        if ingest_game(g, league_id, s):
                            ingested += 1
                    break
                except Exception as e:
                    print(f"   ❌ Erreur {league_name}: {e}")
                    continue
    print(f"\n📊 Bilan : {ingested} ingérés | {skipped} passés (déjà en base)")
    return ingested


def ingest_month(year: int, month: int, max_games: int = 500) -> int:
    """Ingère tous les matchs d'un mois donné."""
    if not supabase:
        return 0
    from calendar import monthrange

    _, last_day = monthrange(year, month)
    print(f"\n🔎 Mois {year}-{month:02d} : du 01 au {last_day}\n")
    ingested = 0
    skipped = 0
    seen_gids = set()

    for day in range(1, last_day + 1):
        if ingested >= max_games:
            break
        date_str = f"{year}-{month:02d}-{day:02d}"
        print(f"📅 DATE : {date_str}")

        for league_name, league_id in LEAGUES.items():
            if ingested >= max_games:
                break
            seasons_to_try = get_seasons_for_date(year, month)
            for s in seasons_to_try:
                try:
                    r = requests.get(
                        f"{BASE_URL}/games",
                        headers=HEADERS,
                        params={"date": date_str, "league": league_id, "season": s},
                        timeout=15,
                    )
                    data = r.json()
                    if data.get("errors"):
                        errs = data["errors"]
                        if "rate" in str(errs).lower() or "limit" in str(errs).lower():
                            print(f"   🛑 Rate Limit API atteint. Relance plus tard.")
                            print(f"\n📊 Bilan partiel : {ingested} ingérés | {skipped} passés")
                            return ingested
                    games = data.get("response", [])
                    if not games:
                        continue
                    print(f"   🎯 {league_name} ({s}): {len(games)} matchs")
                    for g in games:
                        if ingested >= max_games:
                            break
                        gid = g.get("id")
                        if not gid or gid in seen_gids:
                            continue
                        status = g.get("status") or {}
                        status_short = status.get("short", "") if isinstance(status, dict) else str(status)
                        if status_short not in ("FT", "AOT", "AOT1", "AOT2"):
                            continue
                        scores = g.get("scores") or {}
                        sh, sa = scores.get("home"), scores.get("away")
                        pts_h = _safe_int(sh.get("total") if isinstance(sh, dict) else sh)
                        pts_a = _safe_int(sa.get("total") if isinstance(sa, dict) else sa)
                        if pts_h == 0 and pts_a == 0:
                            continue
                        seen_gids.add(gid)
                        if game_already_in_base(gid):
                            skipped += 1
                            continue
                        print(f"   📥 Match {gid} → ingestion...")
                        if ingest_game(g, league_id, s):
                            ingested += 1
                    break  # au moins une saison essayée pour cette ligue/date
                except Exception as e:
                    print(f"   ❌ Erreur {league_name}: {e}")
                    continue

    print(f"\n📊 Bilan : {ingested} ingérés | {skipped} passés (déjà en base)")
    return ingested


# --- 5. ARCHETYPES ---


def _get_league_id_for_team(tid: int) -> int | None:
    """Récupère league_id via box_scores → games_history."""
    try:
        box = (
            supabase.table("box_scores")
            .select("game_id")
            .eq("team_id", tid)
            .limit(1)
            .execute()
            .data
        )
        if not box:
            return None
        gh = (
            supabase.table("games_history")
            .select("league_id")
            .eq("game_id", box[0]["game_id"])
            .limit(1)
            .execute()
            .data
        )
        if gh and gh[0].get("league_id") is not None:
            return int(gh[0]["league_id"])
    except Exception:
        pass
    return None


def update_team_archetypes():
    """Met à jour current_archetype et profile_vector dans teams_metadata."""
    if not supabase:
        return
    print("\n🔄 Mise à jour des archétypes...")
    ok = 0
    skipped = 0
    try:
        teams = supabase.table("box_scores").select("team_id").execute().data
        if not teams:
            print("   ⚠️  Aucun box_score en base.")
            return

        unique_teams = list(set(t["team_id"] for t in teams))
        print(f"   Analyse de {len(unique_teams)} équipes...")

        for tid in unique_teams:
            matches = (
                supabase.table("box_scores")
                .select("*")
                .eq("team_id", tid)
                .order("game_id", desc=True)
                .limit(10)
                .execute()
                .data
            )
            if not matches:
                continue

            league_id = _get_league_id_for_team(tid)
            if league_id is None:
                skipped += 1
                continue

            avg_pace = float(np.mean([m.get("pace") or 72 for m in matches]))
            avg_3pr = float(np.mean([m.get("three_rate") or 0 for m in matches]))
            avg_orb = float(np.mean([m.get("orb_pct") or 0.25 for m in matches]))
            avg_def = float(np.mean([m.get("def_rtg") or 100 for m in matches]))

            atype = "Balanced"
            if avg_pace > 74 and avg_3pr > 0.40:
                atype = "Pace & Space 🚀"
            elif avg_pace < 70 and avg_def < 110:
                atype = "Grit & Grind 🛡️"
            elif avg_orb > 0.30:
                atype = "Paint Beast 💪"

            supabase.table("teams_metadata").upsert(
                {
                    "team_id": tid,
                    "name": f"Équipe {tid}",
                    "league_id": league_id,
                    "current_archetype": atype,
                    "profile_vector": {"pace": avg_pace, "def": avg_def, "orb": avg_orb},
                },
                on_conflict="team_id",
            ).execute()
            ok += 1

        print(f"   ✅ Archétypes mis à jour ({ok} équipes).")
        if skipped:
            print(f"   ⚠️  {skipped} équipe(s) ignorée(s) (league_id introuvable).")
    except Exception as e:
        print(f"   ❌ Erreur archetypes: {e}")


# --- MAIN ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion Basketball → Supabase")
    parser.add_argument(
        "--from-season",
        type=str,
        default=None,
        metavar="SEASON",
        help="Ingérer depuis début de saison (ex: 2025-2026)",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        metavar="YYYY-MM",
        help="Ingérer un mois (ex: 2026-01)",
    )
    parser.add_argument(
        "--months",
        type=str,
        nargs="+",
        default=None,
        metavar=("YYYY-MM", "YYYY-MM"),
        help="Ingérer plusieurs mois (ex: --months 2026-01 2025-12 2025-11 2025-10)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        metavar=("YYYY", "YYYY"),
        help="Ingérer des années complètes (ex: --years 2024 2023 2022 2021)",
    )
    parser.add_argument("--days", type=int, default=3, help="Jours à couvrir")
    parser.add_argument("--max-games", type=int, default=500, help="Max matchs par run")
    parser.add_argument("--skip-archetypes", action="store_true", help="Ne pas recalculer les archétypes")
    parser.add_argument("--future-only", action="store_true", help="Uniquement le calendrier futur (7 jours)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("🚀 SMART INGESTION ENGINE V25 FINAL")
    print("=" * 50)

    if args.future_only:
        fetch_future_games(days=7, max_games=200)
    elif args.years:
        total = 0
        max_per_month = max(args.max_games, 500)
        for year in sorted(args.years, reverse=True):
            print(f"\n📅 ANNÉE {year}")
            for month in range(1, 13):
                total += ingest_month(year=year, month=month, max_games=max_per_month)
        print(f"\n📊 Total ingéré : {total} matchs")
    elif args.months:
        total = 0
        max_per_month = max(args.max_games, 500)
        for month_str in args.months:
            try:
                parts = month_str.strip().split("-")
                y, m = int(parts[0]), int(parts[1])
                total += ingest_month(year=y, month=m, max_games=max_per_month)
            except (ValueError, IndexError):
                print(f"   ⚠️ Format invalide : {month_str}, ignoré.")
        print(f"\n📊 Total ingéré : {total} matchs")
    elif args.month:
        try:
            parts = args.month.split("-")
            y, m = int(parts[0]), int(parts[1])
            ingest_month(year=y, month=m, max_games=args.max_games)
        except (ValueError, IndexError):
            print(f"❌ Format invalide : {args.month}. Utilise YYYY-MM")
    elif args.from_season:
        ingest_from_season_start(season=args.from_season, max_games=args.max_games)
    else:
        ingest_recent_games(days=args.days, max_games_per_run=min(args.max_games, 100))
        fetch_future_games(days=7, max_games=200)

    if not args.skip_archetypes:
        update_team_archetypes()
    print("\n✅ Fin du script.\n")
