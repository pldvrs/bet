#!/usr/bin/env python3
"""
Script de diagnostic Time Machine — à lancer une fois pour vérifier saison / date / box score.
Usage: python debug_data.py
Modifier TEAM_ID et TARGET_DATE ci-dessous si besoin.
"""
import requests
import json
from datetime import date, timedelta

API_KEY = "84077b8a5366ab2bbb14943e583d0ada"
HEADERS = {"x-apisports-key": API_KEY}
TEAM_ID = 2  # Monaco (exemple) — modifier si besoin
TARGET_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # J-1 par défaut


def debug():
    print(f"--- DIAGNOSTIC DATA POUR TEAM {TEAM_ID} (target_date={TARGET_DATE}) ---")

    # 1. Vérifier les Saisons dispos
    url = "https://v1.basketball.api-sports.io/games"
    params = {"team": TEAM_ID, "season": "2025-2026"}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    games = data.get("response", [])
    print(f"Matchs trouvés (Saison 2025-2026): {len(games)}")

    if not games:
        print("⚠️ Essai avec saison '2024-2025'...")
        params["season"] = "2024-2025"
        r = requests.get(url, headers=HEADERS, params=params)
        games = r.json().get("response", [])
        print(f"Matchs trouvés (Saison 2024-2025): {len(games)}")

    if not games:
        print("⚠️ Essai avec saison '2025'...")
        params["season"] = "2025"
        r = requests.get(url, headers=HEADERS, params=params)
        games = r.json().get("response", [])
        print(f"Matchs trouvés (Saison 2025): {len(games)}")

    if not games:
        print("❌ AUCUN MATCH TROUVÉ. Problème de saison.")
        return

    # 2. Vérifier le filtre de date (comparaison YYYY-MM-DD)
    print(f"\nTest Filtre Date (Target: {TARGET_DATE}) :")
    kept = []
    for g in games:
        g_date = g.get("date") or ""
        g_date_short = g_date[:10] if len(g_date) >= 10 else g_date
        before = g_date_short < TARGET_DATE[:10] if (g_date_short and TARGET_DATE) else False
        if len(kept) < 3:
            print(f"  - Match: {g_date!r} → date_short={g_date_short!r} vs target → {'GARDÉ' if before else 'JETÉ'}")
        if before and len(g_date_short) == 10:
            kept.append(g)
    print(f"Total matchs retenus pour le Time Machine (date < target) : {len(kept)}")

    if not kept:
        print("❌ LE FILTRE DE DATE VIDE TOUT.")
        return

    # 3. Vérifier le Box Score du dernier match retenu
    last_game = kept[0]
    last_game_id = last_game.get("id")
    print(f"\nTest Stats Brutes (game id={last_game_id}) :")
    url_stats = "https://v1.basketball.api-sports.io/games/statistics/teams"
    r_stats = requests.get(url_stats, headers=HEADERS, params={"id": last_game_id})
    stats = r_stats.json().get("response", [])

    if not stats:
        print("❌ PAS DE STATS DÉTAILLÉES (Box Score vide).")
    else:
        print("✅ Stats trouvées ! Exemple (premier bloc) :")
        print(json.dumps(stats[0], indent=2))


if __name__ == "__main__":
    debug()
