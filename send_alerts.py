#!/usr/bin/env python3
"""
send_alerts.py — Envoi des alertes SNIPER TARGET via Telegram ou Discord
========================================================================
Scanne les matchs du jour, filtre les 🎯 SNIPER TARGET (Edge > 5%),
et envoie les alertes sur Telegram et/ou Discord.

Secrets GitHub requis :
  - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (optionnel)
  - DISCORD_WEBHOOK_URL (optionnel)
  - API_BASKETBALL_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (pour le scan)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


def send_telegram(text: str, bot_token: str, chat_id: str) -> bool:
    """Envoie un message sur Telegram."""
    import requests
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def send_discord(text: str, webhook_url: str) -> bool:
    """Envoie un message sur Discord via webhook."""
    import requests
    try:
        r = requests.post(
            webhook_url,
            json={"content": text},
            timeout=10,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


def main() -> None:
    print("\n" + "=" * 50)
    print("📤 SEND ALERTS — SNIPER TARGET")
    print("=" * 50)

    bot_token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    discord_webhook = (os.environ.get("DISCORD_WEBHOOK_URL") or "").strip()

    if not bot_token or not chat_id:
        print("   ⚠️ TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant.")
    if not discord_webhook:
        print("   ⚠️ DISCORD_WEBHOOK_URL manquant.")
    if (not bot_token or not chat_id) and not discord_webhook:
        print("   → Configure au moins Telegram ou Discord. Pas d'envoi.")
        return

    # Import après .env chargé
    try:
        from app_sniper_v27 import build_sniper_table
    except ImportError as e:
        print(f"   ❌ Erreur import app_sniper_v27: {e}")
        sys.exit(1)

    print("   Scan des matchs du jour...")
    df = build_sniper_table()
    if df.empty:
        print("   Aucun match trouvé.")
        return

    # Filtrer les SNIPER TARGET uniquement
    sniper = df[df["Confiance"] == "🎯 SNIPER TARGET"]
    if sniper.empty:
        msg = "✅ Aucun 🎯 SNIPER TARGET aujourd'hui — pas de mise recommandée."
        print(f"   {msg}")
    else:
        lines = ["🎯 <b>SNIPER TARGETS — Matchs du jour</b>\n"]
        for _, row in sniper.iterrows():
            match = row.get("Match", "?")
            le_pari = row.get("LE PARI", "?")
            edge = row.get("Edge", "?")
            fiab = row.get("Fiabilité", "?")
            lines.append(f"• {match}")
            lines.append(f"  → {le_pari} | Edge {edge} | Fiab {fiab}\n")
        msg = "\n".join(lines)

    # Envoi
    if bot_token and chat_id:
        ok = send_telegram(msg, bot_token, chat_id)
        print(f"   Telegram: {'✅ envoyé' if ok else '❌ échec'}")
    if discord_webhook:
        # Discord n'interprète pas HTML
        plain = msg.replace("<b>", "**").replace("</b>", "**")
        ok = send_discord(plain, discord_webhook)
        print(f"   Discord: {'✅ envoyé' if ok else '❌ échec'}")

    print("\n✅ send_alerts terminé.\n")


if __name__ == "__main__":
    main()
