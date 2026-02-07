#!/usr/bin/env python3
"""
bot_ingest_data.py — Robot d'ingestion Offline-First
=====================================================
Récupère les matchs des 3 prochains jours, lance les modèles ML (Vainqueur, Spread, Totals),
récupère les cotes via API-Sports, et insère/met à jour daily_projections.

À lancer par CRON ou GitHub Actions (ex: 08h15 et 18h00).
"""

import logging
import re
import sys
import warnings

# Suppression des warnings Streamlit (No runtime / ScriptRunContext) quand le bot tourne hors Streamlit
logging.getLogger("streamlit").setLevel(logging.ERROR)
for _mod in ("streamlit.runtime.caching", "streamlit.runtime.scriptrunner_utils"):
    logging.getLogger(_mod).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="No runtime found")
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# Le bot alimente le cache — il doit toujours faire API + ML (jamais lire daily_projections)
os.environ["SNIPER_OFFLINE"] = "0"


def _parse_cotes(cotes_str: str) -> tuple:
    """Parse 'x.xx | y.yy' → (odds_home, odds_away)."""
    if not cotes_str or "|" not in cotes_str:
        return None, None
    parts = cotes_str.split("|")
    if len(parts) != 2:
        return None, None
    try:
        oh = float(parts[0].strip())
        oa = float(parts[1].strip())
        return oh, oa
    except ValueError:
        return None, None


def _parse_line_book(s: str) -> Optional[float]:
    """Parse '156.5' or 'En attente' → float or None."""
    if not s or "En attente" in s:
        return None
    m = re.search(r"[\d]+[.,]?\d*", str(s))
    if m:
        return float(m.group(0).replace(",", "."))
    return None


def _parse_diff(s: str) -> Optional[float]:
    """Parse '+7.5 pts' or '-3.2 pts' → float."""
    if not s or "—" in s:
        return None
    m = re.search(r"([+-]?[\d]+[.,]?\d*)", str(s))
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _parse_proba_pct(s: str) -> Optional[float]:
    """Parse '65.2%' → 0.652."""
    if not s or "—" in str(s):
        return None
    m = re.search(r"[\d]+[.,]?\d*", str(s))
    if m:
        return float(m.group(0).replace(",", ".")) / 100.0
    return None


def run_ingestion() -> int:
    """
    Lance le pipeline : build_sniper_table → build_sniper_totals_table → upsert daily_projections.
    Retourne le nombre de lignes insérées/mises à jour.
    """
    try:
        from app_sniper_v27 import (
            build_sniper_table,
            build_sniper_totals_table,
            _get_supabase,
        )
    except ImportError as e:
        print(f"❌ Erreur import app_sniper_v27: {e}")
        return 0

    supabase = _get_supabase()
    if not supabase:
        print("❌ Supabase indisponible.")
        return 0

    print("   Récupération des matchs + ML + cotes...")
    df = build_sniper_table()
    if df.empty:
        print("   Aucun match trouvé.")
        return 0

    print("   Récupération des cotes Over/Under...")
    df_totals = build_sniper_totals_table(df)

    # Merge totals sur Match
    if not df_totals.empty and "Match" in df_totals.columns:
        df = df.merge(
            df_totals[["Match", "LIGNE BOOK", "PROJETÉ SNIPER", "DIFF", "PARI TOTAL", "CONFIANCE"]],
            on="Match",
            how="left",
            suffixes=("", "_tot"),
        )
    else:
        df["LIGNE BOOK"] = "En attente"
        df["PROJETÉ SNIPER"] = df["_ml_total_predicted"]
        df["DIFF"] = "—"
        df["PARI TOTAL"] = "En attente"
        df["CONFIANCE"] = "—"

    now = datetime.utcnow().isoformat()
    upserted = 0

    for _, row in df.iterrows():
        gid = row.get("_game_id")
        if not gid:
            continue

        oh, oa = _parse_cotes(row.get("COTES (H/A)", ""))
        line_book = _parse_line_book(row.get("LIGNE BOOK", ""))
        diff_total = _parse_diff(row.get("DIFF", ""))
        proj_total = row.get("_ml_total_predicted") or (row.get("_proj_home", 75) + row.get("_proj_away", 75))

        game_date_str = str(row.get("_date", ""))[:10] if row.get("_date") else None
        payload: Dict[str, Any] = {
            "game_id": gid,
            "match_name": row.get("Match", ""),
            "date": game_date_str,
            "time": None,
            "jour": row.get("Jour"),
            "league_id": row.get("_league_id"),
            "season": row.get("_season"),
            "home_id": row.get("_home_id"),
            "away_id": row.get("_away_id"),
            "proba_ml": float(row.get("_prob_home", 0.5)),
            "proba_calibree": _parse_proba_pct(row.get("Proba calibrée", "")) or float(row.get("_prob_home", 0.5)),
            "edge_percent": float(row.get("_edge", 0)),
            "brain_used": row.get("Cerveau utilisé", ""),
            "confiance_label": row.get("Confiance", ""),
            "le_pari": row.get("LE PARI", ""),
            "pari_outsider": row.get("🎯 PARI OUTSIDER", ""),
            "alerte_trappe": row.get("🚨 ALERTE TRAPPE", ""),
            "message_contexte": (row.get("Message de Contexte", "") or "")[:500],
            "fiabilite": float(str(row.get("Fiabilité", "50")).replace("%", "")) if row.get("Fiabilité") else 50.0,
            "predicted_total": float(proj_total) if proj_total is not None else None,
            "line_bookmaker": line_book,
            "diff_total": diff_total,
            "pari_total": row.get("PARI TOTAL", ""),
            "confiance_ou": row.get("CONFIANCE", ""),
            "style_match": row.get("Style de Match", ""),
            "odds_home": oh,
            "odds_away": oa,
            "updated_at": now,
        }

        try:
            supabase.table("daily_projections").upsert(payload, on_conflict="game_id").execute()
            upserted += 1
        except Exception as e:
            print(f"   ⚠️ game_id {gid}: {e}")

    return upserted


def main() -> None:
    print("\n" + "=" * 50)
    print("🤖 BOT INGEST — Offline-First pipeline")
    print("=" * 50)

    n = run_ingestion()
    print(f"\n✅ {n} projection(s) insérées/mises à jour dans daily_projections.")
    print()


if __name__ == "__main__":
    main()
    sys.exit(0)
