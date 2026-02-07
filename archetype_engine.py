#!/usr/bin/env python3
"""
Archetype Engine — Profilage DNA des équipes
============================================
Pondération temporelle (5 derniers = 60%, reste = 40% baseline)
Matrice 0-100 percentiles vs ligue
Détection Style Shifts + Kryptonites (Matchup Clash)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from database import get_client

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

supabase = get_client()


def to_native(obj):
    """Convertit récursivement les types NumPy en types Python natifs (int, float) pour JSON."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(i) for i in obj]
    return obj
if not supabase:
    print("❌ Connexion Supabase impossible.")
    sys.exit(1)

# --- CONFIG ---
WEIGHT_RECENT = 0.60  # 5 derniers matchs
WEIGHT_BASELINE = 0.40  # reste de la saison
N_RECENT = 5
EFG_STD_THRESHOLD = 0.08  # au-dessus = nuance "Instable"
STYLE_SHIFT_ECART = 0.15  # écart récent vs baseline > 15% = Style Shift


def _log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "📊", "OK": "✅", "WARN": "⚠️", "ERR": "❌", "ALERT": "🔴"}
    icon = icons.get(level, "•")
    print(f"   [{ts}] {icon} {msg}")


# --- 1. EXTRACTION SUPABASE ---


def extract_box_scores_with_meta() -> pd.DataFrame:
    """
    Récupère box_scores + date + league_id via games_history.
    Retourne un DataFrame avec game_id, team_id, date, league_id, pace, off_rtg, def_rtg, efg_pct, orb_pct, ft_rate, three_rate, etc.
    """
    _log("Extraction box_scores...", "INFO")
    try:
        box = supabase.table("box_scores").select("*").execute().data
        gh = supabase.table("games_history").select("game_id, date, league_id").execute().data
    except Exception as e:
        _log(f"Erreur extraction: {e}", "ERR")
        return pd.DataFrame()

    if not box:
        _log("Aucun box_score en base.", "WARN")
        return pd.DataFrame()

    df_box = pd.DataFrame(box)
    df_gh = pd.DataFrame(gh)
    df = df_box.merge(df_gh, on="game_id", how="left")
    # date : merge peut créer date_x (box_scores) et date_y (games_history)
    date_col = (
        df["date_x"].combine_first(df["date_y"])
        if "date_x" in df.columns and "date_y" in df.columns
        else df.get("date_y", df.get("date", df.get("date_x")))
    )
    df["date"] = pd.to_datetime(date_col, errors="coerce")
    df = df.sort_values(["team_id", "date"], ascending=[True, False])
    df = df.dropna(subset=["date"])

    _log(f"{len(df)} lignes box_scores extraites.", "OK")
    return df


# --- 2. PONDÉRATION TEMPORELLE ---


def compute_weighted_profile(matches: pd.DataFrame) -> dict[str, Any]:
    """
    Calcule le profil pondéré : 60% sur les 5 derniers matchs, 40% sur le reste.
    Retourne {pace, off_rtg, def_rtg, efg_pct, orb_pct, ft_rate, three_rate, efg_std, n_recent, n_baseline}.
    """
    if matches.empty:
        return {}

    metrics = ["pace", "off_rtg", "def_rtg", "efg_pct", "orb_pct", "ft_rate", "three_rate"]
    out: dict[str, Any] = {}

    recent = matches.head(N_RECENT)
    baseline = matches.iloc[N_RECENT:]

    def _mean(df: pd.DataFrame, col: str) -> float:
        v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        return float(v.mean()) if len(v) > 0 else np.nan

    for col in metrics:
        if col not in matches.columns:
            continue
        r_val = _mean(recent, col)
        b_val = _mean(baseline, col) if len(baseline) > 0 else r_val
        if pd.isna(r_val) and pd.isna(b_val):
            out[col] = 0.0
        elif pd.isna(b_val):
            out[col] = float(r_val)
        elif pd.isna(r_val):
            out[col] = float(b_val)
        else:
            out[col] = float(WEIGHT_RECENT * r_val + WEIGHT_BASELINE * b_val)

    # Stabilité : std efg_pct
    efg_vals = matches["efg_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    out["efg_std"] = float(efg_vals.std()) if len(efg_vals) > 1 else 0.0
    out["n_recent"] = len(recent)
    out["n_baseline"] = len(baseline)

    # Pour détection Style Shift : comparaison récent vs baseline
    out["efg_recent"] = _mean(recent, "efg_pct")
    out["efg_baseline"] = _mean(baseline, "efg_pct") if len(baseline) > 0 else out["efg_recent"]
    out["pace_recent"] = _mean(recent, "pace")
    out["pace_baseline"] = _mean(baseline, "pace") if len(baseline) > 0 else out.get("pace", 70)

    return out


# --- 3. MATRICE 0-100 PERCENTILES vs LIGUE ---


def compute_league_stats(df: pd.DataFrame) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Calcule moyennes et écarts-types par league_id pour chaque métrique.
    Retourne {league_id: {metric: (mean, std), ...}}.
    """
    metrics = ["three_rate", "orb_pct", "ft_rate", "pace", "def_rtg"]
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    if "league_id" not in df.columns:
        return out
    for lid, grp in df.groupby("league_id", dropna=False):
        lid = int(lid) if pd.notna(lid) else 0
        if lid == 0:
            continue
        out[lid] = {}
        for c in metrics:
            if c not in grp.columns:
                continue
            v = grp[c].replace([np.inf, -np.inf], np.nan).dropna()
            mean_val = float(v.mean()) if len(v) > 0 else 0.0
            std_val = float(v.std()) if len(v) > 1 else 0.01
            out[lid][c] = (mean_val, std_val)
    return out


def value_to_percentile(val: float, mean: float, std: float, higher_better: bool = True) -> float:
    """
    Transforme une valeur en percentile 0-100 par rapport à la distribution ligue.
    Si std=0, retourne 50.
    """
    if pd.isna(val) or pd.isna(mean):
        return 50.0
    if std is None or std == 0:
        return 50.0
    z = (val - mean) / std
    if not higher_better:
        z = -z  # def_rtg : plus bas = meilleur
    # z ~ N(0,1) → percentile approx
    from scipy import stats

    pct = float(stats.norm.cdf(z) * 100)
    return max(0.0, min(100.0, pct))


def build_profile_vector(
    weighted: dict[str, Any],
    league_mean: dict[str, float],
    league_std: dict[str, float],
) -> dict[str, Any]:
    """
    Construit le profile_vector avec scores 0-100.
    Volume Extérieur (three_rate), Pression Intérieure (orb+ft), Contrôle Tempo (pace), Identité Défensive (def_rtg).
    """
    vec: dict[str, Any] = {}

    # Volume Extérieur : three_rate
    vec["volume_exterieur"] = round(
        value_to_percentile(
            weighted.get("three_rate"),
            league_mean.get("three_rate"),
            league_std.get("three_rate") or 0.01,
        ),
        1,
    )

    # Pression Intérieure : moyenne pondérée orb_pct + ft_rate
    orb = weighted.get("orb_pct") or 0.25
    ft = weighted.get("ft_rate") or 0.25
    orb_mean = league_mean.get("orb_pct") or 0.25
    orb_std = league_std.get("orb_pct") or 0.05
    ft_mean = league_mean.get("ft_rate") or 0.25
    ft_std = league_std.get("ft_rate") or 0.05
    vec["pression_interieure"] = round(
        (value_to_percentile(orb, orb_mean, orb_std) + value_to_percentile(ft, ft_mean, ft_std)) / 2,
        1,
    )

    # Contrôle du Tempo : pace
    vec["controle_tempo"] = round(
        value_to_percentile(
            weighted.get("pace"),
            league_mean.get("pace"),
            league_std.get("pace") or 5,
        ),
        1,
    )

    # Identité Défensive : def_rtg (plus bas = meilleur)
    vec["identite_defensive"] = round(
        value_to_percentile(
            weighted.get("def_rtg"),
            league_mean.get("def_rtg"),
            league_std.get("def_rtg") or 5,
            higher_better=False,
        ),
        1,
    )

    vec["efg_std"] = round(weighted.get("efg_std", 0), 4)
    vec["pace"] = round(weighted.get("pace", 70), 1)
    vec["def"] = round(weighted.get("def_rtg", 100), 1)
    vec["orb"] = round((weighted.get("orb_pct") or 0.25) * 100, 1)
    vec["three_rate_pct"] = round((weighted.get("three_rate") or 0) * 100, 1)

    return vec


# --- 4. CLASSIFICATION ARCHÉTYPE + NUANCE ---


def classify_archetype(weighted: dict[str, Any], profile_vec: dict[str, Any]) -> str:
    """
    Classification RELATIVE (Percentiles) et non absolue.
    S'adapte automatiquement à la ligue (Pro B, Euroleague, NBA...).

    Utilise profile_vec (0-100) calculé par rapport à la moyenne de la ligue.
    """
    # On récupère les scores relatifs (0 à 100) calculés vs la ligue
    vol_ext = profile_vec.get("volume_exterieur", 50)  # Dépendance 3 pts
    press_int = profile_vec.get("pression_interieure", 50)  # Rebond Off + Lancers
    tempo = profile_vec.get("controle_tempo", 50)  # Pace
    defense = profile_vec.get("identite_defensive", 50)  # Def Rtg (déjà inversé : haut = bonne def)

    efg_std = weighted.get("efg_std") or 0
    base = "Balanced"

    # --- LOGIQUE RELATIVE (FIBA FRIENDLY) ---

    # 1. PACE & SPACE 🚀
    # Équipe qui court (Top 35%) ET qui tire beaucoup à 3 pts (Top 35%)
    if tempo > 65 and vol_ext > 65:
        base = "Pace & Space 🚀"

    # 2. GRIT & GRIND 🛡️
    # Équipe lente (Bottom 35%) ET grosse défense (Top 30%)
    elif tempo < 35 and defense > 70:
        base = "Grit & Grind 🛡️"

    # 3. PAINT BEAST 💪
    # Équipe qui pilonne la raquette (Top 25% Pression Intérieure)
    elif press_int > 75:
        base = "Paint Beast 💪"

    # 4. SNIPERS 🎯 (Nouveau)
    # Équipe statique mais qui vit par le 3 points (Top 20% Volume)
    elif vol_ext > 80:
        base = "Snipers 🎯"

    # 5. RUN & GUN ⚡ (Nouveau)
    # Équipe qui court très vite (Top 15%) peu importe la défense
    elif tempo > 85:
        base = "Run & Gun ⚡"

    # --- NUANCES ---
    nuance = ""
    if efg_std > 0.08:
        nuance = " [Instable]"
    elif defense > 85 and base == "Balanced":
        # Une équipe équilibrée mais avec une défense d'élite
        nuance = " [Lockdown]"

    return base + nuance


def detect_style_shift(weighted: dict[str, Any]) -> bool:
    """True si l'équipe a radicalement changé de style récemment."""
    efg_r = weighted.get("efg_recent")
    efg_b = weighted.get("efg_baseline")
    pace_r = weighted.get("pace_recent")
    pace_b = weighted.get("pace_baseline")
    if pd.isna(efg_r) or pd.isna(efg_b) or pd.isna(pace_r) or pd.isna(pace_b):
        return False
    efg_ecart = abs(efg_r - efg_b)
    pace_ecart = abs(pace_r - pace_b) / max(pace_b, 1)
    return efg_ecart > STYLE_SHIFT_ECART or pace_ecart > STYLE_SHIFT_ECART


# --- 5. PIPELINE PRINCIPAL ---


def update_all_archetypes() -> int:
    """
    Extraction → Pondération → Percentiles ligue → Classification → Upsert teams_metadata.
    Retourne le nombre d'équipes mises à jour.
    """
    df = extract_box_scores_with_meta()
    if df.empty:
        return 0

    league_stats = compute_league_stats(df)
    teams = df["team_id"].unique()
    _log(f"Traitement de {len(teams)} équipes...", "INFO")

    updated = 0
    for tid_raw in teams:
        # CORRECTION CRITIQUE : Conversion du Numpy Int64 en Int Python pur
        tid = int(tid_raw)

        tm = df[df["team_id"] == tid_raw].copy()
        if tm.empty:
            continue

        lid_val = tm["league_id"].dropna().mode()
        # Conversion explicite en int Python
        league_id = int(lid_val.iloc[0]) if len(lid_val) > 0 else None

        if league_id is None:
            continue

        # VÉRIFICATION : une équipe ne change pas de mode — on ne mélange pas les ligues
        # Ex: Monaco en Betclic Élite ≠ Monaco en EuroLeague (même team_id, styles différents)
        tm = tm[tm["league_id"].fillna(-1).astype(int) == league_id].copy()
        if tm.empty:
            _log(f"Team {tid} : aucun match dans la ligue principale {league_id}, skip.", "WARN")
            continue

        # Moyennes ligue
        lg_data = league_stats.get(league_id, {})
        league_mean = {c: lg_data[c][0] for c in lg_data}
        league_std = {c: max(lg_data[c][1], 0.01) for c in lg_data}

        weighted = compute_weighted_profile(tm)
        if not weighted:
            continue

        profile_vec = build_profile_vector(weighted, league_mean, league_std)
        archetype = classify_archetype(weighted, profile_vec)
        style_shift = detect_style_shift(weighted)

        if style_shift:
            profile_vec["style_shift_detected"] = True

        # CORRECTION CRITIQUE 2 : Nettoyage du JSON avant envoi
        profile_vec_clean = to_native(profile_vec)

        # Préserver nom_equipe/name existants (ne pas écraser par "Équipe {tid}")
        nom_existant = None
        try:
            r = supabase.table("teams_metadata").select("nom_equipe, name").eq("team_id", tid).limit(1).execute()
            if r.data and len(r.data) > 0:
                row = r.data[0]
                nom_existant = (row.get("nom_equipe") or row.get("name") or "").strip() or None
        except Exception:
            pass
        nom_final = nom_existant if nom_existant else f"Équipe {tid}"

        try:
            supabase.table("teams_metadata").upsert(
                {
                    "team_id": tid,
                    "name": nom_final,
                    "nom_equipe": nom_final,
                    "league_id": league_id,
                    "current_archetype": archetype,
                    "profile_vector": profile_vec_clean,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                on_conflict="team_id",
            ).execute()
            updated += 1
        except Exception as e:
            err_msg = str(e) if hasattr(e, "__str__") else repr(e)
            if "nom_equipe" in err_msg and ("PGRST204" in err_msg or "schema" in err_msg.lower()):
                _log("Colonne nom_equipe manquante. Exécute schema_migration_nom_equipe.sql dans Supabase.", "ERR")
                raise SystemExit(1) from e
            _log(f"Team {tid} : {e}", "ERR")
            continue

    _log(f"Mise à jour de {updated} équipes.", "OK")
    return updated


# --- 6. DÉTECTION KRYPTONITE (MATCHUP CLASH) ---


# Matrice de conflit tactique : (archétype_outsider, archétype_favori) → Kryptonite ?
KRYPTONITE_MATRIX = {
    ("Pace & Space 🚀", "Grit & Grind 🛡️"): "Le favori (Grit & Grind) ralentit et défend ; l'outsider (Pace & Space) peut être neutralisé.",
    ("Grit & Grind 🛡️", "Pace & Space 🚀"): "KRYPTONITE : Le Grit & Grind (outsider) ralentit et défend — contre naturel du Pace & Space (favori).",
    ("Paint Beast 💪", "Pace & Space 🚀"): "KRYPTONITE : La pression intérieure (Paint Beast) perturbe les shooteurs extérieurs (Pace & Space).",
    ("Pace & Space 🚀", "Paint Beast 💪"): "Le Pace & Space (outsider) tire à 3 ; le Paint Beast (favori) peut être exposé au périmètre.",
    ("Grit & Grind 🛡️", "Paint Beast 💪"): "Le Grit & Grind (outsider) limite les secondes chances — contre du Paint Beast.",
    ("Paint Beast 💪", "Grit & Grind 🛡️"): "KRYPTONITE : La pression physique (Paint Beast) peut cracker une défense Grit & Grind.",
}


def _base_archetype(label: str) -> str:
    """Retire les nuances [Instable] pour la comparaison."""
    if not label:
        return "Balanced"
    return label.split(" [")[0].strip()


def get_tactical_clash(home_id: int, away_id: int) -> dict[str, Any]:
    """
    Compare les archétypes home vs away.
    Retourne {
        home_archetype, away_archetype,
        kryptonite: bool, kryptonite_msg: str | None,
        style_shift_home, style_shift_away,
        alert: str | None
    }
    """
    result: dict[str, Any] = {
        "home_archetype": None,
        "away_archetype": None,
        "kryptonite": False,
        "kryptonite_msg": None,
        "style_shift_home": False,
        "style_shift_away": False,
        "alert": None,
    }

    try:
        meta = supabase.table("teams_metadata").select("*").in_("team_id", [home_id, away_id]).execute().data
    except Exception as e:
        result["alert"] = f"Erreur Supabase: {e}"
        return result

    home_meta = next((m for m in meta if m["team_id"] == home_id), None)
    away_meta = next((m for m in meta if m["team_id"] == away_id), None)

    if not home_meta:
        result["alert"] = f"Équipe domicile {home_id} introuvable en base."
        return result
    if not away_meta:
        result["alert"] = f"Équipe extérieur {away_id} introuvable en base."
        return result

    home_arch = home_meta.get("current_archetype") or "Balanced"
    away_arch = away_meta.get("current_archetype") or "Balanced"
    result["home_archetype"] = home_arch
    result["away_archetype"] = away_arch

    pv_h = home_meta.get("profile_vector") or {}
    pv_a = away_meta.get("profile_vector") or {}
    result["style_shift_home"] = pv_h.get("style_shift_detected", False)
    result["style_shift_away"] = pv_a.get("style_shift_detected", False)

    base_h = _base_archetype(home_arch)
    base_a = _base_archetype(away_arch)

    # Kryptonite : outsider (away) est le contre naturel du favori (home)
    key = (base_a, base_h)
    if key in KRYPTONITE_MATRIX:
        result["kryptonite"] = True
        result["kryptonite_msg"] = KRYPTONITE_MATRIX[key]

    # Alerte Style Shift
    if result["style_shift_home"]:
        result["alert"] = (result["alert"] or "") + " [STYLE_SHIFT_DETECTED] Équipe domicile a changé de style récemment."
    if result["style_shift_away"]:
        result["alert"] = (result["alert"] or "") + " [STYLE_SHIFT_DETECTED] Équipe extérieur a changé de style récemment."

    return result


# --- MAIN ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Archetype Engine — Profilage DNA")
    parser.add_argument(
        "--clash",
        nargs=2,
        type=int,
        metavar=("HOME_ID", "AWAY_ID"),
        help="Analyse matchup clash (Kryptonite) : python3 archetype_engine.py --clash 16 11",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧬 ARCHETYPE ENGINE — Profilage DNA des équipes")
    print("=" * 60)

    if args.clash:
        home_id, away_id = args.clash
        _log(f"Analyse matchup : domicile {home_id} vs extérieur {away_id}", "INFO")
        result = get_tactical_clash(home_id, away_id)
        print("\n   Résultat :")
        for k, v in result.items():
            print(f"      {k}: {v}")
        if result.get("kryptonite"):
            _log(result.get("kryptonite_msg", ""), "ALERT")
        if result.get("alert"):
            _log(result.get("alert", ""), "WARN")
        print("\n✅ Fin du script.\n")
        sys.exit(0)

    try:
        from scipy import stats  # noqa: F401
    except ImportError:
        print("   ⚠️  pip install scipy pour les percentiles.")
        sys.exit(1)

    _log("Démarrage du recalcul des archétypes...", "INFO")
    n = update_all_archetypes()
    _log(f"Terminé. {n} équipes mises à jour.", "OK")
    print("\n✅ Fin du script.\n")
