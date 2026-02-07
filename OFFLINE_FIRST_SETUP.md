# Architecture Offline-First — Sniper V27

## Vue d'ensemble

- **Bot** (`bot_ingest_data.py`) : API + ML → remplit `daily_projections`
- **App** (`app_sniper_v27.py`) : lit uniquement depuis `daily_projections` (instantané)

## 1. Créer la table

Exécuter dans Supabase → SQL Editor :

```sql
-- Voir schema_migration_daily_projections.sql
```

## 2. Lancer le bot

```bash
python bot_ingest_data.py
```

À planifier en CRON ou GitHub Actions (ex: 08h15, 18h00).

## 3. L'app en mode Offline

Par défaut, `SNIPER_OFFLINE=1` → l'app lit `daily_projections` uniquement.
Pas d'appel API, pas de calcul ML → affichage instantané.

Pour repasser en mode live (API + ML en temps réel) :

```bash
SNIPER_OFFLINE=0 streamlit run app_sniper_v27.py
```

## Colonnes daily_projections

| Colonne | Description |
|---------|-------------|
| game_id | PK |
| match_name | "Équipe A vs Équipe B" |
| date, time, jour | Date/heure du match |
| proba_ml, proba_calibree | Probabilités ML |
| edge_percent | Edge (EV %) |
| brain_used, confiance_label | Cerveau, label confiance |
| predicted_total, line_bookmaker, diff_total | Over/Under |
| odds_home, odds_away | Cotes bookmaker |
| updated_at | Dernière mise à jour |
