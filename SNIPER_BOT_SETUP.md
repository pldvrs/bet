# Configuration du Sniper Bot (GitHub Actions)

## Secrets à configurer

Dans **GitHub** → ton repo → **Settings** → **Secrets and variables** → **Actions** :

| Secret | Description |
|--------|-------------|
| `API_BASKETBALL_KEY` | Clé API basketball.api-sports.io |
| `SUPABASE_URL` | URL du projet Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | Clé service role Supabase |
| `TELEGRAM_BOT_TOKEN` | Token du bot Telegram (@BotFather) |
| `TELEGRAM_CHAT_ID` | ID du chat/chaîne Telegram |
| `DISCORD_WEBHOOK_URL` | URL du webhook Discord (optionnel) |

## Planning quotidien

- **08h00** (Paris) : Collecte des résultats de la veille + Box Scores
- **08h15** : Calendrier du jour (inclu dans fill_history)
- **08h30** : Ré-entraînement des modèles ML
- **09h00** : Envoi des alertes 🎯 SNIPER TARGET

## Look-Ahead Bias — Règle d'or

- **Passé (Train)** : Jusqu'à hier 23h59 — matchs avec scores uniquement
- **Futur (Prediction)** : À partir d'aujourd'hui 00h01 — matchs sans scores

Le code trie toujours par date et utilise un `TimeSeriesSplit` strict.

## Architecture Write Once, Read Many

- **03_predict_daily.py** : s’exécute **une seule fois par jour** (08h00 via le pipeline ou manuellement). Charge uniquement les modèles `.pkl` (aucun ré-entraînement), génère les prédictions J+1 à J+3 et écrit dans **daily_projections_v2**.
- **04_app_dashboard.py** : **ne lance jamais** le script 03 ni aucun calcul ML. Il fait un simple `SELECT` sur `daily_projections_v2`. Les pronos restent stables à chaque rechargement de page.

Avant la première utilisation : exécuter la migration SQL `schema_migration_daily_projections_v2.sql` dans Supabase (création de la table `daily_projections_v2`).

## Lancement manuel

Dans **Actions** → **Sniper Bot Daily** → **Run workflow**
