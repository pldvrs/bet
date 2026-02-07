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

## Lancement manuel

Dans **Actions** → **Sniper Bot Daily** → **Run workflow**
