# Automatisation du pipeline Sniper (cron / GitHub Actions)

## Option 1 : GitHub Actions (recommandé si le repo est sur GitHub)

Le workflow **`.github/workflows/sniper_bot.yml`** est déjà configuré :

- **Quand** : tous les jours à **08h00** (heure de Paris).
- **Quoi** : `python run_pipeline.py` (01 → 02 → 03), puis `send_alerts.py` si configuré.

**À faire :**

1. Dans ton repo GitHub : **Settings → Secrets and variables → Actions**.
2. Ajoute les secrets :
   - `API_BASKETBALL_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - (optionnel) `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DISCORD_WEBHOOK_URL`
3. Le pipeline tournera automatiquement chaque jour. Tu peux aussi le lancer à la main : **Actions → Sniper Bot Daily → Run workflow**.

---

## Option 2 : Cron (sur ton Mac ou serveur)

Pour lancer le pipeline tous les jours à 08h00 **en local** :

```bash
crontab -e
```

Ajoute cette ligne (remplace `/chemin/vers/bet` par le vrai chemin du projet) :

```
0 8 * * * cd /Users/pl/Documents/GitHub/bet && /usr/bin/env python run_pipeline.py >> /tmp/sniper_pipeline.log 2>&1
```

- `0 8 * * *` = à 8h00 tous les jours.
- Pour utiliser le Python du projet (ex. venv) : mets le chemin complet vers `python` dans ton venv, par exemple :
  ```
  0 8 * * * cd /Users/pl/Documents/GitHub/bet && ./venv/bin/python run_pipeline.py >> /tmp/sniper_pipeline.log 2>&1
  ```

Vérifier que cron a accès au `.env` : le script est lancé depuis le dossier du projet, donc `load_dotenv` chargera bien `.env` dans ce dossier.

---

## Résumé

| Méthode            | Avantage                    | Inconvénient              |
|--------------------|-----------------------------|---------------------------|
| **GitHub Actions** | Pas de machine allumée      | Secrets à configurer      |
| **Cron local**    | Tout reste sur ta machine   | Mac/serveur doit être on  |
