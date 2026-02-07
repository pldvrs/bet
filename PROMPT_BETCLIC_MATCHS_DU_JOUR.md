# Prompt : récupérer les matchs du jour Betclic Élite

Tu peux copier-coller ce prompt pour obtenir uniquement les matchs du jour de la Betclic Élite :

---

**Prompt :**

Récupère les matchs du jour de la **Betclic Élite** (basketball France) via l’API Basketball api-sports.

- **Endpoint :** `GET https://v1.basketball.api-sports.io/games`
- **Header :** `x-apisports-key: [MA_CLÉ]`
- **Paramètres :**
  - `date` : aujourd’hui au format `YYYY-MM-DD`
  - `league` : **2** (Betclic Élite)
  - `season` : **2025-2026** (saison en cours — doc API ; pas 2024 = passé récent)
  - `timezone` : `Europe/Paris`

Affiche simplement la liste des matchs (Domicile vs Extérieur). Utilise uniquement la saison courante 2025-2026.

---

## Script fourni

Le fichier **`betclic_matchs_du_jour.py`** fait exactement ça :

```bash
# Avec la clé en variable d’environnement
export API_KEY=ta_cle_api
python betclic_matchs_du_jour.py

# Ou : le script demandera la clé au lancement
python betclic_matchs_du_jour.py
```

Il utilise la saison courante **2025-2026** et affiche les matchs du jour dès qu’une saison en renvoie.
