-- Colonnes home_odd / away_odd pour stocker les cotes historiques (Betclic/Unibet).
-- À exécuter une fois si les colonnes n'existent pas encore.
ALTER TABLE games_history
ADD COLUMN IF NOT EXISTS home_odd FLOAT,
ADD COLUMN IF NOT EXISTS away_odd FLOAT;
