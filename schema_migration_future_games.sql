-- =============================================================================
-- MIGRATION : Autoriser NULL sur home_score/away_score pour les matchs à venir
--
-- Permet de stocker le calendrier futur dans games_history (sans scores).
-- Exécuter dans Supabase → SQL Editor.
-- =============================================================================

ALTER TABLE games_history
ALTER COLUMN home_score DROP NOT NULL;

ALTER TABLE games_history
ALTER COLUMN away_score DROP NOT NULL;
