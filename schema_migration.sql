-- =============================================================================
-- MIGRATION : Ajouter three_rate à box_scores
-- Erreur PGRST204 "Could not find the 'three_rate' column" = exécuter ce fichier
--
-- Étapes :
-- 1. Ouvre Supabase Dashboard → ton projet
-- 2. SQL Editor → New query
-- 3. Colle le contenu ci-dessous
-- 4. Run
-- =============================================================================

ALTER TABLE box_scores 
ADD COLUMN IF NOT EXISTS three_rate FLOAT;
