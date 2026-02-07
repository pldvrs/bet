-- =============================================================================
-- MIGRATION : Ajouter nom_equipe à teams_metadata
--
-- Étapes :
-- 1. Supabase Dashboard → SQL Editor → New query
-- 2. Colle le contenu ci-dessous
-- 3. Run
-- =============================================================================

ALTER TABLE teams_metadata
ADD COLUMN IF NOT EXISTS nom_equipe TEXT;

-- Optionnel : copier name → nom_equipe si nom_equipe vide
UPDATE teams_metadata
SET nom_equipe = COALESCE(NULLIF(TRIM(nom_equipe), ''), name)
WHERE nom_equipe IS NULL AND name IS NOT NULL;
