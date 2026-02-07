-- =============================================================================
-- MIGRATION : Ajouter date à box_scores + reset complet
--
-- Étapes :
-- 1. Supabase Dashboard → SQL Editor → New query
-- 2. Colle le contenu ci-dessous
-- 3. Run
--
-- ⚠️ ATTENTION : TRUNCATE supprime TOUTES les données. Réingestion nécessaire.
-- =============================================================================

-- 1. Ajouter la colonne date à box_scores (redondante pour requêtes rapides)
ALTER TABLE box_scores
ADD COLUMN IF NOT EXISTS date DATE;

-- 2. Supprimer toutes les données (ordre pour respecter les clés étrangères)
TRUNCATE TABLE box_scores;
TRUNCATE TABLE games_history;
TRUNCATE TABLE teams_metadata RESTART IDENTITY;

-- Ensuite : relance l'ingestion (python3 backend_engine.py --month 2026-01, etc.)
