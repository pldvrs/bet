-- MIGRATION : Table daily_projections pour architecture Offline-First
-- =====================================================================
-- Le bot (bot_ingest_data.py) remplit cette table.
-- L'app Sniper (app_sniper_v27.py) lit uniquement depuis ici — plus d'API ni ML en temps réel.

CREATE TABLE IF NOT EXISTS daily_projections (
    game_id INT PRIMARY KEY,
    match_name TEXT NOT NULL,
    date DATE NOT NULL,
    time TEXT,
    jour TEXT,
    league_id INT,
    season TEXT,
    home_id INT,
    away_id INT,

    -- Moneyline / Vainqueur
    proba_ml REAL,
    proba_calibree REAL,
    edge_percent REAL,
    brain_used TEXT,
    confiance_label TEXT,
    le_pari TEXT,
    pari_outsider TEXT,
    alerte_trappe TEXT,
    message_contexte TEXT,
    fiabilite REAL,

    -- Over/Under
    predicted_total REAL,
    line_bookmaker REAL,
    diff_total REAL,
    pari_total TEXT,
    confiance_ou TEXT,
    style_match TEXT,

    -- Cotes (API)
    odds_home REAL,
    odds_away REAL,

    -- Métadonnées
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour filtrage rapide par date
CREATE INDEX IF NOT EXISTS idx_daily_projections_date ON daily_projections(date);
CREATE INDEX IF NOT EXISTS idx_daily_projections_updated ON daily_projections(updated_at DESC);

COMMENT ON TABLE daily_projections IS 'Cache des projections ML + cotes — alimenté par bot_ingest_data.py, lu par app_sniper_v27.py';
