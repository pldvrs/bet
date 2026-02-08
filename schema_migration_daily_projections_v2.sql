-- MIGRATION : Table daily_projections_v2 — Write Once, Read Many
-- ================================================================
-- Clé primaire composée (game_id, date_prediction) pour une projection
-- figée par jour. Alimentée UNE FOIS par jour par 03_predict_daily.py.
-- Le dashboard 04 lit uniquement cette table (aucun calcul ML).

CREATE TABLE IF NOT EXISTS daily_projections_v2 (
    game_id INT NOT NULL,
    date_prediction DATE NOT NULL,

    -- Identité du match
    match_name TEXT NOT NULL,
    league TEXT,
    start_time TIMESTAMPTZ,

    -- Probabilités ML (figées)
    proba_ml REAL NOT NULL,
    proba_ml_calibrated REAL NOT NULL,

    -- Scores projetés
    projected_score_home REAL NOT NULL,
    projected_score_away REAL NOT NULL,
    total_points_projected REAL NOT NULL,

    -- Cotes bookmaker (au moment du figage)
    bookmaker_odds_home REAL,
    bookmaker_odds_away REAL,
    bookmaker_line_total REAL,

    -- Edge calculés une seule fois
    edge_ml REAL,
    edge_total REAL,

    -- Fiabilité 0–100
    confidence_score REAL NOT NULL DEFAULT 50,

    -- Explication du pari (affichée dans le dashboard)
    reasoning_text TEXT NOT NULL DEFAULT '',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (game_id, date_prediction)
);

CREATE INDEX IF NOT EXISTS idx_daily_projections_v2_date_prediction
    ON daily_projections_v2(date_prediction DESC);
CREATE INDEX IF NOT EXISTS idx_daily_projections_v2_created_at
    ON daily_projections_v2(created_at DESC);

COMMENT ON TABLE daily_projections_v2 IS 'Table de vérité des projections — Write Once (03_predict_daily), Read Many (04_app_dashboard). Pas de recalcul à la volée.';
