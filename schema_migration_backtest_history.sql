-- MIGRATION : Table backtest_history — Backtest persistant
-- =======================================================
-- Stocke les résultats de backtest figés (1 ligne par match).
-- Alimenté quotidiennement par 05_run_backtest.py.

CREATE TABLE IF NOT EXISTS backtest_history (
    game_id INT PRIMARY KEY REFERENCES games_history(game_id) ON DELETE CASCADE,
    match_date DATE NOT NULL,
    match_name TEXT NOT NULL,

    prediction_winner TEXT,
    prediction_proba REAL,
    prediction_total_points REAL,

    actual_winner TEXT,
    actual_total_points REAL,

    bet_suggested TEXT,
    odds_taken REAL,
    profit REAL,
    status TEXT CHECK (status IN ('WIN', 'LOSS', 'PUSH')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_history_match_date
    ON backtest_history(match_date DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_history_status
    ON backtest_history(status);
