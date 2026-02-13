#!/usr/bin/env bash
set -euo pipefail

python 99_simulate_past_predictions.py --days 10
python 05_run_backtest.py --days 10
