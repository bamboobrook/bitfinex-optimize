# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitfinex lending rate prediction and optimization system. It downloads historical funding rate data, trains ML models to predict optimal lending rates, and provides API endpoints and Telegram notifications for trading decisions.

## Key Components

- **funding_history_downloader.py**: Downloads minute-level funding rate candles from Bitfinex API, stores in SQLite database (`data/lending_history.db`)
- **auto_optimize.py**: Main ML optimizer using PyTorch neural networks to predict lending rates across multiple currencies (fUST, fUSD) and periods (2-120 days)
- **analyze.py**: Alternative simpler ML optimizer implementation
- **api_server.py**: FastAPI server for prediction endpoints (port 8000)
- **telegram_bot.py**: Telegram notification integration

## Database Schema

SQLite database at `data/lending_history.db` with table `funding_rates`:
- Key columns: `currency`, `period`, `timestamp`, `datetime`, `close_annual` (yield rate), `volume`, `hour`, `day_of_week`, `month`
- Indexed on: `(currency, period)`, `timestamp`, `datetime`, `(currency, period, datetime)`

## Enviroment
- conda activate optimize

## Common Commands

```bash
# Download historical data (runs for a while due to rate limiting)
python funding_history_downloader.py

# Run ML optimization directly
python auto_optimize.py

# Start Flask API server on port 5000
python auto_optimize.py --api

# Start FastAPI server on port 8000
python api_server.py
```

## API Endpoints

Flask API (auto_optimize.py --api, port 5000):
- `GET /api/ml/optimize` - Full optimization with all metrics
- `GET /api/ml/simple_optimize` - Simplified optimal combination
- `GET /api/ml/optimal_combination` - Read cached results from JSON
- `GET /api/ml/health` - Health check

FastAPI (api_server.py, port 8000):
- `POST /api/v1/predict` - Trigger async prediction
- `GET /api/v1/predict/sync` - Synchronous prediction
- `GET /api/v1/latest` - Get latest prediction results
- `GET /api/v1/health` - Health check

## Output Files

- `data/optimal_combination.json` - Full prediction results with all metrics
- `data/optimal_simple.json` - Simplified optimal combination
- `log/ml_optimizer.log` - Application logs (7 day retention)

## Hardware Configuration

The system is designed for high-performance hardware:
- GPU: NVIDIA RTX 5090 (CUDA enabled)
- RAM: 256GB
- CPU: 16 cores / 32 threads

PyTorch uses CUDA with `cudnn.benchmark=True` for optimization.

## ML Model Architecture

Current implementation uses a simple MLP (FastPredictor/ImprovedPredictor):
- Input: Time features (cyclical hour/day/month), yield rates, volume, moving averages, momentum
- Hidden layers: 128-256 units with ReLU and Dropout
- Output: Predicted yield rate

Models are trained per (currency, period, lookback_days) combination and cached in `data/ml_models/`.

## Key Configuration Parameters

In `MLOptimizerConfig`:
- `periods`: [2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120]
- `currencies`: ['fUST', 'fUSD']
- `lookback_days_list`: [7, 15, 30, 60, 90]
- `batch_size`: 4096
- `n_epochs`: 200-1000
