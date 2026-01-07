# Stockbot

Automated trading platform with backtesting and live execution.

## Installation

```bash
pip install -e ".[dev]"
```

## Setup

1. Copy `.env.example` to `.env` and add your Alpaca API credentials
2. Download historical data using the ingestion script
3. Run backtests with your strategy

## Usage

### Download Data

```bash
python scripts/ingest_data.py AAPL MSFT --start 2024-01-01 --end 2024-06-30 --timeframe 1d
```

### Run Backtest

```bash
python scripts/run_backtest.py AAPL --start 2024-01-01 --end 2024-06-30 --strategy sma
```

### Run Tests

```bash
pytest tests/
```

## Architecture

- **Core**: Types, models, and interfaces
- **Data**: Market data providers (Alpaca, Parquet)
- **Strategy**: Trading strategies (SMA crossover, etc.)
- **Risk**: Risk management rules
- **Execution**: Order execution and simulation
- **Engine**: Backtesting engine
