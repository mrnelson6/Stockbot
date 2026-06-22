# Stockbot

Automated trading platform with ML-based portfolio management, backtesting, and live execution via Alpaca.

## Installation

```bash
pip install -e ".[dev]"
```

## Setup

1. Copy `.env.example` to `.env` and add your Alpaca API credentials:
   ```
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
2. Run tests to verify: `pytest tests/`

## Scripts

### Data & Account Management

| Script | Description |
|--------|-------------|
| `ingest_data.py` | Downloads historical price data from Alpaca and stores in parquet files |
| `check_account.py` | Displays Alpaca account status (balance, buying power, positions) |

```bash
# Download data
python scripts/ingest_data.py SPY AAPL MSFT --start 2024-01-01 --end 2024-12-31 --timeframe 1d

# Check account
python scripts/check_account.py
```

### Traditional Strategy Backtesting

| Script | Description |
|--------|-------------|
| `run_backtest.py` | Runs backtest of predefined strategies (SMA crossover, buy-and-hold) |
| `run_paper.py` | Paper trading with a strategy against live market data |
| `optimize.py` | Optimizes strategy parameters using walk-forward analysis |

```bash
# Backtest SMA strategy
python scripts/run_backtest.py SPY --start 2024-01-01 --end 2024-12-31 --strategy sma

# Optimize parameters
python scripts/optimize.py SPY --start 2024-01-01 --end 2024-12-31
```

### Strategy Selection System

| Script | Description |
|--------|-------------|
| `train_selector.py` | Trains a selector that learns which strategy works best in different conditions |
| `run_with_selector.py` | Runs backtests using the trained strategy selector |

```bash
# Train selector on 12 months of data
python scripts/train_selector.py --training-months 12

# Run with trained selector
python scripts/run_with_selector.py SPY
```

### Single-Asset ML Trading

| Script | Description |
|--------|-------------|
| `train_ml.py` | Trains Q-learning RL agent on a single stock to learn position sizing |
| `run_ml.py` | Backtests the trained single-asset ML agent |
| `live_ml.py` | Monitors real-time prices with trained ML agent decisions |

```bash
# Train on SPY
python scripts/train_ml.py --symbol SPY --training-months 12 --epochs 10

# Backtest
python scripts/run_ml.py --symbol SPY

# Live monitoring
python scripts/live_ml.py --symbol SPY
```

### Multi-Asset Portfolio ML

| Script | Description |
|--------|-------------|
| `train_portfolio.py` | Trains portfolio agent on 100 stocks to learn optimal allocations |
| `live_portfolio.py` | Monitors all stocks in real-time and decides portfolio allocations |

```bash
# Train on 25 stocks with 12 months of daily data
python scripts/train_portfolio.py --universe-size 25 --training-months 12 --timeframe 1D

# Full training: 100 stocks, 2 years, 1-minute data (takes hours)
python scripts/train_portfolio.py

# Live monitoring (watch only)
python scripts/live_portfolio.py --universe-size 25

# Live with trade execution
python scripts/live_portfolio.py --universe-size 25 --execute
```

### Random Bot (control / experiment)

A standalone bot that invests **entirely at random** — useful as a control to
benchmark the ML bot against pure chance. It reuses the same Alpaca execution
plumbing but runs its **own isolated portfolio**.

Strategy: on each tick it trades with probability `--trade-prob` (per-tick gate);
a trade event randomly sells a subset of current holdings (**random partial churn**,
`--churn-sell-prob`) and buys a random number of new names sized with **Dirichlet
random weights** over a random fraction of available cash.

| Script | Description |
|--------|-------------|
| `random_portfolio.py` | Trades a separate portfolio with a fully random strategy |

> **Portfolio isolation:** an Alpaca account *is* its API keys. For a truly separate
> portfolio, create a **second Alpaca paper account** and set `RANDOM_ALPACA_API_KEY` /
> `RANDOM_ALPACA_SECRET_KEY` (see `.env.example`). Without them it falls back to the
> shared `ALPACA_*` keys and would fight the ML bot over positions.

**Universe:** by default (`--universe-source alpaca`) it fetches every active, tradable
US equity from Alpaca and keeps the most liquid names, filtered by `--min-price` and
`--min-dollar-volume` and capped at `--max-symbols` (so the bot only throws darts at
liquid, fillable stocks). Use `--universe-source static --universe-size {10,25,50,100}`
for the small hand-curated list instead.

```bash
# Dry run - simulated, no orders placed (default).
# Dynamic universe: liquid US equities over $5 with >$10M/day dollar-volume, top 200.
python scripts/random_portfolio.py --trade-prob 0.5 --seed 1

# Tune the dynamic universe filters
python scripts/random_portfolio.py --min-price 10 --min-dollar-volume 50000000 --max-symbols 100

# Small static universe instead
python scripts/random_portfolio.py --universe-source static --universe-size 25

# Place real paper orders on the random bot's own account
python scripts/random_portfolio.py --execute
```

Trades are **simulated unless you pass `--execute`**, and paper mode defaults on.

**Fractional shares** are on by default (`--fractional`; use `--no-fractional` for whole shares).
This matters a lot for small accounts: with whole shares only, any name priced above a slice of
your cash is skipped, so a ~$1,000 account can't buy pricier stocks and strands most of its cash.

**Position cap** is a fraction of account value: `--max-position-pct` (default `0.2` = no single
position targets more than 20% of equity). Because it's relative, it auto-scales — 20% is ~$200 on
a $1,000 account and ~$20k on a $100k account — and it forces the pool to spread across at least
`1 / max-position-pct` names (≈5 at the default), so you don't need to retune it per account size.

**Public dashboard:** add `--record-db ./data/dashboard.db` to log equity/positions/trades,
then serve a read-only web page (current positions, trade history, and a portfolio-value-vs-SPY
chart):

```bash
pip install -e ".[web]"
python scripts/random_portfolio.py --execute --record-db ./data/dashboard.db   # writer
python scripts/serve_web.py --db ./data/dashboard.db --port 8000                # reader
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for running both as `systemd` services with HTTPS on your
own domain via Caddy.

### Monitoring

| Script | Description |
|--------|-------------|
| `risk_dashboard.py` | Displays real-time risk metrics, position exposure, and P&L |

```bash
python scripts/risk_dashboard.py
```

## Quick Start

```bash
# 1. Check your account connection
python scripts/check_account.py

# 2. Train a portfolio agent (start small)
python scripts/train_portfolio.py --universe-size 10 --training-months 6 --timeframe 1D --epochs 3

# 3. Monitor live (no trades)
python scripts/live_portfolio.py --universe-size 10
```

## Architecture

```
src/stockbot/
├── core/           # Types, models, interfaces
├── config/         # Settings, stock universe
├── data/           # Market data providers (Alpaca, Parquet storage)
├── strategy/       # Trading strategies (SMA crossover, buy-and-hold)
├── risk/           # Risk management rules
├── execution/      # Order execution and simulation
├── engine/         # Backtesting and paper trading engines
├── learning/       # ML components
│   ├── features.py        # 50+ technical indicators
│   ├── rl_agent.py        # Single-asset Q-learning agent
│   ├── portfolio_agent.py # Multi-asset portfolio agent
│   ├── selector.py        # Strategy selector
│   └── callbacks.py       # Learning callbacks for engines
└── monitoring/     # Logging and metrics
```

## Stock Universe

The portfolio system trades 100 liquid stocks/ETFs:

- **Index ETFs**: SPY, QQQ, IWM, DIA, VTI
- **Sector ETFs**: XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE
- **Mega Cap Tech**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, etc.
- **Financials**: JPM, BAC, GS, V, MA, etc.
- **Healthcare**: JNJ, UNH, PFE, LLY, etc.
- **Consumer**: WMT, COST, HD, NKE, SBUX, MCD, etc.
- **Energy**: XOM, CVX, COP, SLB, EOG
- **Industrials**: CAT, BA, HON, UPS, RTX, etc.

## ML Approach

The portfolio agent uses **Q-learning with neural networks**:

1. **Feature Extraction**: 42 features per asset including:
   - Price returns (1, 5, 10, 20, 50 day)
   - SMA ratios and crossovers
   - RSI, MACD, Bollinger Bands
   - Volatility and ATR
   - Volume patterns
   - Mean reversion signals

2. **Neural Network**: 3-layer network (256→128→64) that maps features to Q-values

3. **Action Space**: Position sizes from -100% to +100% per asset

4. **Constraints**:
   - Max 20% in any single asset
   - 100% max gross exposure
   - Transaction cost penalties

5. **Learning**: Experience replay with target network for stable training

## Data Storage

Training data is cached in `data/` directory:
- `data/portfolio/*.parquet` - Historical bars per symbol
- `data/portfolio_agent.json` - Trained neural network weights
- `data/ml_agent.json` - Single-asset agent weights

## Tests

```bash
pytest tests/ -v
```

## License

MIT
