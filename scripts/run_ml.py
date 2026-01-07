#!/usr/bin/env python3
"""Run a trained ML agent for backtesting or paper trading.

Usage:
    python scripts/run_ml.py SPY --backtest
    python scripts/run_ml.py SPY --paper
"""

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

from stockbot.config.settings import load_settings
from stockbot.core.models import Bar
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import load_bars, save_bars
from stockbot.learning.features import FeatureExtractor
from stockbot.learning.rl_agent import Action, TradingAgent
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger

logger = get_logger("run_ml")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run trained ML agent")

    parser.add_argument(
        "symbol",
        type=str,
        help="Symbol to trade",
    )

    parser.add_argument(
        "--agent",
        type=Path,
        default=Path("./data/ml_agent.json"),
        help="Path to trained agent",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on recent data",
    )

    parser.add_argument(
        "--backtest-months",
        type=int,
        default=3,
        help="Months of data for backtest (default: 3)",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital for backtest (default: 100000)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory for data files",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def load_agent(path: Path) -> TradingAgent:
    """Load a trained agent from file."""
    feature_extractor = FeatureExtractor(
        lookback_periods=[5, 10, 20, 50],
        include_volume=True,
        include_volatility=True,
        include_momentum=True,
        include_mean_reversion=True,
    )

    agent = TradingAgent(
        feature_extractor=feature_extractor,
        hidden_size=64,
        epsilon_start=0.0,  # No exploration during execution
        epsilon_end=0.0,
    )

    agent.load(path)
    return agent


def get_bars(symbol: Symbol, months: int, data_dir: Path) -> list[Bar]:
    """Get bars for backtesting."""
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=months * 30)

    start_dt = datetime.fromisoformat(start.isoformat()).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end.isoformat()).replace(tzinfo=timezone.utc)
    start_ts = Timestamp(int(start_dt.timestamp() * 1_000_000_000))
    end_ts = Timestamp(int(end_dt.timestamp() * 1_000_000_000))

    # Try local first
    try:
        all_bars = list(load_bars(data_dir, symbol, Timeframe.DAY_1))
        bars = [b for b in all_bars if start_ts <= b.timestamp <= end_ts]
        if len(bars) > 20:
            return bars
    except Exception:
        pass

    # Fetch from Alpaca
    settings = load_settings()
    provider = AlpacaDataProvider(settings.alpaca)
    bars = list(provider.get_bars(symbol, start_ts, end_ts, Timeframe.DAY_1))

    return bars


def run_backtest(
    agent: TradingAgent,
    bars: list[Bar],
    feature_extractor: FeatureExtractor,
    initial_capital: float,
    symbol: Symbol,
) -> dict:
    """Run backtest with trained agent."""
    agent.reset_episode()
    agent._epsilon = 0.0  # No exploration

    capital = initial_capital
    position = 0  # Shares held
    entry_price = 0.0

    trades = []
    equity_curve = []
    min_bars = 60

    print(f"\nRunning backtest on {len(bars) - min_bars} days...\n")

    for i in range(min_bars, len(bars)):
        historical_bars = bars[:i+1]
        features = feature_extractor.extract(historical_bars)

        if features is None:
            continue

        current_price = features.price

        # Record equity
        equity = capital + position * current_price
        equity_curve.append((bars[i].timestamp, equity))

        # Get agent's action
        action = agent.get_action(features, training=False)

        # Execute action
        if action == Action.BUY and position == 0:
            # Buy with full capital
            shares = int(capital / current_price)
            if shares > 0:
                position = shares
                capital -= shares * current_price
                entry_price = current_price
                logger.debug(f"BUY {shares} @ ${current_price:.2f}")

        elif action == Action.SELL and position > 0:
            # Sell all
            capital += position * current_price
            pnl = (current_price - entry_price) / entry_price
            trades.append({
                "entry": entry_price,
                "exit": current_price,
                "pnl_pct": pnl * 100,
                "shares": position,
            })
            logger.debug(f"SELL {position} @ ${current_price:.2f} (P&L: {pnl*100:.2f}%)")
            position = 0

    # Close any open position
    if position > 0:
        final_price = float(bars[-1].close)
        capital += position * final_price
        pnl = (final_price - entry_price) / entry_price
        trades.append({
            "entry": entry_price,
            "exit": final_price,
            "pnl_pct": pnl * 100,
            "shares": position,
        })

    # Calculate results
    final_equity = capital
    total_return = (final_equity - initial_capital) / initial_capital * 100

    winning_trades = [t for t in trades if t["pnl_pct"] > 0]
    losing_trades = [t for t in trades if t["pnl_pct"] <= 0]

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(trades) * 100 if trades else 0,
        "max_drawdown_pct": max_dd * 100,
        "trades": trades,
    }


def print_backtest_results(results: dict, symbol: Symbol) -> None:
    """Print backtest results."""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS - {symbol}")
    print("=" * 60)

    print(f"\nPerformance:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Equity:    ${results['final_equity']:,.2f}")
    print(f"  Total Return:    {results['total_return_pct']:.2f}%")
    print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")

    print(f"\nTrades:")
    print(f"  Total Trades:    {results['total_trades']}")
    print(f"  Winning Trades:  {results['winning_trades']}")
    print(f"  Losing Trades:   {results['losing_trades']}")
    print(f"  Win Rate:        {results['win_rate']:.1f}%")

    if results['trades']:
        pnls = [t['pnl_pct'] for t in results['trades']]
        print(f"\n  Avg Trade P&L:   {np.mean(pnls):.2f}%")
        print(f"  Best Trade:      {max(pnls):.2f}%")
        print(f"  Worst Trade:     {min(pnls):.2f}%")

    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(level=args.log_level)

    symbol = Symbol(args.symbol.upper())

    # Load trained agent
    if not args.agent.exists():
        print(f"Error: Agent file not found: {args.agent}")
        print("Train an agent first with: python scripts/train_ml.py SPY")
        return 1

    print(f"\nLoading trained agent from {args.agent}...")
    agent = load_agent(args.agent)

    # Create feature extractor (must match training)
    feature_extractor = FeatureExtractor(
        lookback_periods=[5, 10, 20, 50],
        include_volume=True,
        include_volatility=True,
        include_momentum=True,
        include_mean_reversion=True,
    )

    if args.backtest:
        # Get data
        bars = get_bars(symbol, args.backtest_months, args.data_dir)
        if len(bars) < 70:
            print(f"Error: Not enough data ({len(bars)} bars)")
            return 1

        # Run backtest
        results = run_backtest(
            agent=agent,
            bars=bars,
            feature_extractor=feature_extractor,
            initial_capital=args.initial_capital,
            symbol=symbol,
        )

        print_backtest_results(results, symbol)

        # Show what the agent learned
        print("Feature Importance (what the agent thinks matters):")
        importance = agent.get_feature_importance()
        for name, score in importance[:10]:
            print(f"  {name:<30} {score:.4f}")
        print()

    else:
        print("Specify --backtest to run a backtest")
        print("Paper trading with ML agent coming soon")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
