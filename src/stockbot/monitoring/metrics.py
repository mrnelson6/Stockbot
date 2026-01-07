"""Performance metrics calculations."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

import numpy as np

from stockbot.core.models import TradeRecord
from stockbot.core.types import Price, Timestamp


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal

    # Risk metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Average trade metrics
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal
    avg_trade_duration_hours: Decimal


def calculate_sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Decimal:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Sequence of period returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return Decimal("0")

    returns_arr = np.array(returns, dtype=np.float64)
    excess_returns = returns_arr - (risk_free_rate / periods_per_year)

    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0:
        return Decimal("0")

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return Decimal(str(round(sharpe, 4)))


def calculate_sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Decimal:
    """Calculate annualized Sortino ratio.

    Uses downside deviation instead of standard deviation.

    Args:
        returns: Sequence of period returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return Decimal("0")

    returns_arr = np.array(returns, dtype=np.float64)
    excess_returns = returns_arr - (risk_free_rate / periods_per_year)

    mean_return = np.mean(excess_returns)

    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return Decimal("999.99")  # No negative returns

    downside_dev = np.sqrt(np.mean(negative_returns**2))

    if downside_dev == 0:
        return Decimal("0")

    sortino = (mean_return / downside_dev) * np.sqrt(periods_per_year)
    return Decimal(str(round(sortino, 4)))


def calculate_max_drawdown(
    equity_curve: Sequence[tuple[Timestamp, Price]],
) -> tuple[Decimal, Decimal]:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Sequence of (timestamp, equity) tuples

    Returns:
        Tuple of (max_drawdown_amount, max_drawdown_pct)
    """
    if len(equity_curve) < 2:
        return Decimal("0"), Decimal("0")

    equities = [float(eq) for _, eq in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for equity in equities:
        if equity > peak:
            peak = equity

        dd = peak - equity
        dd_pct = dd / peak if peak > 0 else 0

        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return Decimal(str(round(max_dd, 2))), Decimal(str(round(max_dd_pct * 100, 2)))


def calculate_returns_from_equity(
    equity_curve: Sequence[tuple[Timestamp, Price]],
) -> list[float]:
    """Calculate period returns from equity curve.

    Args:
        equity_curve: Sequence of (timestamp, equity) tuples

    Returns:
        List of period returns
    """
    if len(equity_curve) < 2:
        return []

    equities = [float(eq) for _, eq in equity_curve]
    returns = []

    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            ret = (equities[i] - equities[i - 1]) / equities[i - 1]
            returns.append(ret)

    return returns


def calculate_trade_metrics(trades: Sequence[TradeRecord]) -> dict[str, Decimal]:
    """Calculate metrics from trade records.

    Args:
        trades: Sequence of trade records

    Returns:
        Dict of metric name to value
    """
    if not trades:
        return {
            "total_trades": Decimal("0"),
            "winning_trades": Decimal("0"),
            "losing_trades": Decimal("0"),
            "win_rate": Decimal("0"),
            "avg_win": Decimal("0"),
            "avg_loss": Decimal("0"),
            "profit_factor": Decimal("0"),
            "avg_trade_duration_hours": Decimal("0"),
        }

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    total_wins = sum(float(t.pnl) for t in winners)
    total_losses = abs(sum(float(t.pnl) for t in losers))

    avg_win = total_wins / len(winners) if winners else 0
    avg_loss = total_losses / len(losers) if losers else 0

    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
    if profit_factor == float("inf"):
        profit_factor = 999.99

    # Calculate average trade duration
    durations = []
    for trade in trades:
        duration_ns = trade.exit_time - trade.entry_time
        duration_hours = duration_ns / (1_000_000_000 * 3600)
        durations.append(duration_hours)

    avg_duration = sum(durations) / len(durations) if durations else 0

    return {
        "total_trades": Decimal(str(len(trades))),
        "winning_trades": Decimal(str(len(winners))),
        "losing_trades": Decimal(str(len(losers))),
        "win_rate": Decimal(str(round(len(winners) / len(trades) * 100, 2))),
        "avg_win": Decimal(str(round(avg_win, 2))),
        "avg_loss": Decimal(str(round(avg_loss, 2))),
        "profit_factor": Decimal(str(round(profit_factor, 2))),
        "avg_trade_duration_hours": Decimal(str(round(avg_duration, 2))),
    }


def calculate_all_metrics(
    equity_curve: Sequence[tuple[Timestamp, Price]],
    trades: Sequence[TradeRecord],
    initial_capital: Price,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate all performance metrics.

    Args:
        equity_curve: Sequence of (timestamp, equity) tuples
        trades: Sequence of trade records
        initial_capital: Starting capital
        risk_free_rate: Annualized risk-free rate

    Returns:
        PerformanceMetrics dataclass
    """
    # Calculate returns
    if equity_curve:
        final_equity = equity_curve[-1][1]
    else:
        final_equity = initial_capital

    total_return = final_equity - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    # Calculate daily returns for risk metrics
    returns = calculate_returns_from_equity(equity_curve)

    # Estimate annualized return (assuming 252 trading days)
    if len(equity_curve) > 1:
        days = len(equity_curve)
        years = days / 252
        if years > 0:
            annualized = ((1 + float(total_return_pct) / 100) ** (1 / years) - 1) * 100
        else:
            annualized = 0.0
    else:
        annualized = 0.0

    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)

    # Trade metrics
    trade_metrics = calculate_trade_metrics(trades)

    return PerformanceMetrics(
        total_return=total_return,
        total_return_pct=Decimal(str(round(float(total_return_pct), 2))),
        annualized_return=Decimal(str(round(annualized, 2))),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        total_trades=int(trade_metrics["total_trades"]),
        winning_trades=int(trade_metrics["winning_trades"]),
        losing_trades=int(trade_metrics["losing_trades"]),
        win_rate=trade_metrics["win_rate"],
        avg_win=trade_metrics["avg_win"],
        avg_loss=trade_metrics["avg_loss"],
        profit_factor=trade_metrics["profit_factor"],
        avg_trade_duration_hours=trade_metrics["avg_trade_duration_hours"],
    )


def print_metrics(metrics: PerformanceMetrics) -> None:
    """Print metrics in a formatted table."""
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)

    print(f"\n{'RETURNS':-^50}")
    print(f"  Total Return:        ${metrics.total_return:,.2f}")
    print(f"  Total Return %:      {metrics.total_return_pct:.2f}%")
    print(f"  Annualized Return:   {metrics.annualized_return:.2f}%")

    print(f"\n{'RISK METRICS':-^50}")
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:       {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown:        ${metrics.max_drawdown:,.2f}")
    print(f"  Max Drawdown %:      {metrics.max_drawdown_pct:.2f}%")

    print(f"\n{'TRADE STATISTICS':-^50}")
    print(f"  Total Trades:        {metrics.total_trades}")
    print(f"  Winning Trades:      {metrics.winning_trades}")
    print(f"  Losing Trades:       {metrics.losing_trades}")
    print(f"  Win Rate:            {metrics.win_rate:.2f}%")
    print(f"  Avg Win:             ${metrics.avg_win:,.2f}")
    print(f"  Avg Loss:            ${metrics.avg_loss:,.2f}")
    print(f"  Profit Factor:       {metrics.profit_factor:.2f}")
    print(f"  Avg Duration (hrs):  {metrics.avg_trade_duration_hours:.1f}")

    print("=" * 50 + "\n")
