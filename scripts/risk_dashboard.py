#!/usr/bin/env python3
"""Risk dashboard - view current risk status and controls."""

import argparse
from decimal import Decimal
from pathlib import Path

from stockbot.config import load_settings
from stockbot.core.models import PortfolioState
from stockbot.core.types import Price, Timestamp
from stockbot.execution.broker_alpaca import AlpacaBroker
from stockbot.risk.exposure import ExposureTracker
from stockbot.risk.kill_switch import KillSwitch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Risk dashboard")

    parser.add_argument(
        "--reset-kill-switch",
        action="store_true",
        help="Reset the kill switch",
    )

    parser.add_argument(
        "--trigger-kill-switch",
        type=str,
        metavar="REASON",
        help="Trigger the kill switch with given reason",
    )

    return parser.parse_args()


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print("=" * 60)


def print_row(label: str, value: str, width: int = 30) -> None:
    """Print a formatted row."""
    print(f"  {label:<{width}} {value}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle kill switch commands first
    kill_switch = KillSwitch()

    if args.reset_kill_switch:
        kill_switch.reset("manual_cli")
        print("Kill switch has been RESET")
        return 0

    if args.trigger_kill_switch:
        kill_switch.trigger(args.trigger_kill_switch, "manual_cli")
        print(f"Kill switch TRIGGERED: {args.trigger_kill_switch}")
        return 0

    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        return 1

    if settings.alpaca is None:
        print("Alpaca credentials not configured")
        return 1

    # Connect to broker
    print("Connecting to Alpaca...")
    broker = AlpacaBroker(settings.alpaca)

    # Get account info
    account = broker.get_account_info()
    equity = broker.get_equity()
    cash = broker.get_cash()
    buying_power = broker.get_buying_power()
    positions = broker.get_positions()

    # Create exposure tracker
    tracker = ExposureTracker(initial_equity=equity)

    # Build portfolio state
    import time

    timestamp = Timestamp(int(time.time() * 1_000_000_000))
    portfolio = PortfolioState(
        timestamp=timestamp,
        cash=cash,
        positions=positions,
    )

    # Calculate exposure
    exposure = tracker.calculate_exposure(portfolio, timestamp)

    # Print dashboard
    print_header("ACCOUNT STATUS")
    print_row("Account ID:", account.get("id", "N/A"))
    print_row("Status:", account.get("status", "N/A"))
    print_row("Mode:", "PAPER" if settings.alpaca.paper else "LIVE")
    print_row("Market Open:", "Yes" if broker.is_market_open() else "No")

    print_header("PORTFOLIO")
    print_row("Equity:", f"${equity:,.2f}")
    print_row("Cash:", f"${cash:,.2f}")
    print_row("Buying Power:", f"${buying_power:,.2f}")
    print_row("Positions:", str(len(positions)))

    print_header("EXPOSURE")
    print_row("Net Exposure:", f"${exposure.net_exposure:,.2f}")
    print_row("Net Exposure %:", f"{exposure.net_exposure_pct:.1f}%")
    print_row("Gross Exposure:", f"${exposure.gross_exposure:,.2f}")
    print_row("Gross Exposure %:", f"{exposure.gross_exposure_pct:.1f}%")
    print_row("Cash %:", f"{exposure.cash_pct:.1f}%")
    print_row("Long Positions:", str(exposure.num_long_positions))
    print_row("Short Positions:", str(exposure.num_short_positions))

    if exposure.largest_position_symbol:
        print_row(
            "Largest Position:",
            f"{exposure.largest_position_symbol} ({exposure.largest_position_pct:.1f}%)",
        )

    print_header("P&L")
    print_row("Unrealized P&L:", f"${exposure.unrealized_pnl:,.2f}")
    print_row("Unrealized P&L %:", f"{exposure.unrealized_pnl_pct:.2f}%")

    print_header("POSITIONS")
    if positions:
        print(f"  {'Symbol':<10} {'Qty':>10} {'Avg Price':>12} {'Unrealized':>12}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12}")
        for symbol, pos in positions.items():
            print(
                f"  {symbol:<10} {pos.quantity:>10} "
                f"${pos.average_price:>10,.2f} "
                f"${pos.unrealized_pnl:>10,.2f}"
            )
    else:
        print("  No open positions")

    print_header("KILL SWITCH")
    kill_switch.print_status()

    print_header("RISK CONTROLS")
    print_row("Max Position Size:", f"{settings.risk.max_position_size} shares")
    print_row("Max Position Value:", f"${settings.risk.max_position_value:,.2f}")
    print_row("Max Daily Loss:", f"${settings.risk.max_daily_loss:,.2f}")
    print_row("Max Open Positions:", str(settings.risk.max_open_positions))

    # Warnings
    warnings = []

    if kill_switch.is_active:
        warnings.append("KILL SWITCH IS ACTIVE - Trading halted")

    if exposure.largest_position_pct > Decimal("25"):
        warnings.append(
            f"Position concentration high: {exposure.largest_position_symbol} "
            f"is {exposure.largest_position_pct:.1f}% of portfolio"
        )

    if exposure.gross_exposure_pct > Decimal("100"):
        warnings.append(f"Gross exposure {exposure.gross_exposure_pct:.1f}% exceeds 100%")

    if account.get("trading_blocked"):
        warnings.append("Trading is BLOCKED on this account")

    if warnings:
        print_header("⚠️  WARNINGS")
        for warning in warnings:
            print(f"  • {warning}")

    print()
    return 0


if __name__ == "__main__":
    exit(main())
