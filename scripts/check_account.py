#!/usr/bin/env python3
"""Check Alpaca account status and connection."""

from stockbot.config import load_settings
from stockbot.execution.broker_alpaca import AlpacaBroker


def main() -> int:
    """Check account status."""
    print("Loading settings...")

    try:
        settings = load_settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        return 1

    if settings.alpaca is None:
        print("Alpaca credentials not configured")
        return 1

    print(f"Alpaca mode: {'PAPER' if settings.alpaca.paper else 'LIVE'}")
    print(f"Connecting to: {settings.alpaca.base_url}")
    print()

    # Create broker
    broker = AlpacaBroker(settings.alpaca)

    # Check market status
    print(f"Market open: {broker.is_market_open()}")
    print()

    # Get account info
    print("Account Info:")
    print("-" * 40)
    account = broker.get_account_info()
    for key, value in account.items():
        print(f"  {key}: {value}")
    print()

    # Get positions
    print("Current Positions:")
    print("-" * 40)
    positions = broker.get_positions()
    if positions:
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos.quantity} shares @ ${pos.average_price}")
            print(f"    Unrealized P&L: ${pos.unrealized_pnl}")
    else:
        print("  No open positions")
    print()

    # Summary
    print("Summary:")
    print("-" * 40)
    print(f"  Cash: ${broker.get_cash()}")
    print(f"  Equity: ${broker.get_equity()}")
    print(f"  Buying Power: ${broker.get_buying_power()}")

    return 0


if __name__ == "__main__":
    exit(main())
