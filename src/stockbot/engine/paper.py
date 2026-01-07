"""Paper trading engine with live market data."""

import signal
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from stockbot.config.settings import AlpacaConfig, RiskConfig
from stockbot.core.interfaces import Strategy
from stockbot.core.models import Bar, MarketState, PortfolioState, Position, TradeRecord
from stockbot.core.types import OrderSide, Price, Quantity, Signal, Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.engine.clock import RealClock
from stockbot.execution.broker_alpaca import AlpacaBroker
from stockbot.execution.order_manager import OrderManager
from stockbot.monitoring.logger import get_logger, setup_logging
from stockbot.risk.manager import BasicRiskManager

if TYPE_CHECKING:
    from stockbot.learning.callbacks import TradeCallback

logger = get_logger("paper")


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""

    symbols: list[Symbol]
    alpaca_config: AlpacaConfig
    risk_config: RiskConfig
    poll_interval_seconds: float = 60.0  # How often to check market
    bar_lookback: int = 50  # Number of historical bars for strategy
    timeframe: Timeframe = Timeframe.MINUTE_1


class PaperTradingEngine:
    """Paper trading engine using Alpaca paper trading.

    Runs the same strategy code as backtesting but with:
    - Live market data from Alpaca
    - Real order submission to Alpaca paper trading
    - Real-time portfolio tracking
    """

    def __init__(
        self,
        config: PaperTradingConfig,
        strategy: Strategy,
        trade_callback: Optional["TradeCallback"] = None,
    ) -> None:
        """Initialize the paper trading engine.

        Args:
            config: Paper trading configuration
            strategy: Strategy to run
            trade_callback: Optional callback for learning systems
        """
        self._config = config
        self._strategy = strategy
        self._trade_callback = trade_callback
        self._running = False
        self._emergency_stop = False

        # Initialize components
        self._clock = RealClock()
        self._data_provider = AlpacaDataProvider(config.alpaca_config)
        self._broker = AlpacaBroker(config.alpaca_config)
        self._risk_manager = BasicRiskManager(config.risk_config)
        self._order_manager = OrderManager(
            broker=self._broker,
            max_orders_per_minute=config.risk_config.max_orders_per_minute,
        )

        # Set initial capital from account
        initial_equity = self._broker.get_equity()
        self._risk_manager.set_initial_capital(initial_equity)

        # Historical bar cache
        self._bar_cache: dict[Symbol, list[Bar]] = {s: [] for s in config.symbols}

        # Track positions for trade completion detection
        self._previous_positions: dict[Symbol, Position] = {}

        logger.info(
            f"Paper trading engine initialized",
            symbols=config.symbols,
            strategy=strategy.name,
            initial_equity=str(initial_equity),
        )

    def run(self) -> None:
        """Run the paper trading loop.

        Runs until stopped via stop() or SIGINT.
        """
        self._running = True
        self._setup_signal_handlers()

        logger.info("Starting paper trading loop")
        self._strategy.reset()

        # Set day start equity for risk tracking
        self._risk_manager.set_day_start_equity(self._broker.get_equity())

        try:
            while self._running:
                try:
                    self._run_iteration()
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")

                # Wait for next iteration
                if self._running:
                    time.sleep(self._config.poll_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        finally:
            self._shutdown()

    def _run_iteration(self) -> None:
        """Run a single iteration of the trading loop."""
        # Check if market is open
        if not self._broker.is_market_open():
            logger.debug("Market is closed, skipping iteration")
            return

        # Check emergency stop
        if self._emergency_stop:
            logger.warning("Emergency stop active, skipping iteration")
            return

        # Get current timestamp
        timestamp = self._clock.now

        # Update bar cache with latest data
        self._update_bar_cache()

        # Build market state
        portfolio = self._build_portfolio_state(timestamp)
        market_state = MarketState(
            timestamp=timestamp,
            bars=dict(self._bar_cache),
            portfolio=portfolio,
        )

        # Check risk manager emergency stop
        if self._risk_manager.check_emergency_stop(portfolio):
            logger.warning("Risk manager triggered emergency stop")
            self._emergency_stop = True
            self._close_all_positions()
            return

        # Let strategy observe
        self._strategy.observe(market_state)

        # Get signals
        signals = self._strategy.decide()

        # Process each signal
        for symbol, signal in signals.items():
            if signal == Signal.HOLD:
                continue

            self._process_signal(symbol, signal, portfolio, timestamp)

        # Update pending order statuses
        self._order_manager.update_order_statuses()

        # Check for completed trades (positions that closed)
        self._check_for_completed_trades()

        # Log current state
        equity = self._broker.get_equity()
        positions = self._broker.get_positions()
        logger.info(
            f"Iteration complete",
            equity=str(equity),
            positions=len(positions),
            pending_orders=len(self._order_manager.get_pending_orders()),
        )

    def _update_bar_cache(self) -> None:
        """Update the bar cache with latest data."""
        for symbol in self._config.symbols:
            try:
                latest = self._data_provider.get_latest(symbol)
                if latest:
                    cache = self._bar_cache[symbol]

                    # Avoid duplicates
                    if not cache or cache[-1].timestamp != latest.timestamp:
                        cache.append(latest)

                        # Keep only lookback bars
                        if len(cache) > self._config.bar_lookback:
                            self._bar_cache[symbol] = cache[-self._config.bar_lookback :]

            except Exception as e:
                logger.warning(f"Failed to get latest bar for {symbol}: {e}")

    def _build_portfolio_state(self, timestamp: Timestamp) -> PortfolioState:
        """Build current portfolio state from broker."""
        return PortfolioState(
            timestamp=timestamp,
            cash=self._broker.get_cash(),
            positions=self._broker.get_positions(),
            pending_orders=self._order_manager.get_pending_orders(),
        )

    def _process_signal(
        self,
        symbol: Symbol,
        signal: Signal,
        portfolio: PortfolioState,
        timestamp: Timestamp,
    ) -> None:
        """Process a trading signal."""
        # Get current price
        bars = self._bar_cache.get(symbol, [])
        if not bars:
            logger.warning(f"No price data for {symbol}, skipping signal")
            return

        price = bars[-1].close

        # Validate with risk manager
        allowed, reason = self._risk_manager.validate_signal(
            signal, symbol, portfolio, price
        )

        if not allowed:
            logger.warning(f"Signal rejected for {symbol}: {reason}")
            return

        # Calculate position size
        quantity = self._risk_manager.calculate_position_size(
            signal, symbol, price, portfolio
        )

        if quantity <= Decimal("0"):
            logger.debug(f"Zero quantity for {symbol}, skipping")
            return

        # Submit via order manager
        try:
            order_id = self._order_manager.submit_signal(
                symbol=symbol,
                signal=signal,
                price=price,
                quantity=quantity,
                portfolio=portfolio,
                timestamp=timestamp,
                strategy_id=self._strategy.name,
            )

            if order_id:
                logger.info(f"Order submitted for {symbol}: {order_id}")

        except Exception as e:
            logger.error(f"Failed to submit order for {symbol}: {e}")

    def _close_all_positions(self) -> None:
        """Close all positions (emergency stop)."""
        logger.warning("Closing all positions")

        # Cancel pending orders
        self._order_manager.cancel_all_pending()

        # Close positions via broker
        self._broker.close_all_positions()

    def _check_for_completed_trades(self) -> None:
        """Detect closed positions and notify callback."""
        if not self._trade_callback:
            return

        current_positions = self._broker.get_positions()
        timestamp = self._clock.now

        # Check for positions that were open but are now closed or flat
        for symbol, prev_pos in self._previous_positions.items():
            if prev_pos.is_flat:
                continue

            curr_pos = current_positions.get(symbol)
            is_now_flat = curr_pos is None or curr_pos.is_flat

            if is_now_flat:
                # Position was closed - create trade record
                bars = self._bar_cache.get(symbol, [])
                exit_price = bars[-1].close if bars else prev_pos.average_price

                # Determine side from original position
                side = OrderSide.BUY if prev_pos.quantity > Decimal("0") else OrderSide.SELL

                # Calculate P&L
                qty = abs(prev_pos.quantity)
                if side == OrderSide.BUY:
                    pnl = (exit_price - prev_pos.average_price) * qty
                else:
                    pnl = (prev_pos.average_price - exit_price) * qty

                trade = TradeRecord(
                    entry_time=prev_pos.timestamp,
                    exit_time=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=Quantity(qty),
                    entry_price=prev_pos.average_price,
                    exit_price=exit_price,
                    pnl=Price(pnl),
                    commission=Price(Decimal("0")),  # Alpaca paper is commission-free
                    strategy_id=self._strategy.name,
                )

                self._trade_callback.on_trade_complete(trade)
                logger.info(f"Trade completed: {symbol} P&L=${pnl:.2f}")

        # Update previous positions
        self._previous_positions = dict(current_positions)

    def stop(self) -> None:
        """Stop the trading loop."""
        logger.info("Stop requested")
        self._running = False

    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop.

        Args:
            reason: Reason for stopping
        """
        logger.error(f"Emergency stop triggered: {reason}")
        self._emergency_stop = True
        self._risk_manager.trigger_emergency_stop(reason)
        self._close_all_positions()

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down paper trading engine")

        # Cancel any pending orders
        self._order_manager.cancel_all_pending()

        # Notify callback of session end
        if self._trade_callback:
            self._trade_callback.on_session_end()

        logger.info("Shutdown complete")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            if not self._running:
                # Already stopping, force exit on second signal
                logger.warning("Force exit")
                raise SystemExit(1)
            logger.info(f"Received signal {signum}, stopping...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def get_status(self) -> dict:
        """Get current engine status.

        Returns:
            Dict with status information
        """
        return {
            "running": self._running,
            "emergency_stop": self._emergency_stop,
            "strategy": self._strategy.name,
            "symbols": self._config.symbols,
            "market_open": self._broker.is_market_open(),
            "equity": str(self._broker.get_equity()),
            "cash": str(self._broker.get_cash()),
            "positions": len(self._broker.get_positions()),
            "pending_orders": len(self._order_manager.get_pending_orders()),
        }


def run_paper_trading(
    config: PaperTradingConfig,
    strategy: Strategy,
    log_level: str = "INFO",
    trade_callback: Optional["TradeCallback"] = None,
) -> None:
    """Convenience function to run paper trading.

    Args:
        config: Paper trading configuration
        strategy: Strategy to run
        log_level: Logging level
        trade_callback: Optional callback for learning systems
    """
    setup_logging(level=log_level)

    engine = PaperTradingEngine(
        config=config,
        strategy=strategy,
        trade_callback=trade_callback,
    )
    engine.run()
