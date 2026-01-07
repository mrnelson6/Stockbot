"""Backtesting engine."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from stockbot.config.settings import BacktestConfig, RiskConfig
from stockbot.core.interfaces import Strategy
from stockbot.core.models import Bar, Fill, MarketState, PortfolioState, Position, TradeRecord
from stockbot.core.types import Price, Quantity, Signal, Symbol, Timeframe, Timestamp
from stockbot.data.providers.parquet import ParquetDataProvider
from stockbot.engine.clock import SimulatedClock
from stockbot.execution.broker_sim import SimulatedBroker
from stockbot.execution.translator import SignalTranslator
from stockbot.monitoring.logger import get_logger, setup_logging
from stockbot.risk.manager import BasicRiskManager

if TYPE_CHECKING:
    from stockbot.learning.callbacks import TradeCallback

logger = get_logger("backtest")


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Configuration
    start_date: str
    end_date: str
    symbols: list[Symbol]
    initial_capital: Price
    strategy_name: str

    # Performance
    final_equity: Price
    total_return: Decimal
    total_return_pct: Decimal

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Risk metrics
    max_drawdown: Decimal
    max_drawdown_pct: Decimal

    # Detailed records
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[Timestamp, Price]] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)


class BacktestEngine:
    """Event-driven backtesting engine.

    Replays historical data and simulates strategy execution.
    """

    def __init__(
        self,
        config: BacktestConfig,
        risk_config: RiskConfig,
        data_dir: Path,
        strategy: Strategy,
        trade_callback: Optional["TradeCallback"] = None,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            config: Backtest configuration
            risk_config: Risk management configuration
            data_dir: Directory containing parquet data files
            strategy: Strategy to backtest
            trade_callback: Optional callback for learning systems
        """
        self._config = config
        self._strategy = strategy
        self._data_dir = data_dir
        self._trade_callback = trade_callback

        # Parse dates to timestamps
        self._start_ts = self._parse_date(config.start_date)
        self._end_ts = self._parse_date(config.end_date)

        # Initialize components
        self._clock = SimulatedClock(self._start_ts)
        self._data_provider = ParquetDataProvider(data_dir)
        self._broker = SimulatedBroker(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage_pct=config.slippage_pct,
            seed=config.seed,
        )
        self._risk_manager = BasicRiskManager(risk_config)
        self._risk_manager.set_initial_capital(config.initial_capital)
        self._translator = SignalTranslator()

        # State tracking
        self._bars_by_symbol: dict[Symbol, list[Bar]] = {}
        self._equity_curve: list[tuple[Timestamp, Price]] = []
        self._peak_equity = config.initial_capital
        self._max_drawdown = Decimal("0")
        self._last_trade_count = 0  # Track trades for callback

    def _parse_date(self, date_str: str) -> Timestamp:
        """Parse ISO date string to timestamp."""
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        return Timestamp(int(dt.timestamp() * 1_000_000_000))

    def run(self) -> BacktestResult:
        """Run the backtest.

        Returns:
            BacktestResult with performance metrics and trade records
        """
        logger.info(
            f"Starting backtest: {self._config.start_date} to {self._config.end_date}",
            symbols=self._config.symbols,
            strategy=self._strategy.name,
        )

        # Load all bars for each symbol
        self._load_data()

        # Reset strategy
        self._strategy.reset()

        # Get all unique timestamps
        all_timestamps = self._get_all_timestamps()
        logger.info(f"Processing {len(all_timestamps)} bars")

        # Main simulation loop
        bar_count = 0
        for timestamp in all_timestamps:
            self._process_bar(timestamp)
            bar_count += 1

            if bar_count % 1000 == 0:
                logger.debug(f"Processed {bar_count} bars")

        # Notify callback of session end
        if self._trade_callback:
            self._trade_callback.on_session_end()

        # Build result
        return self._build_result()

    def _load_data(self) -> None:
        """Load historical data for all symbols."""
        for symbol in self._config.symbols:
            try:
                bars = list(
                    self._data_provider.get_bars(
                        symbol,
                        self._start_ts,
                        self._end_ts,
                        self._config.timeframe,
                    )
                )
                self._bars_by_symbol[symbol] = bars
                logger.info(f"Loaded {len(bars)} bars for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                self._bars_by_symbol[symbol] = []

    def _get_all_timestamps(self) -> list[Timestamp]:
        """Get sorted list of all unique timestamps across symbols."""
        timestamps: set[Timestamp] = set()
        for bars in self._bars_by_symbol.values():
            for bar in bars:
                timestamps.add(bar.timestamp)
        return sorted(timestamps)

    def _get_bars_at_timestamp(
        self, timestamp: Timestamp
    ) -> dict[Symbol, Optional[Bar]]:
        """Get bars for each symbol at a specific timestamp."""
        result: dict[Symbol, Optional[Bar]] = {}

        for symbol, bars in self._bars_by_symbol.items():
            # Find bar at this timestamp
            bar_at_ts = None
            for bar in bars:
                if bar.timestamp == timestamp:
                    bar_at_ts = bar
                    break
                elif bar.timestamp > timestamp:
                    break

            result[symbol] = bar_at_ts

        return result

    def _get_recent_bars(
        self, timestamp: Timestamp, lookback: int = 50
    ) -> dict[Symbol, list[Bar]]:
        """Get recent bars up to and including timestamp."""
        result: dict[Symbol, list[Bar]] = {}

        for symbol, bars in self._bars_by_symbol.items():
            recent = [b for b in bars if b.timestamp <= timestamp][-lookback:]
            result[symbol] = recent

        return result

    def _process_bar(self, timestamp: Timestamp) -> None:
        """Process a single bar/timestamp."""
        # Update clock
        self._clock.set_time(timestamp)

        # Get current bars and prices
        current_bars = self._get_bars_at_timestamp(timestamp)
        prices = {
            symbol: bar.close
            for symbol, bar in current_bars.items()
            if bar is not None
        }

        # Update broker with current prices
        self._broker.set_market_state(prices, timestamp)

        # Build market state for strategy
        recent_bars = self._get_recent_bars(timestamp)
        portfolio = self._build_portfolio_state(timestamp)
        market_state = MarketState(
            timestamp=timestamp,
            bars=recent_bars,
            portfolio=portfolio,
        )

        # Let strategy observe
        self._strategy.observe(market_state)

        # Get signals
        signals = self._strategy.decide()

        # Process each signal
        for symbol, signal in signals.items():
            if signal == Signal.HOLD:
                continue

            self._process_signal(symbol, signal, prices, portfolio, timestamp)

        # Track equity
        equity = self._broker.get_equity()
        self._equity_curve.append((timestamp, equity))

        # Update drawdown tracking
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        # Check emergency stop
        if self._risk_manager.check_emergency_stop(portfolio):
            logger.warning("Emergency stop triggered - closing all positions")
            self._close_all_positions(prices, timestamp)

        # Notify callback of any new completed trades
        self._check_for_completed_trades()

    def _process_signal(
        self,
        symbol: Symbol,
        signal: Signal,
        prices: dict[Symbol, Price],
        portfolio: PortfolioState,
        timestamp: Timestamp,
    ) -> None:
        """Process a single signal through risk check and execution."""
        if symbol not in prices:
            return

        price = prices[symbol]

        # Risk validation
        allowed, reason = self._risk_manager.validate_signal(
            signal, symbol, portfolio, price
        )

        if not allowed:
            return

        # Calculate position size
        quantity = self._risk_manager.calculate_position_size(
            signal, symbol, price, portfolio
        )

        if quantity <= Decimal("0"):
            return

        # Translate to order
        order = self._translator.translate(
            symbol=symbol,
            signal=signal,
            price=price,
            quantity=quantity,
            portfolio=portfolio,
            timestamp=timestamp,
            strategy_id=self._strategy.name,
        )

        if order is None:
            return

        # Submit order
        logger.order(order, "submitted")
        self._broker.submit_order(order)

        # Log fills
        for fill in self._broker.get_fills(order.id):
            logger.fill(fill)

    def _close_all_positions(
        self, prices: dict[Symbol, Price], timestamp: Timestamp
    ) -> None:
        """Close all open positions (emergency stop)."""
        portfolio = self._build_portfolio_state(timestamp)

        for symbol, position in portfolio.positions.items():
            if not position.is_flat:
                order = self._translator.translate(
                    symbol=symbol,
                    signal=Signal.CLOSE,
                    price=prices.get(symbol, position.average_price),
                    quantity=Quantity(abs(position.quantity)),
                    portfolio=portfolio,
                    timestamp=timestamp,
                    strategy_id=self._strategy.name,
                )
                if order:
                    self._broker.submit_order(order)

    def _check_for_completed_trades(self) -> None:
        """Check for newly completed trades and notify callback."""
        if not self._trade_callback:
            return

        # Build trades from all fills so far
        fills = self._broker.get_all_fills()
        trades = self._build_trade_records(fills)

        # Notify callback of new trades
        new_trade_count = len(trades)
        if new_trade_count > self._last_trade_count:
            for trade in trades[self._last_trade_count:]:
                self._trade_callback.on_trade_complete(trade)
            self._last_trade_count = new_trade_count

    def _build_portfolio_state(self, timestamp: Timestamp) -> PortfolioState:
        """Build current portfolio state from broker."""
        return PortfolioState(
            timestamp=timestamp,
            cash=self._broker.get_cash(),
            positions=self._broker.get_positions(),
            pending_orders=[],
        )

    def _build_result(self) -> BacktestResult:
        """Build the final backtest result."""
        final_equity = self._broker.get_equity()
        initial = self._config.initial_capital
        total_return = final_equity - initial
        total_return_pct = (total_return / initial) * 100

        # Get all fills
        fills = self._broker.get_all_fills()

        # Build trade records from fills
        trades = self._build_trade_records(fills)

        winning = sum(1 for t in trades if t.is_winner)
        losing = len(trades) - winning
        win_rate = Decimal(winning) / Decimal(len(trades)) * 100 if trades else Decimal("0")

        return BacktestResult(
            start_date=self._config.start_date,
            end_date=self._config.end_date,
            symbols=self._config.symbols,
            initial_capital=initial,
            strategy_name=self._strategy.name,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=len(trades),
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            max_drawdown=self._max_drawdown * self._peak_equity,
            max_drawdown_pct=self._max_drawdown * 100,
            trades=trades,
            equity_curve=self._equity_curve,
            fills=fills,
        )

    def _build_trade_records(self, fills: list[Fill]) -> list[TradeRecord]:
        """Build trade records from fills."""
        # Group fills by symbol
        fills_by_symbol: dict[Symbol, list[Fill]] = {}
        for fill in fills:
            if fill.symbol not in fills_by_symbol:
                fills_by_symbol[fill.symbol] = []
            fills_by_symbol[fill.symbol].append(fill)

        trades: list[TradeRecord] = []

        for symbol, symbol_fills in fills_by_symbol.items():
            # Match entries with exits
            position_qty = Decimal("0")
            entry_fills: list[Fill] = []

            for fill in sorted(symbol_fills, key=lambda f: f.timestamp):
                qty_change = fill.quantity if fill.side.name == "BUY" else -fill.quantity
                new_qty = position_qty + qty_change

                if position_qty == Decimal("0"):
                    # Opening new position
                    entry_fills = [fill]
                    position_qty = new_qty

                elif (position_qty > 0 and new_qty <= 0) or (
                    position_qty < 0 and new_qty >= 0
                ):
                    # Closing position
                    avg_entry = sum(f.price * f.quantity for f in entry_fills) / sum(
                        f.quantity for f in entry_fills
                    )
                    closed_qty = min(abs(position_qty), fill.quantity)
                    pnl = (fill.price - avg_entry) * closed_qty
                    if position_qty < 0:
                        pnl = -pnl
                    total_commission = sum(f.commission for f in entry_fills) + fill.commission

                    trades.append(
                        TradeRecord(
                            entry_time=entry_fills[0].timestamp,
                            exit_time=fill.timestamp,
                            symbol=symbol,
                            side=entry_fills[0].side,
                            quantity=Quantity(closed_qty),
                            entry_price=Price(avg_entry),
                            exit_price=fill.price,
                            pnl=Price(pnl - total_commission),
                            commission=Price(total_commission),
                            strategy_id=self._strategy.name,
                        )
                    )

                    position_qty = new_qty
                    if new_qty != Decimal("0"):
                        entry_fills = [fill]
                    else:
                        entry_fills = []

                else:
                    # Adding to position
                    entry_fills.append(fill)
                    position_qty = new_qty

        return trades


def run_backtest(
    config: BacktestConfig,
    risk_config: RiskConfig,
    data_dir: Path,
    strategy: Strategy,
    log_level: str = "INFO",
    trade_callback: Optional["TradeCallback"] = None,
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        config: Backtest configuration
        risk_config: Risk configuration
        data_dir: Data directory
        strategy: Strategy to test
        log_level: Logging level
        trade_callback: Optional callback for learning systems

    Returns:
        BacktestResult
    """
    setup_logging(level=log_level)

    engine = BacktestEngine(
        config=config,
        risk_config=risk_config,
        data_dir=data_dir,
        strategy=strategy,
        trade_callback=trade_callback,
    )

    return engine.run()
