"""
Rust Integration Layer - Connects Python orchestration with Rust core modules
Provides seamless access to high-performance Rust implementations
"""

import asyncio
from typing import Any

from loguru import logger

# Import Rust core modules (compiled via PyO3)
try:
    from cift_core import FastMarketData, FastOrderBook, FastRiskEngine

    RUST_AVAILABLE = True
    logger.info("✓ Rust core modules loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"⚠ Rust core not available, using Python fallback: {e}")
    logger.warning("To build Rust core: cd rust_core && maturin develop")


class RustOrderBookManager:
    """
    High-performance order book manager using Rust backend

    Performance: <10μs per match (100x faster than Python)
    """

    def __init__(self):
        self.books: dict[str, Any] = {}  # symbol -> FastOrderBook
        self.use_rust = RUST_AVAILABLE

    def get_or_create_book(self, symbol: str):
        """Get or create order book for symbol"""
        if symbol not in self.books:
            if self.use_rust:
                self.books[symbol] = FastOrderBook(symbol)
                logger.debug(f"Created Rust order book for {symbol}")
            else:
                # Fallback to Python implementation
                from cift.core.order_book_fallback import PythonOrderBook

                self.books[symbol] = PythonOrderBook(symbol)
                logger.debug(f"Created Python order book for {symbol}")

        return self.books[symbol]

    async def add_limit_order(
        self,
        symbol: str,
        order_id: int,
        side: str,
        price: float,
        quantity: float,
        user_id: int,
    ) -> tuple[int, list[tuple[float, float, int]]]:
        """
        Add limit order to book and get fills

        Returns:
            (order_id, fills) where fills = [(price, quantity, counterparty_id), ...]
        """
        book = self.get_or_create_book(symbol)

        # Execute in thread pool to avoid blocking (Rust is CPU-bound)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            book.add_limit_order,
            order_id,
            side,
            price,
            quantity,
            user_id,
        )

        return result

    async def add_market_order(
        self,
        symbol: str,
        order_id: int,
        side: str,
        quantity: float,
        user_id: int,
    ) -> list[tuple[float, float, int]]:
        """Add market order and get immediate fills"""
        book = self.get_or_create_book(symbol)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            book.add_market_order,
            order_id,
            side,
            quantity,
            user_id,
        )

        return result

    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel order by ID"""
        if symbol not in self.books:
            return False

        book = self.books[symbol]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, book.cancel_order, order_id)

    async def get_best_prices(self, symbol: str) -> tuple[float | None, float | None]:
        """Get best bid and ask prices"""
        if symbol not in self.books:
            return None, None

        book = self.books[symbol]
        loop = asyncio.get_event_loop()

        bid = await loop.run_in_executor(None, book.best_bid)
        ask = await loop.run_in_executor(None, book.best_ask)

        return bid, ask

    async def get_spread(self, symbol: str) -> float | None:
        """Get bid-ask spread"""
        if symbol not in self.books:
            return None

        book = self.books[symbol]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, book.spread)

    async def get_depth(
        self, symbol: str, levels: int = 10
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Get order book depth (top N levels)"""
        if symbol not in self.books:
            return [], []

        book = self.books[symbol]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, book.depth, levels)


class RustMarketDataProcessor:
    """
    High-performance market data processor using Rust backend

    Performance: 100x faster than Python
    """

    def __init__(self):
        if RUST_AVAILABLE:
            self.processor = FastMarketData()
        else:
            # Fallback to Python/Numba implementation
            self.processor = None

        self.use_rust = RUST_AVAILABLE

    async def calculate_vwap(self, ticks: list[tuple[float, float]]) -> float:
        """Calculate VWAP from tick data"""
        if self.use_rust:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.processor.calculate_vwap, ticks)
        else:
            # Use Numba fallback
            import numpy as np

            from cift.core.features_numba import calculate_vwap

            prices = np.array([t[0] for t in ticks])
            volumes = np.array([t[1] for t in ticks])
            return calculate_vwap(prices, volumes)

    async def calculate_ofi(self, bid_volumes: list[float], ask_volumes: list[float]) -> float:
        """Calculate Order Flow Imbalance"""
        if self.use_rust:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.processor.calculate_ofi, bid_volumes, ask_volumes
            )
        else:
            # Use Numba fallback
            import numpy as np

            from cift.core.features_numba import calculate_ofi

            bids = np.array(bid_volumes)
            asks = np.array(ask_volumes)
            return calculate_ofi(bids, asks, levels=len(bid_volumes))

    async def calculate_microprice(
        self,
        best_bid: float,
        best_ask: float,
        bid_volume: float,
        ask_volume: float,
    ) -> float:
        """Calculate microprice"""
        if self.use_rust:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.processor.calculate_microprice,
                best_bid,
                best_ask,
                bid_volume,
                ask_volume,
            )
        else:
            # Simple fallback
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                return (best_bid * ask_volume + best_ask * bid_volume) / total_volume
            else:
                return (best_bid + best_ask) / 2


class RustRiskManager:
    """
    High-performance risk engine using Rust backend

    Performance: <1μs per check (100x faster than Python)
    """

    def __init__(
        self,
        max_position_size: float = 10000.0,
        max_notional: float = 1_000_000.0,
        max_leverage: float = 5.0,
    ):
        if RUST_AVAILABLE:
            self.engine = FastRiskEngine(max_position_size, max_notional, max_leverage)
        else:
            # Fallback to Python implementation
            self.max_position_size = max_position_size
            self.max_notional = max_notional
            self.max_leverage = max_leverage

        self.use_rust = RUST_AVAILABLE

    async def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_position: float,
        account_value: float,
    ) -> tuple[bool, str]:
        """
        Check if order passes risk limits

        Returns:
            (passed, reason)
        """
        if self.use_rust:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.engine.check_order,
                symbol,
                side,
                quantity,
                price,
                current_position,
                account_value,
            )
        else:
            # Python fallback
            return self._check_order_python(
                symbol, side, quantity, price, current_position, account_value
            )

    def _check_order_python(self, symbol, side, quantity, price, current_position, account_value):
        """Python fallback for risk checks"""
        new_position = current_position + quantity if side == "buy" else current_position - quantity

        if abs(new_position) > self.max_position_size:
            return False, f"Position size {abs(new_position)} exceeds max {self.max_position_size}"

        order_notional = quantity * price
        if order_notional > self.max_notional:
            return False, f"Order notional {order_notional} exceeds max {self.max_notional}"

        return True, "OK"

    async def max_order_size(
        self,
        symbol: str,
        side: str,
        price: float,
        current_position: float,
        account_value: float,
    ) -> float:
        """Calculate maximum allowed order size"""
        if self.use_rust:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.engine.max_order_size,
                symbol,
                side,
                price,
                current_position,
                account_value,
            )
        else:
            # Python fallback
            max_from_position = (
                self.max_position_size - current_position
                if side == "buy"
                else self.max_position_size + current_position
            )
            max_from_notional = self.max_notional / price if price > 0 else 0
            return min(max_from_position, max_from_notional, account_value / price)


# =====================================================
# GLOBAL INSTANCES (Singletons)
# =====================================================

_order_book_manager: RustOrderBookManager | None = None
_market_data_processor: RustMarketDataProcessor | None = None
_risk_manager: RustRiskManager | None = None


def get_order_book_manager() -> RustOrderBookManager:
    """Get global order book manager"""
    global _order_book_manager
    if _order_book_manager is None:
        _order_book_manager = RustOrderBookManager()
    return _order_book_manager


def get_market_data_processor() -> RustMarketDataProcessor:
    """Get global market data processor"""
    global _market_data_processor
    if _market_data_processor is None:
        _market_data_processor = RustMarketDataProcessor()
    return _market_data_processor


def get_risk_manager(
    max_position_size: float = 10000.0,
    max_notional: float = 1_000_000.0,
    max_leverage: float = 5.0,
) -> RustRiskManager:
    """Get global risk manager"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RustRiskManager(max_position_size, max_notional, max_leverage)
    return _risk_manager
