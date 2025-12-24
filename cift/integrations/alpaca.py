"""
CIFT Markets - Alpaca Integration

Real-time market data and trading execution via Alpaca API.

Features:
- Real-time market data streaming
- Historical data retrieval
- Order submission and management
- Account information
- WebSocket streaming

API Documentation: https://alpaca.markets/docs/api-documentation/
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import aiohttp
from loguru import logger

from cift.core.config import settings

# ============================================================================
# CONSTANTS
# ============================================================================


class AlpacaEndpoint(str, Enum):
    """Alpaca API endpoints."""

    PAPER = "https://paper-api.alpaca.markets"
    LIVE = "https://api.alpaca.markets"
    BROKER = "https://broker-api.alpaca.markets"  # Broker API for omnibus model
    BROKER_SANDBOX = "https://broker-api.sandbox.alpaca.markets"  # Sandbox for testing
    DATA = "https://data.alpaca.markets"
    STREAM = "wss://stream.data.alpaca.markets"


class TimeFrame(str, Enum):
    """Alpaca bar timeframes."""

    MIN_1 = "1Min"
    MIN_5 = "5Min"
    MIN_15 = "15Min"
    HOUR = "1Hour"
    DAY = "1Day"


# ============================================================================
# ALPACA CLIENT
# ============================================================================


class AlpacaClient:
    """
    Async Alpaca API client for market data and trading.

    Supports both Trading API (individual accounts) and Broker API (omnibus model).
    High-performance async implementation with connection pooling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        use_broker_api: bool | None = None,
    ):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (defaults to config)
            secret_key: Alpaca secret key (defaults to config)
            use_broker_api: Use Broker API (omnibus model) instead of Trading API
        """
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.use_broker_api = use_broker_api if use_broker_api is not None else settings.alpaca_use_broker_api

        # Set base URL based on API type
        if self.use_broker_api:
            self.base_url = settings.alpaca_base_url or AlpacaEndpoint.BROKER
        else:
            self.base_url = AlpacaEndpoint.PAPER
        
        self.data_url = AlpacaEndpoint.DATA

        self.session: aiohttp.ClientSession | None = None
        self._initialized = False

    @property
    def is_configured(self) -> bool:
        """Check if API keys are configured."""
        return bool(self.api_key and self.secret_key)

    async def initialize(self):
        """Initialize HTTP session with connection pooling."""
        if self._initialized:
            return

        if not self.is_configured:
            logger.warning("Alpaca client initialized without API keys")

        connector = aiohttp.TCPConnector(
            limit=100, limit_per_host=30, ttl_dns_cache=300  # Max connections
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=self._get_headers()
        )

        self._initialized = True
        api_type = "Broker API" if self.use_broker_api else "Trading API"
        logger.info(f"Alpaca client initialized ({api_type})")

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self._initialized = False
            logger.info("Alpaca client closed")

    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers based on API type."""
        if self.use_broker_api:
            # Broker API uses HTTP Basic Auth
            import base64
            credentials = base64.b64encode(f"{self.api_key}:{self.secret_key}".encode()).decode()
            return {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
            }
        else:
            # Trading API uses custom headers
            return {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json: dict | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Make HTTP request to Alpaca API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            base_url: Override base URL

        Returns:
            JSON response

        Raises:
            Exception: If request fails
        """
        if not self._initialized:
            await self.initialize()

        if not self.is_configured:
            raise ValueError(
                "Alpaca API keys not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        url = f"{base_url or self.base_url}{endpoint}"

        try:
            async with self.session.request(method, url, params=params, json=json) as response:
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            logger.error(f"Alpaca API error: {e.status} - {e.message}")
            raise

        except Exception as e:
            logger.error(f"Alpaca request failed: {e}")
            raise

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    async def get_latest_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest quote data
        """
        endpoint = f"/v2/stocks/{symbol}/quotes/latest"
        return await self._request("GET", endpoint, base_url=self.data_url)

    async def get_latest_trade(self, symbol: str) -> dict[str, Any]:
        """
        Get latest trade for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest trade data
        """
        endpoint = f"/v2/stocks/{symbol}/trades/latest"
        return await self._request("GET", endpoint, base_url=self.data_url)

    async def get_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame = TimeFrame.MIN_1,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """
        Get historical bars (OHLCV) for symbols.

        Args:
            symbols: List of symbols
            timeframe: Bar timeframe
            start: Start date
            end: End date
            limit: Max bars to return

        Returns:
            Historical bar data
        """
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe.value,
            "limit": limit,
        }

        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        endpoint = "/v2/stocks/bars"
        return await self._request("GET", endpoint, params=params, base_url=self.data_url)

    async def get_snapshot(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get market snapshot for symbols (quotes, trades, bars).

        Args:
            symbols: List of symbols

        Returns:
            Snapshot data for all symbols
        """
        params = {"symbols": ",".join(symbols)}
        endpoint = "/v2/stocks/snapshots"
        return await self._request("GET", endpoint, params=params, base_url=self.data_url)

    # ========================================================================
    # TRADING (Works for both Trading API and Broker API)
    # ========================================================================

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
        account_id: str | None = None,  # For Broker API: specify trading account
    ) -> dict[str, Any]:
        """
        Submit order to Alpaca.

        For Broker API (omnibus model): Uses the sweep/trading account to execute orders.
        The account_id parameter specifies which trading account to use.

        Args:
            symbol: Stock symbol
            qty: Order quantity
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Client-assigned order ID
            account_id: Trading account ID (Broker API only)

        Returns:
            Order data
        """
        data = {
            "symbol": symbol,
            "qty": str(qty),  # Broker API expects string
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        if limit_price:
            data["limit_price"] = str(limit_price)
        if stop_price:
            data["stop_price"] = str(stop_price)
        if client_order_id:
            data["client_order_id"] = client_order_id

        # Broker API uses different endpoint structure
        if self.use_broker_api:
            # Use the configured sweep account or provided account_id
            trading_account = account_id or settings.alpaca_sweep_account_id
            if not trading_account:
                raise ValueError("Broker API requires a trading account ID (ALPACA_SWEEP_ACCOUNT_ID)")
            endpoint = f"/v1/trading/accounts/{trading_account}/orders"
        else:
            endpoint = "/v2/orders"
            
        return await self._request("POST", endpoint, json=data)

    async def get_order(self, order_id: str, account_id: str | None = None) -> dict[str, Any]:
        """Get order by ID."""
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/orders/{order_id}"
        else:
            endpoint = f"/v2/orders/{order_id}"
        return await self._request("GET", endpoint)

    async def cancel_order(self, order_id: str, account_id: str | None = None) -> dict[str, Any]:
        """Cancel order by ID."""
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/orders/{order_id}"
        else:
            endpoint = f"/v2/orders/{order_id}"
        return await self._request("DELETE", endpoint)

    async def get_open_orders(self, account_id: str | None = None) -> list[dict[str, Any]]:
        """Get all open orders."""
        params = {"status": "open"}
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/orders"
        else:
            endpoint = "/v2/orders"
        return await self._request("GET", endpoint, params=params)

    async def get_positions(self, account_id: str | None = None) -> list[dict[str, Any]]:
        """Get all positions in the trading account."""
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/positions"
        else:
            endpoint = "/v2/positions"
        return await self._request("GET", endpoint)

    async def get_position(self, symbol: str, account_id: str | None = None) -> dict[str, Any]:
        """Get position for symbol."""
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/positions/{symbol}"
        else:
            endpoint = f"/v2/positions/{symbol}"
        return await self._request("GET", endpoint)

    async def close_position(self, symbol: str, account_id: str | None = None) -> dict[str, Any]:
        """Close position for symbol."""
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/positions/{symbol}"
        else:
            endpoint = f"/v2/positions/{symbol}"
        return await self._request("DELETE", endpoint)

    # ========================================================================
    # ACCOUNT
    # ========================================================================

    async def get_account(self, account_id: str | None = None) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Account data (cash, buying power, equity, etc.)
        """
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/account"
        else:
            endpoint = "/v2/account"
        return await self._request("GET", endpoint)

    async def get_trading_accounts(self) -> list[dict[str, Any]]:
        """
        Get all trading accounts (Broker API only).
        
        Returns:
            List of trading accounts under this broker
        """
        if not self.use_broker_api:
            raise ValueError("get_trading_accounts is only available with Broker API")
        endpoint = "/v1/accounts"
        return await self._request("GET", endpoint)

    async def get_account_activities(
        self, activity_types: list[str] | None = None, date: datetime | None = None, account_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get account activities (trades, transactions, etc.)."""
        if self.use_broker_api:
            trading_account = account_id or settings.alpaca_sweep_account_id
            endpoint = f"/v1/trading/accounts/{trading_account}/account/activities"
        else:
            endpoint = "/v2/account/activities"
        params = {}

        if activity_types:
            params["activity_types"] = ",".join(activity_types)
        if date:
            params["date"] = date.date().isoformat()

        return await self._request("GET", endpoint, params=params)


# ============================================================================
# DATA INGESTION
# ============================================================================


async def ingest_historical_data(
    symbols: list[str], days: int = 30, timeframe: TimeFrame = TimeFrame.MIN_1
):
    """
    Ingest historical data from Alpaca to QuestDB.

    Args:
        symbols: List of symbols to ingest
        days: Number of days of history
        timeframe: Bar timeframe
    """
    client = AlpacaClient()
    await client.initialize()

    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Ingesting {days} days of data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                # Get bars from Alpaca
                response = await client.get_bars(
                    symbols=[symbol], timeframe=timeframe, start=start_date, end=end_date
                )

                bars = response.get("bars", {}).get(symbol, [])

                if not bars:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Insert to QuestDB
                await _insert_bars_to_questdb(symbol, bars)

                logger.info(f"Ingested {len(bars)} bars for {symbol}")

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error ingesting {symbol}: {e}")
                continue

    finally:
        await client.close()


async def _insert_bars_to_questdb(symbol: str, bars: list[dict]):
    """Insert bars to QuestDB using InfluxDB line protocol."""
    if not bars:
        return

    # Build InfluxDB line protocol messages
    lines = []
    for bar in bars:
        timestamp = int(
            datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp() * 1_000_000_000
        )

        line = (
            f"ticks,symbol={symbol} "
            f"price={bar['c']},volume={bar['v']},"
            f"open={bar['o']},high={bar['h']},low={bar['l']},close={bar['c']} "
            f"{timestamp}"
        )
        lines.append(line)

    # Send to QuestDB via InfluxDB line protocol (port 9009)
    # TODO: Implement InfluxDB line protocol sender
    logger.debug(f"Would insert {len(lines)} bars for {symbol}")


# ============================================================================
# REAL-TIME STREAMING
# ============================================================================


class AlpacaStreamer:
    """
    WebSocket stream for real-time market data.

    TODO: Implement WebSocket streaming for Phase 2.
    """

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.ws = None

    async def connect(self):
        """Connect to Alpaca WebSocket."""
        # TODO: Implement WebSocket connection
        pass

    async def subscribe(self, symbols: list[str]):
        """Subscribe to symbols."""
        # TODO: Implement subscription
        pass

    async def on_trade(self, callback):
        """Register trade callback."""
        # TODO: Implement callback
        pass


# ============================================================================
# GLOBAL CLIENT INSTANCE
# ============================================================================

alpaca_client = AlpacaClient()


# Export public API
__all__ = [
    "AlpacaClient",
    "AlpacaStreamer",
    "TimeFrame",
    "alpaca_client",
    "ingest_historical_data",
]
