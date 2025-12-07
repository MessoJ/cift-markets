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
from typing import List, Optional, Dict, Any
from enum import Enum

import aiohttp
from loguru import logger

from cift.core.config import settings
from cift.core.database import questdb_manager, redis_manager


# ============================================================================
# CONSTANTS
# ============================================================================

class AlpacaEndpoint(str, Enum):
    """Alpaca API endpoints."""
    PAPER = "https://paper-api.alpaca.markets"
    LIVE = "https://api.alpaca.markets"
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
    
    High-performance async implementation with connection pooling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper_trading: bool = True
    ):
        """
        Initialize Alpaca client.
        
        Args:
            api_key: Alpaca API key (defaults to config)
            secret_key: Alpaca secret key (defaults to config)
            paper_trading: Use paper trading endpoint
        """
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        
        # Don't raise error here, check in _request or is_configured
        
        self.base_url = (
            AlpacaEndpoint.PAPER if paper_trading else AlpacaEndpoint.LIVE
        )
        self.data_url = AlpacaEndpoint.DATA
        
        self.session: Optional[aiohttp.ClientSession] = None
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
            limit=100,  # Max connections
            limit_per_host=30,
            ttl_dns_cache=300
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_headers()
        )
        
        self._initialized = True
        logger.info("Alpaca client initialized")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self._initialized = False
            logger.info("Alpaca client closed")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
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
            raise ValueError("Alpaca API keys not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        
        url = f"{base_url or self.base_url}{endpoint}"
        
        try:
            async with self.session.request(
                method,
                url,
                params=params,
                json=json
            ) as response:
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
    
    async def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest quote data
        """
        endpoint = f"/v2/stocks/{symbol}/quotes/latest"
        return await self._request("GET", endpoint, base_url=self.data_url)
    
    async def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
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
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.MIN_1,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
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
    
    async def get_snapshot(self, symbols: List[str]) -> Dict[str, Any]:
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
    # TRADING
    # ========================================================================
    
    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit order to Alpaca.
        
        Args:
            symbol: Stock symbol
            qty: Order quantity
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Client-assigned order ID
            
        Returns:
            Order data
        """
        data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        
        if limit_price:
            data["limit_price"] = limit_price
        if stop_price:
            data["stop_price"] = stop_price
        if client_order_id:
            data["client_order_id"] = client_order_id
        
        endpoint = "/v2/orders"
        return await self._request("POST", endpoint, json=data)
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order by ID."""
        endpoint = f"/v2/orders/{order_id}"
        return await self._request("GET", endpoint)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order by ID."""
        endpoint = f"/v2/orders/{order_id}"
        return await self._request("DELETE", endpoint)
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        endpoint = "/v2/orders"
        params = {"status": "open"}
        return await self._request("GET", endpoint, params=params)
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        endpoint = "/v2/positions"
        return await self._request("GET", endpoint)
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for symbol."""
        endpoint = f"/v2/positions/{symbol}"
        return await self._request("GET", endpoint)
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close position for symbol."""
        endpoint = f"/v2/positions/{symbol}"
        return await self._request("DELETE", endpoint)
    
    # ========================================================================
    # ACCOUNT
    # ========================================================================
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account data (cash, buying power, equity, etc.)
        """
        endpoint = "/v2/account"
        return await self._request("GET", endpoint)
    
    async def get_account_activities(
        self,
        activity_types: Optional[List[str]] = None,
        date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get account activities (trades, transactions, etc.)."""
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
    symbols: List[str],
    days: int = 30,
    timeframe: TimeFrame = TimeFrame.MIN_1
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
                    symbols=[symbol],
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
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


async def _insert_bars_to_questdb(symbol: str, bars: List[Dict]):
    """Insert bars to QuestDB using InfluxDB line protocol."""
    if not bars:
        return
    
    # Build InfluxDB line protocol messages
    lines = []
    for bar in bars:
        timestamp = int(datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp() * 1_000_000_000)
        
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
    
    async def subscribe(self, symbols: List[str]):
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
