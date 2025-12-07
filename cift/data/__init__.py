"""
CIFT Markets - Data Ingestion Module

Real-time market data connectors for order book processing.

Connectors:
- Polygon.io: L2 quotes, trades, WebSocket streams
- Databento: Institutional-grade L3 order book data
- Order Book Processor: Feature extraction from raw market data
"""

from cift.data.polygon_l2_connector import PolygonL2Connector
from cift.data.databento_connector import DatabentoConnector
from cift.data.order_book_processor import OrderBookProcessor, OrderBookSnapshot
from cift.data.data_aggregator import MarketDataAggregator

__all__ = [
    "PolygonL2Connector",
    "DatabentoConnector",
    "OrderBookProcessor",
    "OrderBookSnapshot",
    "MarketDataAggregator",
]
