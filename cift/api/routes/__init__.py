"""
CIFT Markets - API Routes

All API route modules.
"""

from cift.api.routes import (
    auth, market_data, trading, analytics,
    drilldowns, watchlists, transactions, inference
)

__all__ = [
    "auth", "market_data", "trading", "analytics",
    "drilldowns", "watchlists", "transactions", "inference"
]
