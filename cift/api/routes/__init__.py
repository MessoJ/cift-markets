"""
CIFT Markets - API Routes

All API route modules.
"""

from cift.api.routes import (
    analytics,
    auth,
    drilldowns,
    inference,
    market_data,
    trading,
    transactions,
    watchlists,
)

__all__ = [
    "auth",
    "market_data",
    "trading",
    "analytics",
    "drilldowns",
    "watchlists",
    "transactions",
    "inference",
]
