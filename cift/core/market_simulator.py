"""
Market Data Simulator

Generates realistic market data updates for WebSocket testing.
This will be replaced by real market data feeds in production.

Features:
- Realistic price movements (Geometric Brownian Motion)
- Configurable volatility and trend
- Multiple symbol support
- Configurable update frequency
"""

import asyncio
import math
import random
from datetime import UTC, datetime

from loguru import logger


class MarketDataSimulator:
    """
    Simulate realistic market data for multiple symbols.

    Uses Geometric Brownian Motion for price evolution:
    dS = μ*S*dt + σ*S*dW

    Where:
    - μ (mu): drift/trend component
    - σ (sigma): volatility
    - dW: Wiener process (random walk)
    """

    def __init__(self):
        self.symbols: dict[str, dict] = {}
        self.running = False
        self.update_interval = 1.0  # seconds

    def add_symbol(
        self,
        symbol: str,
        initial_price: float,
        volatility: float = 0.02,
        trend: float = 0.0001,
    ):
        """
        Add a symbol to simulate.

        Args:
            symbol: Symbol ticker (e.g., "AAPL")
            initial_price: Starting price
            volatility: Annual volatility (0.02 = 2%)
            trend: Drift component (0.0001 = 0.01% per update)
        """
        self.symbols[symbol] = {
            "price": initial_price,
            "volatility": volatility,
            "trend": trend,
            "bid": initial_price - 0.01,
            "ask": initial_price + 0.01,
            "volume": 0,
        }
        logger.info(f"Added symbol {symbol} at ${initial_price:.2f}")

    def update_price(self, symbol: str) -> dict:
        """
        Generate next price tick using Geometric Brownian Motion.

        Returns:
            Dictionary with updated price data
        """
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not registered")

        data = self.symbols[symbol]
        current_price = data["price"]
        volatility = data["volatility"]
        trend = data["trend"]

        # Geometric Brownian Motion
        dt = self.update_interval / (252 * 6.5 * 3600)  # Convert to trading time fraction
        random_shock = random.gauss(0, 1)
        drift = trend * dt
        diffusion = volatility * math.sqrt(dt) * random_shock

        # Update price
        new_price = current_price * (1 + drift + diffusion)

        # Ensure price stays positive
        new_price = max(new_price, 0.01)

        # Update bid/ask with spread
        spread = new_price * 0.0001  # 0.01% spread
        bid = new_price - spread / 2
        ask = new_price + spread / 2

        # Generate random volume
        volume = random.randint(100, 5000)

        # Update stored data
        data["price"] = new_price
        data["bid"] = bid
        data["ask"] = ask
        data["volume"] = volume

        return {
            "symbol": symbol,
            "price": new_price,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def generate_updates(self, callback):
        """
        Continuously generate price updates for all symbols.

        Args:
            callback: Async function to call with each update
        """
        self.running = True
        logger.info(f"Market simulator started for {len(self.symbols)} symbols")

        try:
            while self.running:
                # Update each symbol
                for symbol in self.symbols.keys():
                    try:
                        tick_data = self.update_price(symbol)
                        await callback(tick_data)
                    except Exception as e:
                        logger.error(f"Error updating {symbol}: {e}")

                # Wait before next update
                await asyncio.sleep(self.update_interval)

        except asyncio.CancelledError:
            logger.info("Market simulator cancelled")
            self.running = False

        except Exception as e:
            logger.error(f"Market simulator error: {e}", exc_info=True)
            self.running = False

    def stop(self):
        """Stop the simulator."""
        self.running = False
        logger.info("Market simulator stopped")


# Global simulator instance
simulator = MarketDataSimulator()

# Add default symbols
simulator.add_symbol("AAPL", 170.0, volatility=0.015, trend=0.0001)
simulator.add_symbol("MSFT", 350.0, volatility=0.018, trend=0.00015)
simulator.add_symbol("GOOGL", 140.0, volatility=0.020, trend=0.00005)
simulator.add_symbol("AMZN", 155.0, volatility=0.022, trend=0.0002)
simulator.add_symbol("TSLA", 245.0, volatility=0.035, trend=-0.0001)
simulator.add_symbol("META", 485.0, volatility=0.025, trend=0.00012)
simulator.add_symbol("NVDA", 485.0, volatility=0.030, trend=0.0003)
simulator.add_symbol("AMD", 140.0, volatility=0.028, trend=0.00008)
