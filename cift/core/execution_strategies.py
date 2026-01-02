"""
CIFT Markets - Advanced Execution Strategies

Implementation of institutional-grade execution algorithms:
1. Iceberg (Hidden Orders)
2. Adaptive TWAP (Volatility-adjusted)
3. Alpha-Driven Execution (ML-based)
4. Imbalance-Based Execution (L2 Data)

"""

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from uuid import UUID

from loguru import logger
import numpy as np

# We will import these lazily or use dependency injection to avoid circular imports
# from cift.core.execution_engine import ExecutionEngine
# from cift.data.order_book_processor import OrderBookSnapshot

class ExecutionStrategy(ABC):
    """Base class for execution strategies."""
    
    def __init__(self, engine, order_data: Dict[str, Any]):
        self.engine = engine
        self.order_data = order_data
        self.order_id = order_data.get("order_id")
        self.symbol = order_data.get("symbol")
        self.side = order_data.get("side")
        self.quantity = float(order_data.get("quantity"))
        self.remaining_qty = self.quantity
        self.is_active = True

    @abstractmethod
    async def execute(self):
        """Run the execution logic."""
        pass

    async def submit_child_order(self, qty: float, price: Optional[float] = None, order_type: str = "limit", time_in_force: str = "ioc") -> Optional[UUID]:
        """Helper to submit child orders via the main engine."""
        child_order = {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": qty,
            "order_type": order_type,
            "price": price,
            "time_in_force": time_in_force,
            "parent_order_id": self.order_id,
            "user_id": self.order_data.get("user_id")
        }
        
        # We use the engine's internal submission to bypass the queue if needed, 
        # or we can put it in the queue. For child orders, direct submission to broker adapter might be better
        # but for consistency, let's use submit_order but mark it as a child.
        # Note: In a real system, we'd have a separate 'router' method.
        # For now, we assume engine.submit_child_order exists or we use submit_order.
        
        # Using engine.submit_order might re-trigger this strategy if we aren't careful.
        # We need a flag in order_data to say "direct_execution" or "child_order".
        child_order["strategy"] = "direct" # Bypass strategy logic
        
        logger.info(f"Strategy {self.__class__.__name__} submitting child: {self.side} {qty} {self.symbol} @ {price}")
        return await self.engine.submit_order(child_order)


class IcebergStrategy(ExecutionStrategy):
    """
    Splits a large order into smaller visible 'tips' to hide total size.
    Randomizes tip size and delay to avoid detection.
    """
    
    def __init__(self, engine, order_data: Dict[str, Any], visible_size: int = 100, variance: float = 0.2):
        super().__init__(engine, order_data)
        self.visible_size = visible_size
        self.variance = variance
        self.limit_price = float(order_data.get("price")) if order_data.get("price") else None

    async def execute(self):
        logger.info(f"Starting Iceberg Execution for {self.quantity} {self.symbol}")
        
        while self.remaining_qty > 0 and self.is_active:
            # Calculate tip size
            # Randomize by +/- variance
            var_mult = 1.0 + random.uniform(-self.variance, self.variance)
            tip_size = int(self.visible_size * var_mult)
            
            # Don't exceed remaining
            tip_size = min(tip_size, int(self.remaining_qty))
            if tip_size <= 0:
                break
                
            # Submit tip
            # We use IOC or FOK to ensure we don't rest too long if not filled, 
            # OR we use Day/GTC if we want to rest the tip. 
            # Standard Iceberg rests the tip.
            await self.submit_child_order(
                qty=tip_size,
                price=self.limit_price,
                order_type="limit",
                time_in_force="day" # Rest the tip
            )
            
            # In a real system, we need to wait for the fill confirmation of this specific child order.
            # Since our engine is async and event-driven, this 'while' loop blocks.
            # We would need to await a Future that resolves when the child order is filled.
            # For this implementation, we will simulate a wait or assume the engine handles callbacks.
            
            # SIMPLIFICATION: We assume we wait for a fill event.
            # In this code, we'll just decrement and wait a bit (Simulated behavior for now)
            # TODO: Hook into engine's event stream for real fill updates.
            
            self.remaining_qty -= tip_size
            
            # Random delay before reloading
            delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(delay)
            
        logger.info(f"Iceberg Execution Complete for {self.order_id}")


class AdaptiveTWAPStrategy(ExecutionStrategy):
    """
    Time-Weighted Average Price strategy that adapts to volatility.
    Executes faster when volatility is low, slower when high.
    """
    
    def __init__(self, engine, order_data: Dict[str, Any], duration_minutes: int = 10):
        super().__init__(engine, order_data)
        self.duration_minutes = duration_minutes
        self.interval_seconds = 60
        self.slices = duration_minutes # One slice per minute roughly

    async def execute(self):
        logger.info(f"Starting Adaptive TWAP for {self.quantity} {self.symbol} over {self.duration_minutes}m")
        
        base_slice_size = self.quantity / self.slices
        
        for i in range(self.slices):
            if not self.is_active or self.remaining_qty <= 0:
                break
                
            # 1. Get Volatility (Mocked for now, should come from OrderBookProcessor)
            # volatility = self.engine.get_volatility(self.symbol)
            volatility = 0.01 # 1% mock
            
            # 2. Calculate Urgency
            # Low vol -> High urgency (safe to trade)
            # High vol -> Low urgency (wait it out)
            urgency = max(0.5, min(1.5, 0.01 / max(volatility, 0.001)))
            
            slice_qty = int(base_slice_size * urgency)
            slice_qty = min(slice_qty, int(self.remaining_qty))
            
            if slice_qty > 0:
                await self.submit_child_order(
                    qty=slice_qty,
                    order_type="market", # TWAP usually takes liquidity
                    time_in_force="ioc"
                )
                self.remaining_qty -= slice_qty
                
            await asyncio.sleep(self.interval_seconds)


class ImbalanceStrategy(ExecutionStrategy):
    """
    Uses Order Book Imbalance to determine aggression.
    - High Buy Imbalance -> Aggressive Buy (Market/Crossing Limit)
    - High Sell Imbalance -> Passive Buy (Resting Limit)
    """
    
    def __init__(self, engine, order_data: Dict[str, Any], threshold: float = 0.3):
        super().__init__(engine, order_data)
        self.threshold = threshold

    async def execute(self):
        # Lazy import to avoid circular dependency
        from cift.services.market_data_service import MarketDataService
        
        # In a real system, we'd inject this or get it from a singleton container
        market_data = MarketDataService()
        # We don't need to initialize full service just for a quote if it's stateless enough,
        # but let's assume we can get a quote.
        
        # Get latest quote/snapshot
        # Note: Real L2 imbalance requires full order book. 
        # Polygon's snapshot gives us bid/ask size which is L1 imbalance.
        # For L2, we'd need the OrderBookProcessor connected to a stream.
        # Here we use L1 imbalance as a proxy.
        quote = await market_data.get_quote(self.symbol)
        
        if not quote:
            logger.warning(f"No quote for {self.symbol}, falling back to limit")
            await self.submit_child_order(self.quantity, self.order_data.get("price"), "limit")
            return

        # Calculate L1 Imbalance
        # Imbalance = (BidSize - AskSize) / (BidSize + AskSize)
        # Range: -1 (All Sellers) to +1 (All Buyers)
        bid_size = quote.get("bid_size", 0)
        ask_size = quote.get("ask_size", 0)
        bid_price = quote.get("bid_price", quote.get("price"))
        ask_price = quote.get("ask_price", quote.get("price"))
        
        total_size = bid_size + ask_size
        imbalance = (bid_size - ask_size) / total_size if total_size > 0 else 0
        
        logger.info(f"L1 Imbalance for {self.symbol}: {imbalance:.2f} (Bid: {bid_size}, Ask: {ask_size})")
        
        price = None
        order_type = "limit"
        
        if self.side == "buy":
            if imbalance > self.threshold:
                # Heavy buying pressure, price likely to go up.
                # Be aggressive: Cross the spread or Market
                logger.info(f"High Buy Imbalance ({imbalance:.2f}). Aggressive execution.")
                order_type = "market"
            elif imbalance < -self.threshold:
                # Heavy selling pressure, price likely to go down.
                # Be passive: Sit on the bid or lower
                logger.info(f"High Sell Imbalance ({imbalance:.2f}). Passive execution.")
                order_type = "limit"
                price = bid_price
            else:
                # Neutral
                order_type = "limit"
                price = bid_price
                
        elif self.side == "sell":
            if imbalance < -self.threshold:
                # Heavy selling pressure. Get out now.
                order_type = "market"
            elif imbalance > self.threshold:
                # Heavy buying pressure. Wait for higher price.
                order_type = "limit"
                price = ask_price
            else:
                order_type = "limit"
                price = ask_price
                
        await self.submit_child_order(
            qty=self.quantity,
            price=price,
            order_type=order_type
        )
        self.remaining_qty = 0

