"""
CIFT Markets - ML Execution Strategy

Production strategy that integrates all advanced ML techniques:
- RL-based action selection
- Toxicity-aware execution
- Market impact optimization
- Queue position management
- Latency arbitrage protection

This strategy replaces simple TWAP/VWAP with intelligent execution.
"""

import asyncio
import time
from datetime import datetime, UTC
from typing import Optional, Dict, Any
from uuid import UUID

from loguru import logger
import numpy as np

from cift.core.execution_strategies import ExecutionStrategy
from cift.ml.advanced_execution import (
    SmartExecutionSystem,
    ExecutionAction,
    ExecutionState,
    RLExecutionConfig,
)


class MLExecutionStrategy(ExecutionStrategy):
    """
    ML-Driven Execution Strategy
    
    Uses reinforcement learning, toxicity detection, and impact prediction
    to optimize order execution in real-time.
    
    Features:
    - Adaptive slicing based on market conditions
    - Toxicity-aware execution (avoid adverse selection)
    - Impact-minimizing trajectory
    - Queue position optimization
    - Anti-gaming protections
    
    Usage:
        strategy = MLExecutionStrategy(
            engine=execution_engine,
            order_data=order_dict,
            urgency=0.5,  # 0=patient, 1=urgent
            duration_minutes=15
        )
        await strategy.execute()
    """
    
    def __init__(
        self,
        engine,
        order_data: Dict[str, Any],
        urgency: float = 0.5,
        duration_minutes: int = 15,
        max_participation_rate: float = 0.1,
        min_slice_size: int = 100,
    ):
        super().__init__(engine, order_data)
        
        self.urgency = urgency
        self.duration_minutes = duration_minutes
        self.max_participation_rate = max_participation_rate
        self.min_slice_size = min_slice_size
        
        # Initialize ML system
        config = RLExecutionConfig(
            alpha_slippage=1.0,
            beta_timing=urgency * 0.2,  # More urgency = more timing cost sensitivity
            gamma_impact=1.0 - urgency * 0.5  # Less urgency = more impact sensitivity
        )
        self.ml_system = SmartExecutionSystem(config)
        
        # Execution tracking
        self.decision_price = float(order_data.get("price")) if order_data.get("price") else None
        self.start_time: Optional[float] = None
        self.slices_executed = 0
        self.total_filled = 0.0
        self.vwap_numerator = 0.0
        self.execution_costs_bps = []
        
        # State for RL
        self.last_state: Optional[ExecutionState] = None
        self.last_action: Optional[ExecutionAction] = None
        
    async def execute(self):
        """
        Main execution loop.
        
        Runs until order is filled or time expires.
        """
        logger.info(
            f"Starting ML Execution for {self.quantity} {self.symbol} | "
            f"Urgency: {self.urgency:.2f} | Duration: {self.duration_minutes}m"
        )
        
        self.start_time = time.time()
        end_time = self.start_time + self.duration_minutes * 60
        
        # Get initial market state
        market_data = await self._get_market_data()
        if not market_data:
            logger.error(f"Cannot get market data for {self.symbol}, aborting")
            return
        
        # Set decision price if not specified
        if self.decision_price is None:
            self.decision_price = market_data['mid']
        
        # Main execution loop
        slice_interval = max(1.0, 60 / (self.urgency * 10 + 1))  # 1-60 seconds
        
        while self.remaining_qty > 0 and self.is_active:
            current_time = time.time()
            
            # Check time remaining
            if current_time >= end_time:
                logger.warning(f"Time expired, {self.remaining_qty} remaining unfilled")
                # Final aggressive sweep
                if self.remaining_qty > 0:
                    await self._execute_sweep()
                break
            
            # Get fresh market data
            market_data = await self._get_market_data()
            if not market_data:
                await asyncio.sleep(1)
                continue
            
            # Calculate time remaining fraction
            time_elapsed = current_time - self.start_time
            total_duration = self.duration_minutes * 60
            time_remaining = max(0, (end_time - current_time) / total_duration)
            
            # Get ML execution decision
            decision = self.ml_system.get_execution_decision(
                symbol=self.symbol,
                side=self.side,
                total_quantity=self.quantity,
                remaining_quantity=self.remaining_qty,
                time_remaining=time_remaining,
                current_price=market_data['mid'],
                bid=market_data['bid'],
                ask=market_data['ask'],
                daily_volume=market_data.get('daily_volume', 1_000_000),
                volatility=market_data.get('volatility', 0.02),
                imbalance=market_data.get('imbalance', 0.0),
                hour_of_day=datetime.now(UTC).hour
            )
            
            logger.info(
                f"ML Decision: {decision['action']} | "
                f"Toxicity: {decision['toxicity']['score']:.2f} | "
                f"Impact: {decision['impact']['total_bps']:.1f}bps | "
                f"Reason: {decision['reason']}"
            )
            
            # Execute based on action
            action = decision['action_enum']
            
            if action == ExecutionAction.WAIT:
                # Don't trade this period
                logger.debug("WAIT action - skipping this period")
                await asyncio.sleep(slice_interval)
                continue
            
            elif action == ExecutionAction.CROSS_SPREAD:
                # Aggressive market order
                slice_size = self._calculate_slice_size(
                    decision['child_order_size'],
                    market_data.get('daily_volume', 1_000_000)
                )
                await self._execute_aggressive(slice_size, market_data)
            
            elif action == ExecutionAction.AGGRESSIVE_LIMIT:
                # Limit order one tick inside spread
                slice_size = self._calculate_slice_size(
                    decision['child_order_size'],
                    market_data.get('daily_volume', 1_000_000)
                )
                await self._execute_aggressive_limit(slice_size, decision['suggested_price'], market_data)
            
            elif action == ExecutionAction.MID_LIMIT:
                # Limit order at mid
                slice_size = self._calculate_slice_size(
                    decision['child_order_size'],
                    market_data.get('daily_volume', 1_000_000)
                )
                await self._execute_mid_limit(slice_size, market_data)
            
            else:  # PASSIVE_LIMIT
                # Join the queue at best bid/ask
                slice_size = self._calculate_slice_size(
                    decision['child_order_size'],
                    market_data.get('daily_volume', 1_000_000)
                )
                await self._execute_passive(slice_size, market_data)
            
            self.slices_executed += 1
            
            # Update toxicity detector with trade
            if self.total_filled > 0:
                self.ml_system.toxicity_detector.update_trade(
                    price=market_data['mid'],
                    size=int(slice_size),
                    is_buy=(self.side == 'buy'),
                    prev_mid=market_data['mid']
                )
            
            # Dynamic interval based on conditions
            if decision['toxicity']['is_toxic']:
                # Slow down in toxic conditions
                await asyncio.sleep(slice_interval * 1.5)
            elif decision['latency']['is_front_run']:
                # Randomize timing to avoid predictability
                await asyncio.sleep(slice_interval * (0.5 + np.random.random()))
            else:
                await asyncio.sleep(slice_interval)
        
        # Log execution summary
        await self._log_summary()
    
    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data for the symbol."""
        try:
            # Import here to avoid circular imports
            from cift.services.market_data_service import MarketDataService
            
            market_data = MarketDataService()
            quote = await market_data.get_quote(self.symbol)
            
            if not quote:
                return None
            
            bid = float(quote.get('bid_price', quote.get('price', 0)))
            ask = float(quote.get('ask_price', quote.get('price', 0)))
            
            if bid <= 0 or ask <= 0:
                return None
            
            mid = (bid + ask) / 2
            
            # Get additional data if available
            bid_size = int(quote.get('bid_size', 100))
            ask_size = int(quote.get('ask_size', 100))
            
            # Calculate L1 imbalance
            total_size = bid_size + ask_size
            imbalance = (bid_size - ask_size) / total_size if total_size > 0 else 0
            
            return {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'imbalance': imbalance,
                'spread_bps': (ask - bid) / mid * 10000,
                'daily_volume': float(quote.get('volume', 1_000_000)),
                'volatility': float(quote.get('volatility', 0.02))
            }
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None
    
    def _calculate_slice_size(self, suggested_size: float, daily_volume: float) -> float:
        """Calculate appropriate slice size respecting constraints."""
        # Start with ML suggestion
        size = suggested_size
        
        # Don't exceed remaining quantity
        size = min(size, self.remaining_qty)
        
        # Respect max participation rate
        # Assuming we trade once per minute, volume per minute = daily_volume / 390
        minute_volume = daily_volume / 390
        max_from_participation = minute_volume * self.max_participation_rate
        size = min(size, max_from_participation)
        
        # Enforce minimum slice size
        size = max(size, self.min_slice_size)
        
        # Final cap at remaining
        size = min(size, self.remaining_qty)
        
        return size
    
    async def _execute_aggressive(self, size: float, market_data: Dict):
        """Execute aggressive market order."""
        logger.info(f"AGGRESSIVE: Market order for {size} shares")
        
        await self.submit_child_order(
            qty=size,
            price=None,
            order_type="market",
            time_in_force="ioc"
        )
        
        # Record for tracking (actual fill comes from engine callback)
        expected_price = market_data['ask'] if self.side == 'buy' else market_data['bid']
        self._record_execution(size, expected_price)
    
    async def _execute_aggressive_limit(self, size: float, price: Optional[float], market_data: Dict):
        """Execute aggressive limit order (one tick inside)."""
        if price is None:
            tick = 0.01
            if self.side == 'buy':
                price = market_data['bid'] + tick
            else:
                price = market_data['ask'] - tick
        
        logger.info(f"AGGRESSIVE LIMIT: {size} shares @ {price}")
        
        await self.submit_child_order(
            qty=size,
            price=price,
            order_type="limit",
            time_in_force="ioc"  # IOC to avoid resting
        )
        
        self._record_execution(size, price)
    
    async def _execute_mid_limit(self, size: float, market_data: Dict):
        """Execute limit order at mid price."""
        mid = market_data['mid']
        
        logger.info(f"MID LIMIT: {size} shares @ {mid}")
        
        await self.submit_child_order(
            qty=size,
            price=mid,
            order_type="limit",
            time_in_force="ioc"
        )
        
        self._record_execution(size, mid)
    
    async def _execute_passive(self, size: float, market_data: Dict):
        """Execute passive limit order at BBO."""
        if self.side == 'buy':
            price = market_data['bid']
        else:
            price = market_data['ask']
        
        logger.info(f"PASSIVE: {size} shares @ {price}")
        
        # For passive orders, use longer TIF to rest
        await self.submit_child_order(
            qty=size,
            price=price,
            order_type="limit",
            time_in_force="day"  # Rest in queue
        )
        
        # For passive, we might not fill - don't count until confirmed
        # In production, this comes from fill callbacks
        self._record_execution(size * 0.3, price)  # Assume 30% fill rate for passive
    
    async def _execute_sweep(self):
        """Final aggressive sweep for remaining quantity."""
        logger.warning(f"Executing final sweep for {self.remaining_qty} shares")
        
        await self.submit_child_order(
            qty=self.remaining_qty,
            price=None,
            order_type="market",
            time_in_force="ioc"
        )
        
        self.remaining_qty = 0
    
    def _record_execution(self, size: float, price: float):
        """Record execution for VWAP tracking."""
        self.total_filled += size
        self.vwap_numerator += size * price
        self.remaining_qty -= size
        
        # Track execution cost vs decision price
        if self.decision_price and self.decision_price > 0:
            slippage = (price - self.decision_price) / self.decision_price * 10000
            if self.side == 'sell':
                slippage = -slippage
            self.execution_costs_bps.append(slippage)
    
    async def _log_summary(self):
        """Log execution summary."""
        if self.total_filled <= 0:
            logger.warning(f"No fills for order {self.order_id}")
            return
        
        vwap = self.vwap_numerator / self.total_filled
        
        # Implementation shortfall vs decision price
        if self.decision_price and self.decision_price > 0:
            is_bps = (vwap - self.decision_price) / self.decision_price * 10000
            if self.side == 'sell':
                is_bps = -is_bps
        else:
            is_bps = 0
        
        avg_cost = np.mean(self.execution_costs_bps) if self.execution_costs_bps else 0
        
        logger.info(
            f"ML Execution Complete for {self.order_id}:\n"
            f"  Filled: {self.total_filled:.0f} / {self.quantity:.0f} shares\n"
            f"  VWAP: ${vwap:.4f}\n"
            f"  Decision Price: ${self.decision_price:.4f}\n"
            f"  Implementation Shortfall: {is_bps:.2f} bps\n"
            f"  Slices: {self.slices_executed}\n"
            f"  Avg Cost per Slice: {avg_cost:.2f} bps"
        )
        
        # Record for ML training
        self.ml_system.impact_predictor.record_execution(
            symbol=self.symbol,
            side=self.side,
            quantity=self.total_filled,
            expected_price=self.decision_price,
            actual_vwap=vwap,
            features={'predicted_impact': avg_cost}
        )


class AdaptivePoVStrategy(ExecutionStrategy):
    """
    Adaptive Percentage of Volume (PoV) Strategy
    
    Executes as a fixed percentage of market volume,
    adapting to real-time flow conditions.
    
    Features:
    - Dynamic participation rate based on toxicity
    - Spread-aware aggression
    - Volume burst detection
    """
    
    def __init__(
        self,
        engine,
        order_data: Dict[str, Any],
        target_pov: float = 0.1,  # 10% of volume
        min_pov: float = 0.02,
        max_pov: float = 0.25,
    ):
        super().__init__(engine, order_data)
        
        self.target_pov = target_pov
        self.min_pov = min_pov
        self.max_pov = max_pov
        
        # Volume tracking
        self.volume_window: list = []
        self.last_volume_check = 0
        
    async def execute(self):
        """Execute PoV strategy."""
        logger.info(f"Starting Adaptive PoV for {self.quantity} {self.symbol} @ {self.target_pov*100:.1f}%")
        
        while self.remaining_qty > 0 and self.is_active:
            # Get current market data
            market_data = await self._get_market_data()
            if not market_data:
                await asyncio.sleep(1)
                continue
            
            # Calculate current volume rate
            current_volume = market_data.get('volume', 0)
            volume_rate = self._calculate_volume_rate(current_volume)
            
            # Adjust participation based on conditions
            adjusted_pov = self._adjust_participation(market_data)
            
            # Calculate slice size
            # Volume in last minute * our target participation
            slice_size = max(100, volume_rate * adjusted_pov)
            slice_size = min(slice_size, self.remaining_qty)
            
            # Determine aggression
            spread_bps = market_data.get('spread_bps', 10)
            if spread_bps < 5:
                order_type = "limit"
                price = market_data['bid'] if self.side == 'buy' else market_data['ask']
            else:
                order_type = "market"
                price = None
            
            await self.submit_child_order(
                qty=slice_size,
                price=price,
                order_type=order_type,
                time_in_force="ioc"
            )
            
            self.remaining_qty -= slice_size
            
            # Wait proportional to volume rate
            wait_time = max(1, 60 * slice_size / (volume_rate + 1))
            await asyncio.sleep(min(wait_time, 10))
        
        logger.info(f"Adaptive PoV Complete for {self.order_id}")
    
    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get market data (same as MLExecutionStrategy)."""
        try:
            from cift.services.market_data_service import MarketDataService
            market_data = MarketDataService()
            quote = await market_data.get_quote(self.symbol)
            
            if not quote:
                return None
            
            return {
                'bid': float(quote.get('bid_price', 0)),
                'ask': float(quote.get('ask_price', 0)),
                'mid': (float(quote.get('bid_price', 0)) + float(quote.get('ask_price', 0))) / 2,
                'volume': int(quote.get('volume', 0)),
                'spread_bps': 10  # Default
            }
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None
    
    def _calculate_volume_rate(self, current_volume: int) -> float:
        """Calculate volume rate per minute."""
        now = time.time()
        self.volume_window.append((now, current_volume))
        
        # Keep last 5 minutes
        cutoff = now - 300
        self.volume_window = [(t, v) for t, v in self.volume_window if t >= cutoff]
        
        if len(self.volume_window) < 2:
            return 1000  # Default rate
        
        time_span = self.volume_window[-1][0] - self.volume_window[0][0]
        volume_span = self.volume_window[-1][1] - self.volume_window[0][1]
        
        if time_span <= 0:
            return 1000
        
        return volume_span / time_span * 60  # Per minute
    
    def _adjust_participation(self, market_data: Dict) -> float:
        """Adjust participation rate based on conditions."""
        pov = self.target_pov
        
        # Reduce participation if spread is wide
        spread_bps = market_data.get('spread_bps', 10)
        if spread_bps > 20:
            pov *= 0.7
        elif spread_bps > 10:
            pov *= 0.85
        
        # Clamp to bounds
        return max(self.min_pov, min(self.max_pov, pov))


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

STRATEGY_REGISTRY = {
    'ml_smart': MLExecutionStrategy,
    'adaptive_pov': AdaptivePoVStrategy,
}


def create_ml_strategy(
    strategy_type: str,
    engine,
    order_data: Dict[str, Any],
    **kwargs
) -> ExecutionStrategy:
    """
    Factory function to create ML execution strategies.
    
    Args:
        strategy_type: 'ml_smart' or 'adaptive_pov'
        engine: Execution engine
        order_data: Order details
        **kwargs: Strategy-specific parameters
    
    Returns:
        ExecutionStrategy instance
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_type)
    
    if not strategy_class:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return strategy_class(engine, order_data, **kwargs)


__all__ = [
    'MLExecutionStrategy',
    'AdaptivePoVStrategy',
    'create_ml_strategy',
    'STRATEGY_REGISTRY',
]
