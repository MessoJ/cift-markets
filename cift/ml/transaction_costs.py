"""
Realistic Transaction Cost Model

This module implements sophisticated transaction cost modeling:
1. Square-root market impact (Kyle's Lambda)
2. Temporary vs Permanent impact
3. Spread dynamics
4. Execution timing costs
5. Almgren-Chriss optimal execution framework

Reference:
- Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"
- Gatheral & Schied (2013), "Dynamical Models of Market Impact"
- Kyle (1985), "Continuous Auctions and Insider Trading"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# COST MODEL CONFIGURATION
# =============================================================================

@dataclass
class MarketConditions:
    """Current market conditions for cost estimation."""
    bid: float
    ask: float
    mid_price: float
    daily_volume: float      # Shares per day
    volatility: float        # Daily volatility (e.g., 0.02 = 2%)
    avg_trade_size: float    # Average trade size in shares
    spread_bps: Optional[float] = None  # Override spread
    
    def __post_init__(self):
        if self.spread_bps is None:
            self.spread_bps = (self.ask - self.bid) / self.mid_price * 10000


class ExecutionStyle(Enum):
    """Execution urgency style."""
    PASSIVE = "passive"      # Minimize impact, accept timing risk
    NEUTRAL = "neutral"      # Balance impact and timing
    AGGRESSIVE = "aggressive" # Minimize timing risk, accept impact


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost model."""
    # Fixed costs
    commission_per_share: float = 0.005    # $0.005 per share
    commission_min: float = 1.0            # Minimum commission
    commission_max: float = 20.0           # Maximum commission
    
    # SEC/FINRA fees (US equities)
    sec_fee_per_dollar: float = 0.0000278  # SEC fee
    taf_fee_per_share: float = 0.000119    # TAF fee
    
    # Impact model parameters (calibrated to typical liquid stocks)
    # Permanent impact: eta * sigma * (Q/V)^0.5
    permanent_impact_eta: float = 0.1
    
    # Temporary impact: gamma * sigma * (q/V)^0.6
    temporary_impact_gamma: float = 0.314
    temporary_impact_exponent: float = 0.6
    
    # Spread model
    spread_half_bps: float = 5.0           # Half-spread default
    
    # Timing risk (volatility cost)
    timing_risk_lambda: float = 1e-6       # Risk aversion parameter


# =============================================================================
# CORE COST CALCULATIONS
# =============================================================================

def calculate_spread_cost(
    shares: float,
    market: MarketConditions,
    is_buy: bool,
    aggressive: bool = True
) -> float:
    """
    Calculate spread crossing cost.
    
    Args:
        shares: Number of shares
        market: Market conditions
        is_buy: True for buy, False for sell
        aggressive: True to cross spread, False to provide liquidity
        
    Returns:
        Cost in dollars
    """
    if aggressive:
        # Cross the spread
        if is_buy:
            execution_price = market.ask
        else:
            execution_price = market.bid
    else:
        # Provide liquidity (may not fill)
        if is_buy:
            execution_price = market.bid
        else:
            execution_price = market.ask
    
    # Cost relative to mid
    price_diff = abs(execution_price - market.mid_price)
    cost = price_diff * abs(shares)
    
    return cost


def calculate_market_impact(
    shares: float,
    market: MarketConditions,
    config: TransactionCostConfig,
    execution_time_days: float = 1.0
) -> Tuple[float, float]:
    """
    Calculate market impact using Almgren-Chriss square-root model.
    
    Total impact = Permanent impact + Temporary impact
    
    Permanent impact affects all future trades (information revelation)
    Temporary impact is short-lived (liquidity consumption)
    
    Args:
        shares: Number of shares to trade
        market: Market conditions
        config: Cost configuration
        execution_time_days: Time to execute in days
        
    Returns:
        (permanent_impact_cost, temporary_impact_cost) in dollars
    """
    q = abs(shares)  # Trade quantity
    v = market.daily_volume  # Daily volume
    sigma = market.volatility  # Daily vol
    p = market.mid_price
    
    if v <= 0 or q <= 0:
        return 0.0, 0.0
    
    # Participation rate
    participation = q / (v * execution_time_days)
    
    # Permanent impact: eta * sigma * (Q/V)^0.5
    # This is price impact per share, multiply by shares for total cost
    permanent_impact_pct = config.permanent_impact_eta * sigma * math.sqrt(q / v)
    permanent_cost = permanent_impact_pct * p * q
    
    # Temporary impact: gamma * sigma * (q_t / V)^delta
    # For single-day execution, q_t = q / execution_time_days
    q_rate = q / max(execution_time_days, 1e-6)
    temporary_impact_pct = config.temporary_impact_gamma * sigma * (q_rate / v) ** config.temporary_impact_exponent
    temporary_cost = temporary_impact_pct * p * q
    
    return permanent_cost, temporary_cost


def calculate_fixed_costs(
    shares: float,
    price: float,
    config: TransactionCostConfig,
    is_sell: bool = False
) -> float:
    """
    Calculate fixed transaction costs (commissions, fees).
    
    Args:
        shares: Number of shares
        price: Execution price
        config: Cost configuration
        is_sell: True for sell (SEC fee applies)
        
    Returns:
        Total fixed costs in dollars
    """
    shares = abs(shares)
    notional = shares * price
    
    # Commission
    commission = shares * config.commission_per_share
    commission = max(config.commission_min, min(config.commission_max, commission))
    
    # Regulatory fees (US equities, sell-side only)
    fees = config.taf_fee_per_share * shares
    if is_sell:
        fees += config.sec_fee_per_dollar * notional
    
    return commission + fees


def estimate_total_cost(
    shares: float,
    market: MarketConditions,
    config: TransactionCostConfig,
    is_buy: bool = True,
    execution_time_days: float = 1.0,
    aggressive: bool = True
) -> dict:
    """
    Estimate total transaction cost.
    
    Args:
        shares: Number of shares
        market: Market conditions
        config: Cost configuration
        is_buy: True for buy order
        execution_time_days: Time to execute
        aggressive: True to cross spread
        
    Returns:
        Dict with cost breakdown
    """
    shares = abs(shares)
    notional = shares * market.mid_price
    
    # Spread cost
    spread_cost = calculate_spread_cost(shares, market, is_buy, aggressive)
    
    # Market impact
    permanent_cost, temporary_cost = calculate_market_impact(
        shares, market, config, execution_time_days
    )
    
    # Fixed costs
    fixed_cost = calculate_fixed_costs(
        shares, market.mid_price, config, is_sell=not is_buy
    )
    
    # Total
    total_cost = spread_cost + permanent_cost + temporary_cost + fixed_cost
    
    return {
        'shares': shares,
        'notional': notional,
        'spread_cost': spread_cost,
        'permanent_impact': permanent_cost,
        'temporary_impact': temporary_cost,
        'fixed_costs': fixed_cost,
        'total_cost': total_cost,
        'cost_bps': total_cost / notional * 10000 if notional > 0 else 0,
        'breakdown_pct': {
            'spread': spread_cost / total_cost * 100 if total_cost > 0 else 0,
            'permanent_impact': permanent_cost / total_cost * 100 if total_cost > 0 else 0,
            'temporary_impact': temporary_cost / total_cost * 100 if total_cost > 0 else 0,
            'fixed': fixed_cost / total_cost * 100 if total_cost > 0 else 0
        }
    }


# =============================================================================
# ALMGREN-CHRISS OPTIMAL EXECUTION
# =============================================================================

@dataclass
class AlmgrenChrissParams:
    """Parameters for Almgren-Chriss optimal execution."""
    # Risk aversion (higher = more aggressive execution)
    lambda_risk: float = 1e-6
    
    # Impact model (default: linear temporary, linear permanent)
    eta: float = 0.01           # Permanent impact coefficient
    gamma: float = 0.01         # Temporary impact coefficient
    
    # Market parameters
    sigma: float = 0.02         # Daily volatility
    
    @classmethod
    def from_market_conditions(
        cls,
        market: MarketConditions,
        risk_aversion: float = 1e-6
    ) -> 'AlmgrenChrissParams':
        """Estimate parameters from market conditions."""
        # Empirical estimates based on typical market microstructure
        daily_vol = market.daily_volume
        volatility = market.volatility
        
        # Eta and gamma scale with volatility and inversely with liquidity
        liquidity_factor = 1e6 / (daily_vol + 1e6)
        
        eta = 0.01 * liquidity_factor * volatility / 0.02
        gamma = 0.01 * liquidity_factor * volatility / 0.02
        
        return cls(
            lambda_risk=risk_aversion,
            eta=eta,
            gamma=gamma,
            sigma=volatility
        )


def optimal_execution_trajectory(
    total_shares: float,
    time_horizon: int,
    params: AlmgrenChrissParams
) -> np.ndarray:
    """
    Compute Almgren-Chriss optimal execution trajectory.
    
    The optimal strategy minimizes:
    E[Cost] + lambda * Var[Cost]
    
    For linear impact, the optimal trajectory is:
    x_t = X * sinh(kappa * (T - t)) / sinh(kappa * T)
    
    where kappa = sqrt(lambda * sigma^2 / eta)
    
    Args:
        total_shares: Total shares to execute
        time_horizon: Number of periods
        params: Model parameters
        
    Returns:
        Array of remaining shares at each time step
    """
    if time_horizon <= 0:
        return np.array([total_shares])
    
    X = abs(total_shares)
    T = time_horizon
    
    # Kappa determines execution speed
    # Higher lambda (risk aversion) -> faster execution
    if params.eta > 0 and params.lambda_risk > 0:
        kappa = math.sqrt(params.lambda_risk * params.sigma**2 / params.eta)
    else:
        kappa = 0.1  # Default moderate speed
    
    # Optimal trajectory
    t_grid = np.arange(T + 1)
    
    try:
        sinh_kappa_T = math.sinh(kappa * T)
        if abs(sinh_kappa_T) < 1e-10:
            # Near-linear trajectory
            trajectory = X * (T - t_grid) / T
        else:
            trajectory = X * np.sinh(kappa * (T - t_grid)) / sinh_kappa_T
    except (OverflowError, ValueError):
        # Fallback to linear
        trajectory = X * (T - t_grid) / T
    
    # Ensure sign matches original order
    if total_shares < 0:
        trajectory = -trajectory
    
    return trajectory


def optimal_execution_schedule(
    total_shares: float,
    time_horizon: int,
    params: AlmgrenChrissParams
) -> np.ndarray:
    """
    Compute optimal shares to trade at each time step.
    
    Args:
        total_shares: Total shares to execute
        time_horizon: Number of periods
        params: Model parameters
        
    Returns:
        Array of shares to trade at each step
    """
    trajectory = optimal_execution_trajectory(total_shares, time_horizon, params)
    
    # Schedule is negative of trajectory differences
    schedule = -np.diff(trajectory)
    
    return schedule


def expected_execution_cost(
    total_shares: float,
    time_horizon: int,
    params: AlmgrenChrissParams,
    price: float = 100.0
) -> dict:
    """
    Calculate expected cost of optimal execution.
    
    Args:
        total_shares: Total shares to execute
        time_horizon: Number of periods
        params: Model parameters
        price: Current price
        
    Returns:
        Dict with cost components
    """
    X = abs(total_shares)
    T = time_horizon
    
    if T <= 0 or X <= 0:
        return {
            'expected_cost': 0,
            'variance': 0,
            'permanent_cost': 0,
            'temporary_cost': 0,
            'volatility_cost': 0
        }
    
    # Get optimal trajectory
    trajectory = optimal_execution_trajectory(X, T, params)
    schedule = -np.diff(trajectory)
    
    # Permanent impact cost (one-time, depends on total quantity)
    permanent_cost = params.eta * X**2 * price
    
    # Temporary impact cost (depends on execution speed)
    temp_cost = 0
    for t, n_t in enumerate(schedule):
        temp_cost += params.gamma * n_t**2
    temporary_cost = temp_cost * price
    
    # Volatility cost (execution risk)
    vol_cost = 0
    for t in range(T):
        x_t = trajectory[t]
        vol_cost += x_t**2
    volatility_cost = params.sigma**2 * vol_cost * price**2 / T
    
    total_cost = permanent_cost + temporary_cost
    variance = volatility_cost
    
    return {
        'expected_cost': total_cost,
        'expected_cost_bps': total_cost / (X * price) * 10000,
        'variance': variance,
        'std_dev': math.sqrt(variance),
        'permanent_cost': permanent_cost,
        'temporary_cost': temporary_cost,
        'volatility_cost': volatility_cost,
        'schedule': schedule.tolist(),
        'trajectory': trajectory.tolist()
    }


# =============================================================================
# TWAP/VWAP EXECUTION
# =============================================================================

def twap_schedule(
    total_shares: float,
    n_intervals: int
) -> np.ndarray:
    """
    Time-Weighted Average Price (TWAP) execution schedule.
    
    Equal-sized trades at each interval.
    
    Args:
        total_shares: Total shares to execute
        n_intervals: Number of time intervals
        
    Returns:
        Array of shares per interval
    """
    if n_intervals <= 0:
        return np.array([total_shares])
    
    shares_per_interval = total_shares / n_intervals
    return np.full(n_intervals, shares_per_interval)


def vwap_schedule(
    total_shares: float,
    historical_volume_profile: np.ndarray
) -> np.ndarray:
    """
    Volume-Weighted Average Price (VWAP) execution schedule.
    
    Trade proportional to historical volume pattern.
    
    Args:
        total_shares: Total shares to execute
        historical_volume_profile: Volume at each interval (normalized)
        
    Returns:
        Array of shares per interval
    """
    # Normalize volume profile
    vol_profile = np.array(historical_volume_profile)
    vol_profile = vol_profile / vol_profile.sum()
    
    # Allocate shares proportionally
    schedule = total_shares * vol_profile
    
    return schedule


# =============================================================================
# COST-ADJUSTED BACKTEST
# =============================================================================

def apply_transaction_costs(
    returns: np.ndarray,
    positions: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    config: Optional[TransactionCostConfig] = None,
    daily_volatility: float = 0.02
) -> Tuple[np.ndarray, dict]:
    """
    Apply realistic transaction costs to backtest returns.
    
    Args:
        returns: Gross returns before costs
        positions: Position sizes at each time step (shares or notional)
        prices: Prices at each time step
        volumes: Daily volumes at each time step
        config: Transaction cost config
        daily_volatility: Average daily volatility
        
    Returns:
        (net_returns, cost_summary)
    """
    if config is None:
        config = TransactionCostConfig()
    
    n = len(returns)
    net_returns = returns.copy()
    
    total_spread_cost = 0
    total_impact_cost = 0
    total_fixed_cost = 0
    
    # Track position changes
    position_changes = np.diff(np.concatenate([[0], positions]))
    
    for i in range(n):
        if abs(position_changes[i]) < 1e-10:
            continue
        
        # Create market conditions
        p = prices[i]
        v = volumes[i] if i < len(volumes) else 1e6
        spread_estimate = max(0.01, p * 0.001)  # Estimate 10 bps spread
        
        market = MarketConditions(
            bid=p - spread_estimate/2,
            ask=p + spread_estimate/2,
            mid_price=p,
            daily_volume=v,
            volatility=daily_volatility,
            avg_trade_size=1000
        )
        
        # Estimate costs
        cost_info = estimate_total_cost(
            shares=abs(position_changes[i]),
            market=market,
            config=config,
            is_buy=position_changes[i] > 0
        )
        
        # Apply to returns
        if abs(positions[i]) > 0:
            cost_as_return = cost_info['total_cost'] / (abs(positions[i]) * p)
            net_returns[i] -= cost_as_return
        
        total_spread_cost += cost_info['spread_cost']
        total_impact_cost += cost_info['permanent_impact'] + cost_info['temporary_impact']
        total_fixed_cost += cost_info['fixed_costs']
    
    total_cost = total_spread_cost + total_impact_cost + total_fixed_cost
    
    summary = {
        'total_cost': total_cost,
        'spread_cost': total_spread_cost,
        'impact_cost': total_impact_cost,
        'fixed_cost': total_fixed_cost,
        'avg_cost_per_trade': total_cost / max(1, np.sum(np.abs(position_changes) > 0)),
        'total_turnover': np.sum(np.abs(position_changes))
    }
    
    return net_returns, summary


# =============================================================================
# SLIPPAGE MODELS
# =============================================================================

def linear_slippage(shares: float, avg_volume: float, price: float) -> float:
    """Simple linear slippage model."""
    participation = abs(shares) / avg_volume
    slippage_bps = participation * 10  # 10 bps per 1% participation
    return slippage_bps * price * abs(shares) / 10000


def square_root_slippage(
    shares: float,
    avg_volume: float,
    price: float,
    volatility: float = 0.02,
    eta: float = 0.1
) -> float:
    """
    Square-root slippage model (industry standard).
    
    Cost = eta * sigma * price * sqrt(shares / volume)
    """
    if avg_volume <= 0:
        return 0
    
    impact = eta * volatility * price * math.sqrt(abs(shares) / avg_volume) * abs(shares)
    return impact


def power_law_slippage(
    shares: float,
    avg_volume: float,
    price: float,
    alpha: float = 0.5,
    beta: float = 0.01
) -> float:
    """
    General power-law slippage model.
    
    Cost = beta * price * (shares / volume)^alpha
    """
    if avg_volume <= 0:
        return 0
    
    participation = abs(shares) / avg_volume
    impact = beta * price * (participation ** alpha) * abs(shares)
    return impact
