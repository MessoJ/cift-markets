"""
Kelly Criterion and Position Sizing

Implements optimal position sizing:
1. Full Kelly criterion
2. Fractional Kelly (risk reduction)
3. Multi-asset Kelly with covariance
4. Drawdown-adjusted Kelly
5. Meta-labeled position sizing

Reference:
- Kelly (1956), "A New Interpretation of Information Rate"
- Thorp (2006), "The Kelly Criterion in Blackjack Sports Betting"
- De Prado (2018), "Advances in Financial Machine Learning", Chapter 10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np


# =============================================================================
# BASIC KELLY CRITERION
# =============================================================================

def kelly_criterion_simple(
    win_probability: float,
    win_loss_ratio: float
) -> float:
    """
    Classic Kelly criterion for binary outcomes.
    
    f* = (p * b - q) / b = p - q/b
    
    where:
    - p = probability of winning
    - q = 1 - p = probability of losing
    - b = odds (win/loss ratio)
    
    Args:
        win_probability: Probability of winning (0 to 1)
        win_loss_ratio: Expected win / Expected loss (absolute values)
        
    Returns:
        Optimal fraction of capital to bet (can be negative = short)
    """
    p = win_probability
    q = 1 - p
    b = win_loss_ratio
    
    if b <= 0:
        return 0.0
    
    kelly = (p * b - q) / b
    
    return kelly


def kelly_criterion_continuous(
    expected_return: float,
    variance: float
) -> float:
    """
    Kelly criterion for continuous returns (Gaussian approximation).
    
    f* = mu / sigma^2
    
    This is the leverage that maximizes log utility.
    
    Args:
        expected_return: Expected return (e.g., 0.05 for 5%)
        variance: Variance of returns
        
    Returns:
        Optimal leverage (fraction of capital)
    """
    if variance <= 0:
        return 0.0
    
    return expected_return / variance


def fractional_kelly(
    kelly_fraction: float,
    fraction: float = 0.5,
    max_leverage: float = 2.0
) -> float:
    """
    Fractional Kelly - risk-adjusted position sizing.
    
    Using a fraction of Kelly reduces:
    - Variance by fraction^2
    - Expected growth by only fraction
    
    Typical fractions:
    - 0.25: Very conservative
    - 0.50: Standard (recommended)
    - 0.75: Aggressive
    - 1.00: Full Kelly (risky)
    
    Args:
        kelly_fraction: Full Kelly fraction
        fraction: Kelly fraction to use (0.5 = half Kelly)
        max_leverage: Maximum allowed leverage
        
    Returns:
        Risk-adjusted position size
    """
    adjusted = kelly_fraction * fraction
    
    # Apply leverage limit
    return np.clip(adjusted, -max_leverage, max_leverage)


# =============================================================================
# MULTI-ASSET KELLY
# =============================================================================

def multi_asset_kelly(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0
) -> np.ndarray:
    """
    Multi-asset Kelly criterion (optimal portfolio weights).
    
    For multiple assets with covariance, optimal weights are:
    f* = Sigma^{-1} * (mu - r_f)
    
    This maximizes expected log utility.
    
    Args:
        expected_returns: Expected returns vector (N,)
        covariance_matrix: Covariance matrix (N x N)
        risk_free_rate: Risk-free rate
        
    Returns:
        Optimal weights vector
    """
    mu = np.array(expected_returns) - risk_free_rate
    sigma = np.array(covariance_matrix)
    
    try:
        # Inverse of covariance
        sigma_inv = np.linalg.inv(sigma)
        
        # Kelly weights
        weights = sigma_inv @ mu
        
    except np.linalg.LinAlgError:
        # Singular matrix - use pseudo-inverse
        sigma_inv = np.linalg.pinv(sigma)
        weights = sigma_inv @ mu
    
    return weights


def constrained_multi_kelly(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    max_weight: float = 0.5,
    max_gross_leverage: float = 2.0,
    risk_free_rate: float = 0.0
) -> np.ndarray:
    """
    Constrained multi-asset Kelly with position limits.
    
    Args:
        expected_returns: Expected returns
        covariance_matrix: Covariance matrix
        max_weight: Maximum weight per asset
        max_gross_leverage: Maximum gross leverage (sum of |weights|)
        risk_free_rate: Risk-free rate
        
    Returns:
        Constrained weights
    """
    # Get unconstrained Kelly
    weights = multi_asset_kelly(expected_returns, covariance_matrix, risk_free_rate)
    
    # Clip individual positions
    weights = np.clip(weights, -max_weight, max_weight)
    
    # Scale down if gross leverage exceeded
    gross_leverage = np.sum(np.abs(weights))
    
    if gross_leverage > max_gross_leverage:
        weights = weights * max_gross_leverage / gross_leverage
    
    return weights


# =============================================================================
# DRAWDOWN-ADJUSTED SIZING
# =============================================================================

@dataclass
class DrawdownState:
    """Track drawdown state for position sizing."""
    peak_equity: float = 1.0
    current_equity: float = 1.0
    
    @property
    def drawdown(self) -> float:
        """Current drawdown as fraction (0 = no drawdown, 0.2 = 20% down)."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def update(self, new_equity: float) -> float:
        """Update with new equity and return current drawdown."""
        self.current_equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        return self.drawdown


def drawdown_kelly(
    base_kelly: float,
    current_drawdown: float,
    max_drawdown: float = 0.20,
    reduction_speed: float = 2.0
) -> float:
    """
    Reduce position size based on current drawdown.
    
    As drawdown increases, reduce Kelly fraction to preserve capital.
    
    f_adjusted = f_kelly * (1 - (dd / max_dd)^speed)
    
    Args:
        base_kelly: Base Kelly fraction
        current_drawdown: Current drawdown (e.g., 0.10 for 10%)
        max_drawdown: Maximum acceptable drawdown
        reduction_speed: How aggressively to reduce (higher = faster)
        
    Returns:
        Drawdown-adjusted position size
    """
    if current_drawdown <= 0:
        return base_kelly
    
    if current_drawdown >= max_drawdown:
        # At or beyond max drawdown - stop trading
        return 0.0
    
    # Reduction factor
    dd_ratio = current_drawdown / max_drawdown
    reduction = (1 - dd_ratio ** reduction_speed)
    
    return base_kelly * reduction


def time_weighted_kelly(
    base_kelly: float,
    days_since_loss: int,
    recovery_days: int = 20
) -> float:
    """
    Gradually return to full Kelly after a loss.
    
    Prevents revenge trading and allows system to stabilize.
    
    Args:
        base_kelly: Target Kelly fraction
        days_since_loss: Days since last significant loss
        recovery_days: Days to return to full Kelly
        
    Returns:
        Time-adjusted position size
    """
    if days_since_loss >= recovery_days:
        return base_kelly
    
    # Linear ramp-up
    recovery_fraction = days_since_loss / recovery_days
    
    return base_kelly * recovery_fraction


# =============================================================================
# META-LABELED POSITION SIZING
# =============================================================================

def meta_label_sizing(
    primary_signal: float,
    meta_probability: float,
    confidence_threshold: float = 0.55,
    max_size: float = 1.0
) -> float:
    """
    Position sizing based on meta-labeling.
    
    Meta-model predicts probability that primary signal will be profitable.
    Size position based on meta-model confidence.
    
    Args:
        primary_signal: Direction from primary model (-1 to 1)
        meta_probability: Probability primary signal is correct (0 to 1)
        confidence_threshold: Minimum confidence to trade
        max_size: Maximum position size
        
    Returns:
        Sized position (-max_size to max_size)
    """
    if meta_probability < confidence_threshold:
        return 0.0
    
    # Scale by excess confidence
    confidence_excess = (meta_probability - confidence_threshold) / (1 - confidence_threshold)
    
    # Size = direction * confidence * max_size
    size = np.sign(primary_signal) * confidence_excess * max_size
    
    return float(np.clip(size, -max_size, max_size))


def probability_weighted_kelly(
    predicted_probability: float,
    win_loss_ratio: float,
    confidence_calibration: float = 1.0
) -> float:
    """
    Kelly sizing using ML model probabilities.
    
    Args:
        predicted_probability: Model's win probability prediction
        win_loss_ratio: Expected win/loss ratio
        confidence_calibration: Multiplier for probability (< 1 = more conservative)
        
    Returns:
        Position size
    """
    # Calibrate probability
    p = predicted_probability * confidence_calibration
    p = np.clip(p, 0.01, 0.99)
    
    # Kelly
    return kelly_criterion_simple(p, win_loss_ratio)


# =============================================================================
# VOLATILITY TARGETING
# =============================================================================

def volatility_target_sizing(
    base_size: float,
    current_volatility: float,
    target_volatility: float,
    max_scale: float = 3.0,
    min_scale: float = 0.2
) -> float:
    """
    Scale position size to target portfolio volatility.
    
    When volatility is low, increase size.
    When volatility is high, decrease size.
    
    Args:
        base_size: Base position size
        current_volatility: Realized volatility (annualized)
        target_volatility: Target volatility
        max_scale: Maximum scaling factor
        min_scale: Minimum scaling factor
        
    Returns:
        Volatility-adjusted position size
    """
    if current_volatility <= 0:
        return base_size
    
    scale = target_volatility / current_volatility
    scale = np.clip(scale, min_scale, max_scale)
    
    return base_size * scale


def inverse_vol_weights(
    volatilities: np.ndarray,
    target_vol: Optional[float] = None
) -> np.ndarray:
    """
    Weight assets inversely proportional to volatility.
    
    Risk parity for uncorrelated assets.
    
    Args:
        volatilities: Asset volatilities
        target_vol: Optional target portfolio volatility
        
    Returns:
        Weights
    """
    inv_vol = 1 / (volatilities + 1e-10)
    weights = inv_vol / inv_vol.sum()
    
    if target_vol is not None:
        # Approximate portfolio vol (assumes correlation = 0)
        port_vol = np.sqrt(np.sum((weights * volatilities) ** 2))
        scale = target_vol / port_vol if port_vol > 0 else 1.0
        weights = weights * scale
    
    return weights


# =============================================================================
# POSITION SIZING ENGINE
# =============================================================================

@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    method: str = "fractional_kelly"  # 'kelly', 'fractional_kelly', 'vol_target', 'meta'
    
    # Kelly parameters
    kelly_fraction: float = 0.5
    max_leverage: float = 2.0
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% annualized
    vol_lookback: int = 20
    
    # Drawdown control
    max_drawdown: float = 0.20
    reduce_on_drawdown: bool = True
    
    # Meta-labeling
    meta_confidence_threshold: float = 0.55


class PositionSizer:
    """
    Position sizing engine combining multiple methods.
    """
    
    def __init__(self, config: SizingConfig):
        self.config = config
        self.drawdown_state = DrawdownState()
        self.volatility_history: List[float] = []
    
    def update_equity(self, equity: float) -> float:
        """Update equity and return current drawdown."""
        return self.drawdown_state.update(equity)
    
    def update_volatility(self, volatility: float) -> None:
        """Update volatility history."""
        self.volatility_history.append(volatility)
        
        # Keep only recent
        if len(self.volatility_history) > self.config.vol_lookback:
            self.volatility_history = self.volatility_history[-self.config.vol_lookback:]
    
    @property
    def current_volatility(self) -> float:
        """Get current volatility estimate."""
        if not self.volatility_history:
            return self.config.target_volatility
        return np.mean(self.volatility_history)
    
    def calculate_size(
        self,
        expected_return: float,
        variance: float,
        meta_probability: Optional[float] = None,
        signal_direction: float = 1.0
    ) -> float:
        """
        Calculate position size based on configured method.
        
        Args:
            expected_return: Expected return of trade
            variance: Variance of returns
            meta_probability: Optional meta-model probability
            signal_direction: Trade direction (-1 or 1)
            
        Returns:
            Position size (can be fractional, negative = short)
        """
        # Base Kelly
        if variance > 0:
            base_kelly = kelly_criterion_continuous(expected_return, variance)
        else:
            base_kelly = 0.0
        
        # Apply method-specific adjustments
        if self.config.method == "kelly":
            size = base_kelly
        
        elif self.config.method == "fractional_kelly":
            size = fractional_kelly(
                base_kelly,
                fraction=self.config.kelly_fraction,
                max_leverage=self.config.max_leverage
            )
        
        elif self.config.method == "vol_target":
            size = fractional_kelly(base_kelly, self.config.kelly_fraction)
            size = volatility_target_sizing(
                size,
                self.current_volatility,
                self.config.target_volatility
            )
        
        elif self.config.method == "meta" and meta_probability is not None:
            size = meta_label_sizing(
                signal_direction,
                meta_probability,
                self.config.meta_confidence_threshold,
                max_size=self.config.max_leverage
            )
        
        else:
            size = fractional_kelly(base_kelly, self.config.kelly_fraction)
        
        # Drawdown adjustment
        if self.config.reduce_on_drawdown:
            current_dd = self.drawdown_state.drawdown
            size = drawdown_kelly(
                size,
                current_dd,
                self.config.max_drawdown
            )
        
        # Direction
        size = size * np.sign(signal_direction)
        
        # Final leverage cap
        size = np.clip(size, -self.config.max_leverage, self.config.max_leverage)
        
        return float(size)
    
    def calculate_portfolio_sizes(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate sizes for portfolio of assets.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            
        Returns:
            Position sizes vector
        """
        # Multi-asset Kelly
        sizes = constrained_multi_kelly(
            expected_returns,
            covariance_matrix,
            max_weight=self.config.max_leverage / len(expected_returns),
            max_gross_leverage=self.config.max_leverage
        )
        
        # Apply fraction
        sizes = sizes * self.config.kelly_fraction
        
        # Drawdown adjustment
        if self.config.reduce_on_drawdown:
            current_dd = self.drawdown_state.drawdown
            adjustment = 1.0 - (current_dd / self.config.max_drawdown) ** 2
            adjustment = max(0.0, adjustment)
            sizes = sizes * adjustment
        
        return sizes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_win_probability(
    returns: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Estimate win probability from historical returns.
    
    Args:
        returns: Historical returns
        threshold: Minimum return to count as win
        
    Returns:
        Estimated win probability
    """
    wins = np.sum(returns > threshold)
    total = len(returns)
    
    if total == 0:
        return 0.5
    
    return wins / total


def estimate_win_loss_ratio(
    returns: np.ndarray
) -> float:
    """
    Estimate average win / average loss ratio.
    
    Args:
        returns: Historical returns
        
    Returns:
        Win/loss ratio
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = -np.mean(losses) if len(losses) > 0 else 1  # Negative to positive
    
    if avg_loss <= 0:
        return float('inf')
    
    return avg_win / avg_loss


def optimal_f_from_returns(
    returns: np.ndarray,
    fraction: float = 0.5
) -> float:
    """
    Calculate optimal f directly from historical returns.
    
    Args:
        returns: Historical returns
        fraction: Fraction of Kelly to use
        
    Returns:
        Recommended position size
    """
    if len(returns) < 10:
        return 0.0
    
    # Continuous Kelly
    mu = np.mean(returns)
    var = np.var(returns)
    
    if var <= 0:
        return 0.0
    
    kelly = mu / var
    
    return kelly * fraction
