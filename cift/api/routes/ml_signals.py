"""
CIFT Markets - ML Signals API Routes

Endpoints for ML-powered trading signals, portfolio recommendations,
and paper trading stats.
"""

from dataclasses import asdict
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.services.ml_signal_service import (
    SignalType,
    TradingSignal,
    get_ml_signal_service,
)


router = APIRouter(prefix="/ml-signals", tags=["ML Signals"])


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class SignalResponse(BaseModel):
    """Trading signal response."""
    symbol: str
    signal_type: str
    confidence: float
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: float
    expected_return_pct: float
    max_loss_pct: float
    hold_duration: str
    urgency: str
    reasons: list[str]
    ai_explanation: str
    timestamp: datetime


class PortfolioRecommendationResponse(BaseModel):
    """Portfolio position recommendation."""
    symbol: str
    current_price: float
    avg_cost: float
    quantity: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Recommendation
    action: str
    confidence: float
    
    # Targets
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Scores
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    ml_prediction: str
    
    # Reasoning
    summary: str
    bullish_factors: list[str]
    bearish_factors: list[str]
    key_risks: list[str]
    
    # Actions
    should_add: bool
    should_trim: bool
    should_hold: bool
    should_exit: bool
    suggested_add_pct: float
    suggested_trim_pct: float


class PaperTradingStatsResponse(BaseModel):
    """Paper trading statistics."""
    total_trades: int
    open_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe_ratio: float
    best_trade: float
    worst_trade: float


class BulkSignalRequest(BaseModel):
    """Request for multiple symbols."""
    symbols: list[str] = Field(..., min_length=1, max_length=20)


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/signal/{symbol}", response_model=SignalResponse)
async def get_signal(
    symbol: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get current ML trading signal for a symbol.
    
    Returns the latest prediction including:
    - Buy/Sell/Hold recommendation
    - Confidence level
    - Price targets
    - AI-powered reasoning
    """
    service = get_ml_signal_service()
    
    # Check cached signal first
    signal = service.signals.get(symbol.upper())
    
    # Generate fresh if stale (>5 min) or missing
    if not signal or (datetime.utcnow() - signal.timestamp).seconds > 300:
        signal = await service.generate_signal(symbol.upper())
    
    if not signal:
        raise HTTPException(
            status_code=404,
            detail=f"Could not generate signal for {symbol}"
        )
    
    return SignalResponse(
        symbol=signal.symbol,
        signal_type=signal.signal_type.value,
        confidence=signal.confidence,
        entry_price=signal.entry_price,
        target_price=signal.target_price,
        stop_loss=signal.stop_loss,
        risk_reward_ratio=signal.risk_reward_ratio,
        expected_return_pct=signal.expected_return_pct,
        max_loss_pct=signal.max_loss_pct,
        hold_duration=signal.hold_duration,
        urgency=signal.urgency,
        reasons=signal.reasons,
        ai_explanation=signal.ai_explanation,
        timestamp=signal.timestamp,
    )


@router.post("/signals/bulk", response_model=list[SignalResponse])
async def get_bulk_signals(
    request: BulkSignalRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get ML signals for multiple symbols at once.
    
    Useful for screening or dashboard views.
    """
    service = get_ml_signal_service()
    signals = []
    
    for symbol in request.symbols:
        try:
            signal = service.signals.get(symbol.upper())
            if not signal or (datetime.utcnow() - signal.timestamp).seconds > 300:
                signal = await service.generate_signal(symbol.upper())
            
            if signal:
                signals.append(SignalResponse(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    risk_reward_ratio=signal.risk_reward_ratio,
                    expected_return_pct=signal.expected_return_pct,
                    max_loss_pct=signal.max_loss_pct,
                    hold_duration=signal.hold_duration,
                    urgency=signal.urgency,
                    reasons=signal.reasons,
                    ai_explanation=signal.ai_explanation,
                    timestamp=signal.timestamp,
                ))
        except Exception as e:
            logger.warning(f"Failed to get signal for {symbol}: {e}")
    
    return signals


@router.get("/portfolio/recommendations", response_model=list[PortfolioRecommendationResponse])
async def get_portfolio_recommendations(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get detailed ML recommendations for all positions in portfolio.
    
    Each position includes:
    - Action recommendation (Buy More, Hold, Trim, Exit)
    - Confidence level
    - Technical/Fundamental/Sentiment scores
    - AI-powered reasoning
    - Key risks
    - Suggested position changes
    """
    service = get_ml_signal_service()
    
    recommendations = await service.get_portfolio_recommendations(user_id)
    
    return [
        PortfolioRecommendationResponse(
            symbol=rec.symbol,
            current_price=rec.current_price,
            avg_cost=rec.avg_cost,
            quantity=rec.quantity,
            market_value=rec.market_value,
            unrealized_pnl=rec.unrealized_pnl,
            unrealized_pnl_pct=rec.unrealized_pnl_pct,
            action=rec.action.value,
            confidence=rec.confidence,
            target_price=rec.target_price,
            stop_loss=rec.stop_loss,
            technical_score=rec.technical_score,
            fundamental_score=rec.fundamental_score,
            sentiment_score=rec.sentiment_score,
            ml_prediction=rec.ml_prediction,
            summary=rec.summary,
            bullish_factors=rec.bullish_factors,
            bearish_factors=rec.bearish_factors,
            key_risks=rec.key_risks,
            should_add=rec.should_add,
            should_trim=rec.should_trim,
            should_hold=rec.should_hold,
            should_exit=rec.should_exit,
            suggested_add_pct=rec.suggested_add_pct,
            suggested_trim_pct=rec.suggested_trim_pct,
        )
        for rec in recommendations
    ]


@router.get("/paper-trading/stats", response_model=PaperTradingStatsResponse)
async def get_paper_trading_stats(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get paper trading performance statistics.
    
    Includes:
    - Win rate
    - Total P&L
    - Sharpe ratio
    - Trade counts
    """
    service = get_ml_signal_service()
    stats = service.get_paper_trading_stats()
    
    return PaperTradingStatsResponse(**stats)


@router.get("/signals/active")
async def get_active_signals(
    min_confidence: float = 0.6,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get all active signals above a confidence threshold.
    
    Useful for finding immediate trading opportunities.
    """
    service = get_ml_signal_service()
    
    active_signals = [
        {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "target_price": signal.target_price,
            "urgency": signal.urgency,
            "reasons": signal.reasons[:3],
        }
        for signal in service.signals.values()
        if signal.confidence >= min_confidence
        and signal.signal_type != SignalType.HOLD
    ]
    
    # Sort by confidence descending
    active_signals.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "count": len(active_signals),
        "signals": active_signals,
    }


@router.post("/signals/refresh/{symbol}")
async def refresh_signal(
    symbol: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Force refresh the signal for a symbol.
    """
    service = get_ml_signal_service()
    signal = await service.generate_signal(symbol.upper())
    
    if not signal:
        raise HTTPException(
            status_code=404,
            detail=f"Could not generate signal for {symbol}"
        )
    
    return {
        "status": "refreshed",
        "symbol": signal.symbol,
        "signal_type": signal.signal_type.value,
        "confidence": signal.confidence,
    }


@router.get("/status")
async def get_ml_service_status():
    """
    Get status of the ML signal service.
    """
    service = get_ml_signal_service()
    
    return {
        "running": service.running,
        "tracked_symbols": len(service.signals),
        "open_paper_trades": len([t for t in service.paper_trades if t.status == "open"]),
        "total_paper_trades": len(service.paper_trades),
        "sharpe_ratio": service.calculate_sharpe_ratio(),
        "alert_threshold": service.alert_confidence_threshold,
        "check_interval_seconds": service.check_interval,
    }
