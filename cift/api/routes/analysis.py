"""
CIFT Markets - Stock Analysis API Routes

Comprehensive stock analysis and trading recommendations.

Endpoints:
- GET /analysis/{symbol} - Full stock analysis with ratings
- GET /analysis/{symbol}/technical - Technical analysis only
- GET /analysis/{symbol}/fundamental - Fundamental analysis only
- GET /analysis/{symbol}/quick - Quick overview (faster)
- POST /analysis/batch - Analyze multiple stocks

Performance:
- Single stock: <200ms
- Quick analysis: <50ms
- Batch (10 stocks): <500ms
"""

from dataclasses import asdict
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from cift.services.stock_analyzer import (
    MomentumAnalysis,
    Rating,
    RiskAnalysis,
    SignalStrength,
    StockAnalysis,
    stock_analyzer,
)


router = APIRouter(prefix="/analysis", tags=["Stock Analysis"])


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class TechnicalResponse(BaseModel):
    """Technical analysis response."""
    
    score: float
    signal: str
    rsi_14: float | None = None
    rsi_signal: str | None = None
    macd_line: float | None = None
    macd_signal_line: float | None = None
    macd_histogram: float | None = None
    macd_crossover: str | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    price_vs_sma: dict | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_position: str | None = None
    atr_14: float | None = None
    atr_percent: float | None = None
    volume_trend: str | None = None
    volume_vs_avg: float | None = None
    support_levels: list[float] = []
    resistance_levels: list[float] = []
    short_term_trend: str | None = None
    medium_term_trend: str | None = None
    long_term_trend: str | None = None


class FundamentalResponse(BaseModel):
    """Fundamental analysis response."""
    
    score: float
    signal: str
    pe_ratio: float | None = None
    pe_percentile: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda: float | None = None
    peg_ratio: float | None = None
    roe: float | None = None
    roa: float | None = None
    profit_margin: float | None = None
    operating_margin: float | None = None
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    revenue_growth_3y: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    interest_coverage: float | None = None
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    dividend_growth_5y: float | None = None
    earnings_surprise_avg: float | None = None
    earnings_consistency: float | None = None


class SentimentResponse(BaseModel):
    """Sentiment analysis response."""
    
    score: float
    signal: str
    news_sentiment: float | None = None
    news_volume: int | None = None
    news_trend: str | None = None
    analyst_rating: str | None = None
    analyst_target: float | None = None
    analyst_target_upside: float | None = None
    analyst_count: int | None = None
    institutional_ownership: float | None = None
    institutional_change: float | None = None
    insider_net_shares: int | None = None
    insider_sentiment: str | None = None


class RiskResponse(BaseModel):
    """Risk analysis response."""
    
    score: float
    risk_level: str
    volatility_30d: float | None = None
    volatility_percentile: float | None = None
    beta: float | None = None
    max_drawdown_1y: float | None = None
    current_drawdown: float | None = None
    var_95: float | None = None
    avg_volume_20d: float | None = None
    bid_ask_spread: float | None = None
    correlation_spy: float | None = None


class MomentumResponse(BaseModel):
    """Momentum analysis response."""
    
    score: float
    signal: str
    return_1d: float | None = None
    return_1w: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_12m: float | None = None
    return_ytd: float | None = None
    momentum_12_1: float | None = None
    momentum_percentile: float | None = None


class TradeSuggestion(BaseModel):
    """Trade suggestion (not financial advice)."""
    
    action: str | None = None
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    risk_reward: float | None = None


class FullAnalysisResponse(BaseModel):
    """Complete stock analysis response."""
    
    symbol: str
    timestamp: datetime
    
    # Price info
    price: float
    change: float
    change_percent: float
    
    # Overall
    overall_score: float
    rating: str
    confidence: float
    
    # Components
    technical: TechnicalResponse
    fundamental: FundamentalResponse
    sentiment: SentimentResponse
    risk: RiskResponse
    momentum: MomentumResponse
    
    # Insights
    bullish_factors: list[str]
    bearish_factors: list[str]
    key_risks: list[str]
    
    # Trade suggestion
    trade_suggestion: TradeSuggestion
    
    # Meta
    analysis_latency_ms: float
    
    class Config:
        from_attributes = True


class QuickAnalysisResponse(BaseModel):
    """Quick analysis response (faster, less detail)."""
    
    symbol: str
    price: float
    change_percent: float
    overall_score: float
    rating: str
    confidence: float
    technical_signal: str
    fundamental_signal: str
    sentiment_signal: str
    risk_level: str
    momentum_signal: str
    top_bullish: str | None = None
    top_bearish: str | None = None
    suggested_action: str | None = None


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""
    
    symbols: list[str]
    quick: bool = True  # Use quick analysis by default for batch


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response."""
    
    analyses: list[QuickAnalysisResponse]
    total_time_ms: float


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/{symbol}", response_model=FullAnalysisResponse)
async def analyze_stock(symbol: str):
    """
    Get comprehensive stock analysis.
    
    Returns:
    - Overall score (0-100) and rating (STRONG_BUY to STRONG_SELL)
    - Technical analysis with indicators
    - Fundamental analysis with valuation/quality metrics
    - Sentiment analysis from news
    - Risk metrics (volatility, drawdown, VaR)
    - Momentum factor scores
    - Human-readable insights
    - Trade suggestion (not financial advice)
    
    Performance: <200ms
    
    **Disclaimer**: This is not financial advice. The analysis is for
    informational purposes only. Always do your own research before investing.
    """
    try:
        analysis = await stock_analyzer.analyze(symbol)
        
        return FullAnalysisResponse(
            symbol=analysis.symbol,
            timestamp=analysis.timestamp,
            price=analysis.price,
            change=analysis.change,
            change_percent=analysis.change_percent,
            overall_score=round(analysis.overall_score, 1),
            rating=analysis.rating.value,
            confidence=round(analysis.confidence, 2),
            technical=TechnicalResponse(
                score=round(analysis.technical.score, 1),
                signal=analysis.technical.signal.value,
                rsi_14=round(analysis.technical.rsi_14, 2) if analysis.technical.rsi_14 else None,
                rsi_signal=analysis.technical.rsi_signal,
                macd_line=round(analysis.technical.macd_line, 4) if analysis.technical.macd_line else None,
                macd_signal_line=round(analysis.technical.macd_signal_line, 4) if analysis.technical.macd_signal_line else None,
                macd_histogram=round(analysis.technical.macd_histogram, 4) if analysis.technical.macd_histogram else None,
                macd_crossover=analysis.technical.macd_crossover,
                sma_20=round(analysis.technical.sma_20, 2) if analysis.technical.sma_20 else None,
                sma_50=round(analysis.technical.sma_50, 2) if analysis.technical.sma_50 else None,
                sma_200=round(analysis.technical.sma_200, 2) if analysis.technical.sma_200 else None,
                price_vs_sma=analysis.technical.price_vs_sma,
                bollinger_upper=round(analysis.technical.bollinger_upper, 2) if analysis.technical.bollinger_upper else None,
                bollinger_lower=round(analysis.technical.bollinger_lower, 2) if analysis.technical.bollinger_lower else None,
                bollinger_position=analysis.technical.bollinger_position,
                atr_14=round(analysis.technical.atr_14, 2) if analysis.technical.atr_14 else None,
                atr_percent=round(analysis.technical.atr_percent, 2) if analysis.technical.atr_percent else None,
                volume_trend=analysis.technical.volume_trend,
                volume_vs_avg=round(analysis.technical.volume_vs_avg, 2) if analysis.technical.volume_vs_avg else None,
                support_levels=[round(s, 2) for s in analysis.technical.support_levels],
                resistance_levels=[round(r, 2) for r in analysis.technical.resistance_levels],
                short_term_trend=analysis.technical.short_term_trend,
                medium_term_trend=analysis.technical.medium_term_trend,
                long_term_trend=analysis.technical.long_term_trend,
            ),
            fundamental=FundamentalResponse(
                score=round(analysis.fundamental.score, 1),
                signal=analysis.fundamental.signal.value,
                pe_ratio=round(analysis.fundamental.pe_ratio, 2) if analysis.fundamental.pe_ratio else None,
                pb_ratio=round(analysis.fundamental.pb_ratio, 2) if analysis.fundamental.pb_ratio else None,
                ps_ratio=round(analysis.fundamental.ps_ratio, 2) if analysis.fundamental.ps_ratio else None,
                roe=round(analysis.fundamental.roe, 2) if analysis.fundamental.roe else None,
                roa=round(analysis.fundamental.roa, 2) if analysis.fundamental.roa else None,
                profit_margin=round(analysis.fundamental.profit_margin, 2) if analysis.fundamental.profit_margin else None,
                revenue_growth_yoy=round(analysis.fundamental.revenue_growth_yoy, 2) if analysis.fundamental.revenue_growth_yoy else None,
                earnings_growth_yoy=round(analysis.fundamental.earnings_growth_yoy, 2) if analysis.fundamental.earnings_growth_yoy else None,
                debt_to_equity=round(analysis.fundamental.debt_to_equity, 2) if analysis.fundamental.debt_to_equity else None,
                current_ratio=round(analysis.fundamental.current_ratio, 2) if analysis.fundamental.current_ratio else None,
                dividend_yield=round(analysis.fundamental.dividend_yield, 2) if analysis.fundamental.dividend_yield else None,
                payout_ratio=round(analysis.fundamental.payout_ratio, 2) if analysis.fundamental.payout_ratio else None,
            ),
            sentiment=SentimentResponse(
                score=round(analysis.sentiment.score, 1),
                signal=analysis.sentiment.signal.value,
                news_sentiment=round(analysis.sentiment.news_sentiment, 2) if analysis.sentiment.news_sentiment else None,
                news_volume=analysis.sentiment.news_volume,
                news_trend=analysis.sentiment.news_trend,
            ),
            risk=RiskResponse(
                score=round(analysis.risk.score, 1),
                risk_level=analysis.risk.risk_level,
                volatility_30d=round(analysis.risk.volatility_30d, 2) if analysis.risk.volatility_30d else None,
                max_drawdown_1y=round(analysis.risk.max_drawdown_1y, 2) if analysis.risk.max_drawdown_1y else None,
                current_drawdown=round(analysis.risk.current_drawdown, 2) if analysis.risk.current_drawdown else None,
                var_95=round(analysis.risk.var_95, 2) if analysis.risk.var_95 else None,
            ),
            momentum=MomentumResponse(
                score=round(analysis.momentum.score, 1),
                signal=analysis.momentum.signal.value,
                return_1d=round(analysis.momentum.return_1d, 2) if analysis.momentum.return_1d else None,
                return_1w=round(analysis.momentum.return_1w, 2) if analysis.momentum.return_1w else None,
                return_1m=round(analysis.momentum.return_1m, 2) if analysis.momentum.return_1m else None,
                return_3m=round(analysis.momentum.return_3m, 2) if analysis.momentum.return_3m else None,
                return_6m=round(analysis.momentum.return_6m, 2) if analysis.momentum.return_6m else None,
                return_12m=round(analysis.momentum.return_12m, 2) if analysis.momentum.return_12m else None,
                momentum_12_1=round(analysis.momentum.momentum_12_1, 2) if analysis.momentum.momentum_12_1 else None,
            ),
            bullish_factors=analysis.bullish_factors,
            bearish_factors=analysis.bearish_factors,
            key_risks=analysis.key_risks,
            trade_suggestion=TradeSuggestion(
                action=analysis.suggested_action,
                entry_zone_low=analysis.entry_zone[0] if analysis.entry_zone else None,
                entry_zone_high=analysis.entry_zone[1] if analysis.entry_zone else None,
                stop_loss=analysis.stop_loss,
                target_1=analysis.target_1,
                target_2=analysis.target_2,
                risk_reward=analysis.risk_reward,
            ),
            analysis_latency_ms=round(analysis.analysis_latency_ms, 2),
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{symbol}/quick", response_model=QuickAnalysisResponse)
async def quick_analyze_stock(symbol: str):
    """
    Get quick stock analysis (faster, less detail).
    
    Returns key signals without full detail:
    - Overall score and rating
    - Signal for each component
    - Top bullish/bearish factor
    - Suggested action
    
    Performance: <100ms
    """
    try:
        analysis = await stock_analyzer.analyze(symbol)
        
        return QuickAnalysisResponse(
            symbol=analysis.symbol,
            price=analysis.price,
            change_percent=analysis.change_percent,
            overall_score=round(analysis.overall_score, 1),
            rating=analysis.rating.value,
            confidence=round(analysis.confidence, 2),
            technical_signal=analysis.technical.signal.value,
            fundamental_signal=analysis.fundamental.signal.value,
            sentiment_signal=analysis.sentiment.signal.value,
            risk_level=analysis.risk.risk_level,
            momentum_signal=analysis.momentum.signal.value,
            top_bullish=analysis.bullish_factors[0] if analysis.bullish_factors else None,
            top_bearish=analysis.bearish_factors[0] if analysis.bearish_factors else None,
            suggested_action=analysis.suggested_action,
        )
        
    except Exception as e:
        logger.error(f"Quick analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{symbol}/technical", response_model=TechnicalResponse)
async def get_technical_analysis(symbol: str):
    """
    Get technical analysis only.
    
    Includes:
    - RSI, MACD, SMAs, Bollinger Bands
    - Volume analysis
    - Support/Resistance levels
    - Trend analysis
    """
    try:
        analysis = await stock_analyzer.analyze(symbol)
        t = analysis.technical
        
        return TechnicalResponse(
            score=round(t.score, 1),
            signal=t.signal.value,
            rsi_14=round(t.rsi_14, 2) if t.rsi_14 else None,
            rsi_signal=t.rsi_signal,
            macd_line=round(t.macd_line, 4) if t.macd_line else None,
            macd_signal_line=round(t.macd_signal_line, 4) if t.macd_signal_line else None,
            macd_histogram=round(t.macd_histogram, 4) if t.macd_histogram else None,
            macd_crossover=t.macd_crossover,
            sma_20=round(t.sma_20, 2) if t.sma_20 else None,
            sma_50=round(t.sma_50, 2) if t.sma_50 else None,
            sma_200=round(t.sma_200, 2) if t.sma_200 else None,
            price_vs_sma=t.price_vs_sma,
            bollinger_upper=round(t.bollinger_upper, 2) if t.bollinger_upper else None,
            bollinger_lower=round(t.bollinger_lower, 2) if t.bollinger_lower else None,
            bollinger_position=t.bollinger_position,
            atr_14=round(t.atr_14, 2) if t.atr_14 else None,
            atr_percent=round(t.atr_percent, 2) if t.atr_percent else None,
            volume_trend=t.volume_trend,
            volume_vs_avg=round(t.volume_vs_avg, 2) if t.volume_vs_avg else None,
            support_levels=[round(s, 2) for s in t.support_levels],
            resistance_levels=[round(r, 2) for r in t.resistance_levels],
            short_term_trend=t.short_term_trend,
            medium_term_trend=t.medium_term_trend,
            long_term_trend=t.long_term_trend,
        )
        
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{symbol}/fundamental", response_model=FundamentalResponse)
async def get_fundamental_analysis(symbol: str):
    """
    Get fundamental analysis only.
    
    Includes:
    - Valuation (P/E, P/B, P/S)
    - Profitability (ROE, ROA, margins)
    - Growth rates
    - Financial health
    - Dividends
    """
    try:
        analysis = await stock_analyzer.analyze(symbol)
        f = analysis.fundamental
        
        return FundamentalResponse(
            score=round(f.score, 1),
            signal=f.signal.value,
            pe_ratio=round(f.pe_ratio, 2) if f.pe_ratio else None,
            pb_ratio=round(f.pb_ratio, 2) if f.pb_ratio else None,
            ps_ratio=round(f.ps_ratio, 2) if f.ps_ratio else None,
            roe=round(f.roe, 2) if f.roe else None,
            roa=round(f.roa, 2) if f.roa else None,
            profit_margin=round(f.profit_margin, 2) if f.profit_margin else None,
            revenue_growth_yoy=round(f.revenue_growth_yoy, 2) if f.revenue_growth_yoy else None,
            earnings_growth_yoy=round(f.earnings_growth_yoy, 2) if f.earnings_growth_yoy else None,
            debt_to_equity=round(f.debt_to_equity, 2) if f.debt_to_equity else None,
            current_ratio=round(f.current_ratio, 2) if f.current_ratio else None,
            dividend_yield=round(f.dividend_yield, 2) if f.dividend_yield else None,
            payout_ratio=round(f.payout_ratio, 2) if f.payout_ratio else None,
        )
        
    except Exception as e:
        logger.error(f"Fundamental analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze_stocks(request: BatchAnalysisRequest):
    """
    Analyze multiple stocks at once.
    
    Use this for portfolio analysis or screening results.
    
    Performance: ~50ms per stock (parallelized)
    """
    import asyncio
    
    start_time = datetime.utcnow()
    
    if len(request.symbols) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 symbols per batch request"
        )
    
    try:
        # Analyze all symbols concurrently
        tasks = [stock_analyzer.analyze(symbol) for symbol in request.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyses = []
        for symbol, result in zip(request.symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for {symbol}: {result}")
                continue
            
            analysis = result
            analyses.append(QuickAnalysisResponse(
                symbol=analysis.symbol,
                price=analysis.price,
                change_percent=analysis.change_percent,
                overall_score=round(analysis.overall_score, 1),
                rating=analysis.rating.value,
                confidence=round(analysis.confidence, 2),
                technical_signal=analysis.technical.signal.value,
                fundamental_signal=analysis.fundamental.signal.value,
                sentiment_signal=analysis.sentiment.signal.value,
                risk_level=analysis.risk.risk_level,
                momentum_signal=analysis.momentum.signal.value,
                top_bullish=analysis.bullish_factors[0] if analysis.bullish_factors else None,
                top_bearish=analysis.bearish_factors[0] if analysis.bearish_factors else None,
                suggested_action=analysis.suggested_action,
            ))
        
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return BatchAnalysisResponse(
            analyses=analyses,
            total_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/screener/top-rated")
async def get_top_rated_stocks(
    limit: int = Query(10, ge=1, le=50),
    min_score: float = Query(60.0, ge=0, le=100),
):
    """
    Get top-rated stocks based on analysis scores.
    
    Analyzes cached stocks and returns those with highest scores.
    """
    import asyncio
    from cift.core.database import get_postgres_pool
    
    try:
        pool = await get_postgres_pool()
        
        # Get symbols with recent data
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT symbol 
                FROM market_data_cache 
                WHERE updated_at > NOW() - INTERVAL '1 day'
                LIMIT 100
                """
            )
        
        if not rows:
            return {"stocks": [], "message": "No recent market data available"}
        
        symbols = [row["symbol"] for row in rows]
        
        # Analyze all
        tasks = [stock_analyzer.analyze(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter and sort
        scored_stocks = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                continue
            if result.overall_score >= min_score:
                scored_stocks.append({
                    "symbol": result.symbol,
                    "price": result.price,
                    "change_percent": result.change_percent,
                    "overall_score": round(result.overall_score, 1),
                    "rating": result.rating.value,
                    "top_reason": result.bullish_factors[0] if result.bullish_factors else None,
                })
        
        # Sort by score
        scored_stocks.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return {
            "stocks": scored_stocks[:limit],
            "total_analyzed": len(symbols),
            "qualifying_count": len(scored_stocks),
        }
        
    except Exception as e:
        logger.error(f"Top rated screener failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ML MODEL STATUS ENDPOINTS
# ============================================================================


@router.get("/models/status")
async def get_model_status():
    """
    Get status of all ML models.
    
    Returns detailed information about which models are trained,
    their capabilities, and honest assessment of system state.
    """
    try:
        from cift.services.model_status_service import model_status_service
        
        capabilities = model_status_service.get_capabilities()
        
        return {
            "ml_enabled": capabilities.ml_enabled,
            "ai_sentiment_enabled": capabilities.ai_sentiment_enabled,
            "real_time_inference": capabilities.real_time_inference,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "status": m.status.value,
                    "status_message": m.status_message,
                    "capabilities": m.capabilities,
                    "architecture": m.architecture,
                    "parameters": m.parameters,
                    "accuracy": m.accuracy,
                    "latency_ms": m.latency_ms,
                    "requires_gpu": m.requires_gpu,
                    "estimated_training_cost": m.estimated_training_cost,
                    "estimated_inference_cost": m.estimated_inference_cost,
                }
                for m in capabilities.models
            ],
            "available_features": capabilities.available_features,
            "planned_features": capabilities.planned_features,
            "limitations": capabilities.limitations,
            "avg_analysis_latency_ms": capabilities.avg_analysis_latency_ms,
            "api_version": capabilities.api_version,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Model status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/honest-summary")
async def get_honest_summary():
    """
    Get brutally honest summary of ML capabilities.
    
    No marketing BS - just the truth about what works and what doesn't.
    """
    try:
        from cift.services.model_status_service import model_status_service
        
        return model_status_service.get_honest_summary()
        
    except Exception as e:
        logger.error(f"Honest summary endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """
    Get detailed information about a specific model.
    """
    try:
        from cift.services.model_status_service import model_status_service
        
        model = model_status_service.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "status": model.status.value,
            "status_message": model.status_message,
            "capabilities": model.capabilities,
            "architecture": model.architecture,
            "parameters": model.parameters,
            "accuracy": model.accuracy,
            "latency_ms": model.latency_ms,
            "memory_mb": model.memory_mb,
            "requires_gpu": model.requires_gpu,
            "min_memory_gb": model.min_memory_gb,
            "estimated_training_cost": model.estimated_training_cost,
            "estimated_inference_cost": model.estimated_inference_cost,
            "last_trained": model.last_trained.isoformat() if model.last_trained else None,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

