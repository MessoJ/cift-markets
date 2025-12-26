"""
CIFT Markets - World-Class Stock Analysis Engine

Evidence-based multi-factor analysis with real-time scoring.
Combines technical, fundamental, sentiment, and risk analysis.

Performance Target: <200ms end-to-end analysis

Based on academic research:
- Fama-French Factor Models
- Carhart 4-Factor Model
- AQR Quality Minus Junk
- De Prado's Financial Machine Learning

Honest Approach:
- We don't claim to "beat the market"
- We provide data-driven insights to help decision making
- Transparent scoring methodology
- Clear confidence intervals
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from cift.core.database import get_postgres_pool


class Rating(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(str, Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


@dataclass
class TechnicalAnalysis:
    """Technical analysis scores and signals."""
    
    # Overall score (0-100)
    score: float
    signal: SignalStrength
    
    # Individual indicators
    rsi_14: float | None = None
    rsi_signal: str | None = None  # "oversold", "neutral", "overbought"
    
    macd_line: float | None = None
    macd_signal_line: float | None = None
    macd_histogram: float | None = None
    macd_crossover: str | None = None  # "bullish", "bearish", "none"
    
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    price_vs_sma: dict | None = None  # {"sma_20": "above", "sma_50": "above", ...}
    
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_position: str | None = None  # "above_upper", "upper_half", "lower_half", "below_lower"
    
    atr_14: float | None = None
    atr_percent: float | None = None  # ATR as % of price
    
    volume_trend: str | None = None  # "increasing", "decreasing", "stable"
    volume_vs_avg: float | None = None  # Current volume / 20-day avg
    
    # Support/Resistance
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    
    # Trend
    short_term_trend: str | None = None  # "up", "down", "sideways"
    medium_term_trend: str | None = None
    long_term_trend: str | None = None


@dataclass
class FundamentalAnalysis:
    """Fundamental analysis scores."""
    
    # Overall score (0-100)
    score: float
    signal: SignalStrength
    
    # Valuation (lower is better for value)
    pe_ratio: float | None = None
    pe_percentile: float | None = None  # vs sector/market
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda: float | None = None
    peg_ratio: float | None = None
    
    # Profitability (Quality)
    roe: float | None = None
    roa: float | None = None
    profit_margin: float | None = None
    operating_margin: float | None = None
    
    # Growth
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    revenue_growth_3y: float | None = None
    
    # Financial Health
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    interest_coverage: float | None = None
    
    # Dividend
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    dividend_growth_5y: float | None = None
    
    # Earnings Quality
    earnings_surprise_avg: float | None = None  # Avg beat/miss %
    earnings_consistency: float | None = None  # % of quarters beating


@dataclass
class SentimentAnalysis:
    """Sentiment analysis from news and social."""
    
    # Overall score (0-100)
    score: float
    signal: SignalStrength
    
    # News sentiment
    news_sentiment: float | None = None  # -1 to 1
    news_volume: int | None = None  # Articles in last 24h
    news_trend: str | None = None  # "improving", "deteriorating", "stable"
    
    # Analyst ratings
    analyst_rating: str | None = None  # "buy", "hold", "sell"
    analyst_target: float | None = None  # Avg price target
    analyst_target_upside: float | None = None  # % upside to target
    analyst_count: int | None = None
    
    # Institutional
    institutional_ownership: float | None = None  # % owned by institutions
    institutional_change: float | None = None  # Change in ownership
    
    # Insider activity
    insider_net_shares: int | None = None  # Net shares bought/sold (90 days)
    insider_sentiment: str | None = None  # "buying", "selling", "neutral"


@dataclass
class RiskAnalysis:
    """Risk metrics and analysis."""
    
    # Overall score (0-100, higher = riskier)
    score: float
    risk_level: str  # "low", "medium", "high", "very_high"
    
    # Volatility
    volatility_30d: float | None = None  # Annualized
    volatility_percentile: float | None = None  # vs history
    
    # Beta
    beta: float | None = None  # vs S&P 500
    
    # Drawdown
    max_drawdown_1y: float | None = None
    current_drawdown: float | None = None  # From 52-week high
    
    # Value at Risk
    var_95: float | None = None  # 95% daily VaR
    
    # Liquidity
    avg_volume_20d: float | None = None
    bid_ask_spread: float | None = None
    
    # Correlation
    correlation_spy: float | None = None


@dataclass
class MomentumAnalysis:
    """Momentum factor analysis."""
    
    score: float
    signal: SignalStrength
    
    # Returns
    return_1d: float | None = None
    return_1w: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_12m: float | None = None
    return_ytd: float | None = None
    
    # Momentum score (12-1 month, academic standard)
    momentum_12_1: float | None = None
    momentum_percentile: float | None = None  # vs universe


@dataclass
class StockAnalysis:
    """Complete stock analysis output."""
    
    symbol: str
    timestamp: datetime
    
    # Current price info
    price: float
    change: float
    change_percent: float
    
    # Component analyses
    technical: TechnicalAnalysis
    fundamental: FundamentalAnalysis
    sentiment: SentimentAnalysis
    risk: RiskAnalysis
    momentum: MomentumAnalysis
    
    # Combined rating
    overall_score: float  # 0-100
    rating: Rating
    confidence: float  # 0-1
    
    # Key insights (human readable)
    bullish_factors: list[str] = field(default_factory=list)
    bearish_factors: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    
    # Trade suggestion (not financial advice)
    suggested_action: str | None = None
    entry_zone: tuple[float, float] | None = None  # (low, high)
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    risk_reward: float | None = None
    
    # Timing
    analysis_latency_ms: float = 0


class StockAnalyzer:
    """
    High-performance stock analysis engine.
    
    Combines multiple analysis methodologies:
    1. Technical Analysis - Price patterns, indicators
    2. Fundamental Analysis - Valuation, quality, growth
    3. Sentiment Analysis - News, analysts, insiders
    4. Risk Analysis - Volatility, drawdown, VaR
    5. Momentum Analysis - Factor-based momentum
    
    Scoring methodology is transparent and evidence-based.
    """
    
    def __init__(self):
        self.finnhub = None
        self._initialized = False
        
        # Factor weights (based on academic research)
        self.factor_weights = {
            "technical": 0.20,
            "fundamental": 0.25,
            "sentiment": 0.15,
            "risk": 0.15,  # Inverse - low risk = good
            "momentum": 0.25,
        }
    
    async def initialize(self):
        """Initialize data connections."""
        if self._initialized:
            return
        
        from cift.services.market_data_service import market_data_service
        self.market_data = market_data_service
        self.finnhub = market_data_service.finnhub
        
        await self.market_data.initialize()
        self._initialized = True
        logger.info("Stock Analyzer initialized")
    
    async def analyze(self, symbol: str) -> StockAnalysis:
        """
        Perform comprehensive stock analysis.
        
        Target latency: <200ms
        """
        start_time = datetime.utcnow()
        
        if not self._initialized:
            await self.initialize()
        
        symbol = symbol.upper()
        
        # Fetch all data concurrently
        quote_task = self._get_quote(symbol)
        technicals_task = self._analyze_technical(symbol)
        fundamentals_task = self._analyze_fundamental(symbol)
        sentiment_task = self._analyze_sentiment(symbol)
        momentum_task = self._analyze_momentum(symbol)
        
        # Wait for all analyses
        quote, technical, fundamental, sentiment, momentum = await asyncio.gather(
            quote_task,
            technicals_task,
            fundamentals_task,
            sentiment_task,
            momentum_task,
            return_exceptions=True
        )
        
        # Handle any failures gracefully
        if isinstance(quote, Exception):
            logger.error(f"Quote fetch failed for {symbol}: {quote}")
            quote = {"price": 0, "change": 0, "change_percent": 0}
        
        if isinstance(technical, Exception):
            logger.error(f"Technical analysis failed for {symbol}: {technical}")
            technical = self._default_technical()
        
        if isinstance(fundamental, Exception):
            logger.error(f"Fundamental analysis failed for {symbol}: {fundamental}")
            fundamental = self._default_fundamental()
        
        if isinstance(sentiment, Exception):
            logger.error(f"Sentiment analysis failed for {symbol}: {sentiment}")
            sentiment = self._default_sentiment()
        
        if isinstance(momentum, Exception):
            logger.error(f"Momentum analysis failed for {symbol}: {momentum}")
            momentum = self._default_momentum()
        
        # Calculate risk score
        risk = await self._analyze_risk(symbol, quote, technical, momentum)
        
        # Calculate overall score and rating
        overall_score, rating, confidence = self._calculate_overall_rating(
            technical, fundamental, sentiment, risk, momentum
        )
        
        # Generate insights
        bullish_factors, bearish_factors, key_risks = self._generate_insights(
            technical, fundamental, sentiment, risk, momentum
        )
        
        # Generate trade suggestion
        trade_suggestion = self._generate_trade_suggestion(
            quote, technical, fundamental, overall_score, risk
        )
        
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return StockAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=quote.get("price", 0),
            change=quote.get("change", 0),
            change_percent=quote.get("change_percent", 0),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            risk=risk,
            momentum=momentum,
            overall_score=overall_score,
            rating=rating,
            confidence=confidence,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            key_risks=key_risks,
            suggested_action=trade_suggestion.get("action"),
            entry_zone=trade_suggestion.get("entry_zone"),
            stop_loss=trade_suggestion.get("stop_loss"),
            target_1=trade_suggestion.get("target_1"),
            target_2=trade_suggestion.get("target_2"),
            risk_reward=trade_suggestion.get("risk_reward"),
            analysis_latency_ms=latency,
        )
    
    async def _get_quote(self, symbol: str) -> dict:
        """Get current quote."""
        try:
            quotes = await self.market_data.get_quotes_batch([symbol])
            return quotes.get(symbol, {})
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
    
    async def _analyze_technical(self, symbol: str) -> TechnicalAnalysis:
        """
        Technical analysis using price/volume data.
        """
        try:
            # Get historical bars
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                bars = await conn.fetch(
                    """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_bars
                    WHERE symbol = $1 AND timeframe = '1d'
                    ORDER BY timestamp DESC
                    LIMIT 200
                    """,
                    symbol
                )
            
            if not bars or len(bars) < 20:
                return self._default_technical()
            
            # Convert to arrays (most recent first)
            closes = np.array([float(b["close"]) for b in bars])
            highs = np.array([float(b["high"]) for b in bars])
            lows = np.array([float(b["low"]) for b in bars])
            volumes = np.array([float(b["volume"]) for b in bars])
            
            # Reverse for calculations (oldest first)
            closes = closes[::-1]
            highs = highs[::-1]
            lows = lows[::-1]
            volumes = volumes[::-1]
            
            current_price = closes[-1]
            
            # Calculate indicators
            rsi_14 = self._calculate_rsi(closes, 14)
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
            sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else None
            
            # MACD
            ema_12 = self._calculate_ema(closes, 12)
            ema_26 = self._calculate_ema(closes, 26)
            macd_line = ema_12 - ema_26
            macd_signal = self._calculate_ema_from_array(
                np.array([self._calculate_ema(closes[:i+1], 12) - self._calculate_ema(closes[:i+1], 26) 
                          for i in range(len(closes)-1, max(len(closes)-10, 25), -1)][::-1]), 
                9
            ) if len(closes) >= 35 else 0
            macd_hist = macd_line - macd_signal
            
            # Bollinger Bands
            bb_sma = sma_20
            bb_std = np.std(closes[-20:])
            bb_upper = bb_sma + 2 * bb_std
            bb_lower = bb_sma - 2 * bb_std
            
            # ATR
            atr_14 = self._calculate_atr(highs, lows, closes, 14)
            
            # Volume analysis
            vol_20_avg = np.mean(volumes[-20:])
            current_vol = volumes[-1]
            vol_ratio = current_vol / vol_20_avg if vol_20_avg > 0 else 1.0
            
            # Determine signals
            rsi_signal = "oversold" if rsi_14 < 30 else "overbought" if rsi_14 > 70 else "neutral"
            
            macd_crossover = "none"
            if len(closes) > 26:
                prev_macd = self._calculate_ema(closes[:-1], 12) - self._calculate_ema(closes[:-1], 26)
                if macd_line > 0 and prev_macd <= 0:
                    macd_crossover = "bullish"
                elif macd_line < 0 and prev_macd >= 0:
                    macd_crossover = "bearish"
            
            # Price vs SMAs
            price_vs_sma = {}
            if sma_20:
                price_vs_sma["sma_20"] = "above" if current_price > sma_20 else "below"
            if sma_50:
                price_vs_sma["sma_50"] = "above" if current_price > sma_50 else "below"
            if sma_200:
                price_vs_sma["sma_200"] = "above" if current_price > sma_200 else "below"
            
            # Bollinger position
            if current_price > bb_upper:
                bb_position = "above_upper"
            elif current_price > bb_sma:
                bb_position = "upper_half"
            elif current_price > bb_lower:
                bb_position = "lower_half"
            else:
                bb_position = "below_lower"
            
            # Trend analysis
            short_trend = "up" if current_price > sma_20 else "down"
            medium_trend = "up" if sma_50 and current_price > sma_50 else "down" if sma_50 else None
            long_trend = "up" if sma_200 and current_price > sma_200 else "down" if sma_200 else None
            
            # Volume trend
            vol_trend = "increasing" if vol_ratio > 1.2 else "decreasing" if vol_ratio < 0.8 else "stable"
            
            # Calculate technical score (0-100)
            score = self._calculate_technical_score(
                rsi_14, macd_line, macd_crossover, price_vs_sma, bb_position, vol_ratio
            )
            
            signal = self._score_to_signal(score)
            
            # Calculate support/resistance (simple method)
            support_levels = self._find_support_resistance(lows, current_price, is_support=True)
            resistance_levels = self._find_support_resistance(highs, current_price, is_support=False)
            
            return TechnicalAnalysis(
                score=score,
                signal=signal,
                rsi_14=rsi_14,
                rsi_signal=rsi_signal,
                macd_line=macd_line,
                macd_signal_line=macd_signal,
                macd_histogram=macd_hist,
                macd_crossover=macd_crossover,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                price_vs_sma=price_vs_sma,
                bollinger_upper=bb_upper,
                bollinger_lower=bb_lower,
                bollinger_position=bb_position,
                atr_14=atr_14,
                atr_percent=(atr_14 / current_price * 100) if current_price > 0 else None,
                volume_trend=vol_trend,
                volume_vs_avg=vol_ratio,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                short_term_trend=short_trend,
                medium_term_trend=medium_trend,
                long_term_trend=long_trend,
            )
            
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return self._default_technical()
    
    async def _analyze_fundamental(self, symbol: str) -> FundamentalAnalysis:
        """
        Fundamental analysis using Finnhub data.
        """
        try:
            if not self.finnhub:
                return self._default_fundamental()
            
            # Fetch financials from Finnhub
            financials = await self.finnhub.get_financials(symbol)
            
            if not financials or "metric" not in financials:
                return self._default_fundamental()
            
            metrics = financials.get("metric", {})
            
            # Extract metrics
            pe_ratio = metrics.get("peBasicExclExtraTTM")
            pb_ratio = metrics.get("pbAnnual")
            ps_ratio = metrics.get("psAnnual")
            
            roe = metrics.get("roeTTM")
            roa = metrics.get("roaTTM")
            profit_margin = metrics.get("netProfitMarginTTM")
            
            revenue_growth = metrics.get("revenueGrowthTTMYoy")
            eps_growth = metrics.get("epsGrowthTTMYoy")
            
            debt_equity = metrics.get("totalDebt/totalEquityAnnual")
            current_ratio = metrics.get("currentRatioAnnual")
            
            dividend_yield = metrics.get("dividendYieldIndicatedAnnual")
            payout_ratio = metrics.get("payoutRatioAnnual")
            
            # Calculate fundamental score
            score = self._calculate_fundamental_score(
                pe_ratio, pb_ratio, roe, profit_margin, 
                revenue_growth, debt_equity, current_ratio
            )
            
            signal = self._score_to_signal(score)
            
            return FundamentalAnalysis(
                score=score,
                signal=signal,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                ps_ratio=ps_ratio,
                roe=roe,
                roa=roa,
                profit_margin=profit_margin,
                revenue_growth_yoy=revenue_growth,
                earnings_growth_yoy=eps_growth,
                debt_to_equity=debt_equity,
                current_ratio=current_ratio,
                dividend_yield=dividend_yield,
                payout_ratio=payout_ratio,
            )
            
        except Exception as e:
            logger.error(f"Fundamental analysis error for {symbol}: {e}")
            return self._default_fundamental()
    
    async def _analyze_sentiment(self, symbol: str) -> SentimentAnalysis:
        """
        Sentiment analysis from news and analysts.
        
        Now with AI-powered analysis using Google Gemini!
        Falls back to rule-based if API unavailable.
        """
        try:
            if not self.finnhub:
                return self._default_sentiment()
            
            # Get company news
            from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
            to_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            news = await self.finnhub.get_company_news(symbol, from_date, to_date)
            
            if not news:
                return self._default_sentiment()
            
            news_volume = len(news)
            headlines = [article.get("headline", "") for article in news[:20]]
            
            # Try AI-powered sentiment analysis first
            try:
                from cift.services.ai_analysis_service import ai_analysis_service
                
                # Get current price for context
                quote = await self._get_quote(symbol)
                price = quote.get("price", 0)
                change_pct = quote.get("change_percent", 0)
                
                ai_result = await ai_analysis_service.analyze_sentiment(
                    symbol=symbol,
                    news_headlines=headlines,
                    current_price=price,
                    price_change_pct=change_pct,
                )
                
                # Use AI sentiment if successful
                if ai_result.model_used != "rule-based":
                    logger.info(f"AI sentiment for {symbol}: {ai_result.sentiment_label} ({ai_result.model_used})")
                    
                    return SentimentAnalysis(
                        score=50 + (ai_result.sentiment_score * 30),  # Map to 0-100
                        signal=self._score_to_signal(50 + (ai_result.sentiment_score * 30)),
                        news_sentiment=ai_result.sentiment_score,
                        news_volume=news_volume,
                        news_trend="stable",
                        # AI extras stored in optional fields
                    )
                    
            except ImportError:
                logger.debug("AI analysis service not available")
            except Exception as ai_err:
                logger.warning(f"AI sentiment failed, using fallback: {ai_err}")
            
            # Fallback: Simple keyword-based sentiment
            news_sentiment = 0.0
            positive_words = ["surge", "jump", "gain", "beat", "upgrade", "buy", "bullish", "strong", "growth", "profit", "soar", "rally"]
            negative_words = ["drop", "fall", "miss", "downgrade", "sell", "bearish", "weak", "loss", "decline", "cut", "crash", "plunge"]
            
            for headline in headlines:
                headline_lower = headline.lower()
                pos_count = sum(1 for word in positive_words if word in headline_lower)
                neg_count = sum(1 for word in negative_words if word in headline_lower)
                news_sentiment += (pos_count - neg_count)
            
            news_sentiment = np.clip(news_sentiment / 10, -1, 1)
            
            # Calculate sentiment score (0-100)
            score = 50 + (news_sentiment * 30)
            signal = self._score_to_signal(score)
            
            return SentimentAnalysis(
                score=score,
                signal=signal,
                news_sentiment=news_sentiment,
                news_volume=news_volume,
                news_trend="stable",
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return self._default_sentiment()
    
    async def _analyze_momentum(self, symbol: str) -> MomentumAnalysis:
        """
        Momentum factor analysis.
        """
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                bars = await conn.fetch(
                    """
                    SELECT timestamp, close
                    FROM ohlcv_bars
                    WHERE symbol = $1 AND timeframe = '1d'
                    ORDER BY timestamp DESC
                    LIMIT 260
                    """,
                    symbol
                )
            
            if not bars or len(bars) < 5:
                return self._default_momentum()
            
            closes = [float(b["close"]) for b in bars]
            current = closes[0]
            
            # Calculate returns
            def calc_return(days):
                if len(closes) > days:
                    return (current - closes[days]) / closes[days] * 100
                return None
            
            return_1d = calc_return(1)
            return_1w = calc_return(5)
            return_1m = calc_return(21)
            return_3m = calc_return(63)
            return_6m = calc_return(126)
            return_12m = calc_return(252)
            
            # 12-1 month momentum (academic standard: skip last month)
            if len(closes) > 252:
                price_12m_ago = closes[252]
                price_1m_ago = closes[21]
                momentum_12_1 = (price_1m_ago - price_12m_ago) / price_12m_ago * 100
            else:
                momentum_12_1 = None
            
            # Calculate momentum score
            score = 50.0
            if return_1m is not None:
                score += return_1m * 2  # Weight recent momentum
            if return_3m is not None:
                score += return_3m * 1
            if momentum_12_1 is not None:
                score += momentum_12_1 * 0.5
            
            score = np.clip(score, 0, 100)
            signal = self._score_to_signal(score)
            
            return MomentumAnalysis(
                score=score,
                signal=signal,
                return_1d=return_1d,
                return_1w=return_1w,
                return_1m=return_1m,
                return_3m=return_3m,
                return_6m=return_6m,
                return_12m=return_12m,
                momentum_12_1=momentum_12_1,
            )
            
        except Exception as e:
            logger.error(f"Momentum analysis error for {symbol}: {e}")
            return self._default_momentum()
    
    async def _analyze_risk(
        self, symbol: str, quote: dict, technical: TechnicalAnalysis, momentum: MomentumAnalysis
    ) -> RiskAnalysis:
        """
        Risk analysis.
        """
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                bars = await conn.fetch(
                    """
                    SELECT timestamp, close, high
                    FROM ohlcv_bars
                    WHERE symbol = $1 AND timeframe = '1d'
                    ORDER BY timestamp DESC
                    LIMIT 252
                    """,
                    symbol
                )
            
            if not bars or len(bars) < 30:
                return self._default_risk()
            
            closes = np.array([float(b["close"]) for b in bars])
            highs = np.array([float(b["high"]) for b in bars])
            
            # Calculate volatility (30-day annualized)
            returns = np.diff(closes) / closes[:-1]
            volatility_30d = np.std(returns[:30]) * np.sqrt(252) * 100
            
            # Max drawdown (1 year)
            max_drawdown_1y = 0.0
            peak = closes[0]
            for close in closes:
                if close > peak:
                    peak = close
                dd = (peak - close) / peak * 100
                if dd > max_drawdown_1y:
                    max_drawdown_1y = dd
            
            # Current drawdown from 52-week high
            high_52w = max(highs[:252]) if len(highs) >= 252 else max(highs)
            current_price = closes[0]
            current_drawdown = (high_52w - current_price) / high_52w * 100
            
            # VaR 95% (parametric)
            var_95 = np.percentile(returns[:30] * -100, 95)
            
            # Risk score (higher = riskier)
            risk_score = 0.0
            risk_score += min(volatility_30d * 2, 40)  # Volatility contributes up to 40
            risk_score += min(max_drawdown_1y, 30)  # Drawdown contributes up to 30
            risk_score += min(current_drawdown, 30)  # Current drawdown up to 30
            
            risk_score = min(risk_score, 100)
            
            risk_level = "low"
            if risk_score > 70:
                risk_level = "very_high"
            elif risk_score > 50:
                risk_level = "high"
            elif risk_score > 30:
                risk_level = "medium"
            
            return RiskAnalysis(
                score=risk_score,
                risk_level=risk_level,
                volatility_30d=volatility_30d,
                max_drawdown_1y=max_drawdown_1y,
                current_drawdown=current_drawdown,
                var_95=var_95,
            )
            
        except Exception as e:
            logger.error(f"Risk analysis error for {symbol}: {e}")
            return self._default_risk()
    
    def _calculate_overall_rating(
        self,
        technical: TechnicalAnalysis,
        fundamental: FundamentalAnalysis,
        sentiment: SentimentAnalysis,
        risk: RiskAnalysis,
        momentum: MomentumAnalysis,
    ) -> tuple[float, Rating, float]:
        """
        Calculate overall score and rating.
        
        Uses evidence-based factor weighting.
        """
        # Weighted score (risk is inverted - lower risk = better)
        risk_inverted = 100 - risk.score
        
        overall_score = (
            technical.score * self.factor_weights["technical"] +
            fundamental.score * self.factor_weights["fundamental"] +
            sentiment.score * self.factor_weights["sentiment"] +
            risk_inverted * self.factor_weights["risk"] +
            momentum.score * self.factor_weights["momentum"]
        )
        
        # Determine rating
        if overall_score >= 75:
            rating = Rating.STRONG_BUY
        elif overall_score >= 60:
            rating = Rating.BUY
        elif overall_score >= 40:
            rating = Rating.HOLD
        elif overall_score >= 25:
            rating = Rating.SELL
        else:
            rating = Rating.STRONG_SELL
        
        # Confidence based on data availability and agreement
        signals = [technical.signal, fundamental.signal, sentiment.signal, momentum.signal]
        bullish_count = sum(1 for s in signals if s in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH])
        bearish_count = sum(1 for s in signals if s in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH])
        
        agreement = max(bullish_count, bearish_count) / len(signals)
        confidence = 0.5 + (agreement * 0.5)  # 0.5 to 1.0
        
        return overall_score, rating, confidence
    
    def _generate_insights(
        self,
        technical: TechnicalAnalysis,
        fundamental: FundamentalAnalysis,
        sentiment: SentimentAnalysis,
        risk: RiskAnalysis,
        momentum: MomentumAnalysis,
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate human-readable insights."""
        
        bullish = []
        bearish = []
        risks = []
        
        # Technical insights
        if technical.rsi_signal == "oversold":
            bullish.append("RSI indicates oversold conditions (potential reversal)")
        elif technical.rsi_signal == "overbought":
            bearish.append("RSI indicates overbought conditions (potential pullback)")
        
        if technical.macd_crossover == "bullish":
            bullish.append("MACD bullish crossover detected")
        elif technical.macd_crossover == "bearish":
            bearish.append("MACD bearish crossover detected")
        
        if technical.price_vs_sma:
            if technical.price_vs_sma.get("sma_200") == "above":
                bullish.append("Price above 200-day moving average (long-term uptrend)")
            else:
                bearish.append("Price below 200-day moving average (long-term downtrend)")
        
        # Fundamental insights
        if fundamental.pe_ratio and fundamental.pe_ratio < 15:
            bullish.append(f"Attractive valuation (P/E: {fundamental.pe_ratio:.1f})")
        elif fundamental.pe_ratio and fundamental.pe_ratio > 30:
            bearish.append(f"Expensive valuation (P/E: {fundamental.pe_ratio:.1f})")
        
        if fundamental.roe and fundamental.roe > 15:
            bullish.append(f"Strong profitability (ROE: {fundamental.roe:.1f}%)")
        
        if fundamental.revenue_growth_yoy and fundamental.revenue_growth_yoy > 20:
            bullish.append(f"Strong revenue growth ({fundamental.revenue_growth_yoy:.1f}% YoY)")
        
        # Momentum insights
        if momentum.return_1m and momentum.return_1m > 10:
            bullish.append(f"Strong short-term momentum (+{momentum.return_1m:.1f}% in 1 month)")
        elif momentum.return_1m and momentum.return_1m < -10:
            bearish.append(f"Weak short-term momentum ({momentum.return_1m:.1f}% in 1 month)")
        
        # Risk insights
        if risk.volatility_30d and risk.volatility_30d > 50:
            risks.append(f"High volatility ({risk.volatility_30d:.1f}% annualized)")
        
        if risk.current_drawdown and risk.current_drawdown > 20:
            risks.append(f"Significant drawdown from high (-{risk.current_drawdown:.1f}%)")
        
        if risk.max_drawdown_1y and risk.max_drawdown_1y > 30:
            risks.append(f"Large historical drawdown (-{risk.max_drawdown_1y:.1f}% in past year)")
        
        return bullish, bearish, risks
    
    def _generate_trade_suggestion(
        self,
        quote: dict,
        technical: TechnicalAnalysis,
        fundamental: FundamentalAnalysis,
        overall_score: float,
        risk: RiskAnalysis,
    ) -> dict:
        """
        Generate trade suggestion based on analysis.
        
        DISCLAIMER: This is not financial advice.
        """
        price = quote.get("price", 0)
        if price <= 0:
            return {}
        
        atr = technical.atr_14 or (price * 0.02)  # Default 2% ATR
        
        result = {}
        
        if overall_score >= 60:
            result["action"] = "Consider buying on pullbacks"
            result["entry_zone"] = (
                round(price * 0.98, 2),  # 2% below current
                round(price * 1.01, 2),  # 1% above current
            )
            result["stop_loss"] = round(price - (2 * atr), 2)
            result["target_1"] = round(price + (2 * atr), 2)
            result["target_2"] = round(price + (4 * atr), 2)
            
        elif overall_score <= 40:
            result["action"] = "Consider reducing position or avoiding"
            result["entry_zone"] = None
            result["stop_loss"] = round(price * 0.95, 2)  # 5% stop if holding
            result["target_1"] = None
            result["target_2"] = None
            
        else:
            result["action"] = "Hold - Wait for clearer signals"
            result["entry_zone"] = None
            result["stop_loss"] = round(price - (2 * atr), 2) if price > 0 else None
            result["target_1"] = None
            result["target_2"] = None
        
        # Calculate risk/reward if applicable
        if result.get("entry_zone") and result.get("stop_loss") and result.get("target_1"):
            entry = (result["entry_zone"][0] + result["entry_zone"][1]) / 2
            risk_amount = entry - result["stop_loss"]
            reward_amount = result["target_1"] - entry
            if risk_amount > 0:
                result["risk_reward"] = round(reward_amount / risk_amount, 2)
        
        return result
    
    # =========================================================================
    # CALCULATION HELPERS
    # =========================================================================
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA (most recent value)."""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _calculate_ema_from_array(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA from array."""
        return self._calculate_ema(prices, period)
    
    def _calculate_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        """Calculate ATR."""
        if len(highs) < 2:
            return 0.0
        
        tr = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr.append(max(hl, hc, lc))
        
        return np.mean(tr[-period:])
    
    def _calculate_technical_score(
        self,
        rsi: float,
        macd: float,
        macd_crossover: str,
        price_vs_sma: dict,
        bb_position: str,
        vol_ratio: float,
    ) -> float:
        """Calculate technical score (0-100)."""
        score = 50.0  # Start neutral
        
        # RSI (oversold = bullish, overbought = bearish)
        if rsi < 30:
            score += 15
        elif rsi < 40:
            score += 5
        elif rsi > 70:
            score -= 15
        elif rsi > 60:
            score -= 5
        
        # MACD
        if macd > 0:
            score += 10
        else:
            score -= 10
        
        if macd_crossover == "bullish":
            score += 10
        elif macd_crossover == "bearish":
            score -= 10
        
        # Price vs SMAs
        above_count = sum(1 for v in price_vs_sma.values() if v == "above")
        below_count = sum(1 for v in price_vs_sma.values() if v == "below")
        score += (above_count - below_count) * 5
        
        # Bollinger
        if bb_position == "below_lower":
            score += 10  # Oversold
        elif bb_position == "above_upper":
            score -= 10  # Overbought
        
        # Volume confirmation
        if vol_ratio > 1.5:
            score += 5  # Strong volume
        
        return np.clip(score, 0, 100)
    
    def _calculate_fundamental_score(
        self,
        pe: float | None,
        pb: float | None,
        roe: float | None,
        margin: float | None,
        growth: float | None,
        debt_equity: float | None,
        current_ratio: float | None,
    ) -> float:
        """Calculate fundamental score (0-100)."""
        score = 50.0
        
        # Valuation (lower = better)
        if pe is not None:
            if pe < 10:
                score += 15
            elif pe < 15:
                score += 10
            elif pe < 20:
                score += 5
            elif pe > 40:
                score -= 10
            elif pe > 30:
                score -= 5
        
        if pb is not None:
            if pb < 1:
                score += 10
            elif pb < 2:
                score += 5
            elif pb > 5:
                score -= 5
        
        # Quality (higher = better)
        if roe is not None:
            if roe > 20:
                score += 10
            elif roe > 15:
                score += 5
            elif roe < 5:
                score -= 10
        
        if margin is not None:
            if margin > 20:
                score += 5
            elif margin < 5:
                score -= 5
        
        # Growth
        if growth is not None:
            if growth > 20:
                score += 10
            elif growth > 10:
                score += 5
            elif growth < 0:
                score -= 10
        
        # Financial health
        if debt_equity is not None:
            if debt_equity < 0.5:
                score += 5
            elif debt_equity > 2:
                score -= 10
        
        if current_ratio is not None:
            if current_ratio > 2:
                score += 5
            elif current_ratio < 1:
                score -= 10
        
        return np.clip(score, 0, 100)
    
    def _score_to_signal(self, score: float) -> SignalStrength:
        """Convert score to signal strength."""
        if score >= 75:
            return SignalStrength.VERY_BULLISH
        elif score >= 60:
            return SignalStrength.BULLISH
        elif score >= 40:
            return SignalStrength.NEUTRAL
        elif score >= 25:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.VERY_BEARISH
    
    def _find_support_resistance(
        self, prices: np.ndarray, current_price: float, is_support: bool, num_levels: int = 3
    ) -> list[float]:
        """Find support/resistance levels using local extrema."""
        levels = []
        window = 5
        
        for i in range(window, len(prices) - window):
            if is_support:
                if prices[i] == min(prices[i-window:i+window+1]):
                    if prices[i] < current_price:
                        levels.append(float(prices[i]))
            else:
                if prices[i] == max(prices[i-window:i+window+1]):
                    if prices[i] > current_price:
                        levels.append(float(prices[i]))
        
        # Return closest levels
        levels = sorted(set(levels), reverse=not is_support)
        return levels[:num_levels]
    
    # =========================================================================
    # DEFAULT VALUES
    # =========================================================================
    
    def _default_technical(self) -> TechnicalAnalysis:
        return TechnicalAnalysis(score=50.0, signal=SignalStrength.NEUTRAL)
    
    def _default_fundamental(self) -> FundamentalAnalysis:
        return FundamentalAnalysis(score=50.0, signal=SignalStrength.NEUTRAL)
    
    def _default_sentiment(self) -> SentimentAnalysis:
        return SentimentAnalysis(score=50.0, signal=SignalStrength.NEUTRAL)
    
    def _default_momentum(self) -> MomentumAnalysis:
        return MomentumAnalysis(score=50.0, signal=SignalStrength.NEUTRAL)
    
    def _default_risk(self) -> RiskAnalysis:
        return RiskAnalysis(score=50.0, risk_level="medium")


# Global instance
stock_analyzer = StockAnalyzer()
