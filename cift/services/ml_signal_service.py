"""
CIFT Markets - ML Signal & Alert Service

Integrates ML models with alerts system to provide:
- Automatic buy/sell alerts based on ML predictions
- Portfolio recommendations with AI reasoning
- Continuous paper trading simulation
- Sharpe ratio tracking

This is the brain that connects:
- Order Flow Transformer (direction prediction)
- Technical Analysis (indicators)
- Gemini AI (reasoning & explanations)
- Alerts System (notifications)
- Trading Engine (execution)
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from loguru import logger

from cift.core.database import get_postgres_pool


class SignalType(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SignalSource(str, Enum):
    ML_ORDERFLOW = "ml_orderflow"       # Order Flow Transformer
    TECHNICAL = "technical"              # Technical indicators
    FUNDAMENTAL = "fundamental"          # Fundamental analysis
    AI_GEMINI = "ai_gemini"             # Gemini AI insights
    COMBINED = "combined"                # Ensemble of all sources


@dataclass
class TradingSignal:
    """A trading signal from ML/analysis."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    source: SignalSource
    
    # Price targets
    entry_price: float
    target_price: float | None = None
    stop_loss: float | None = None
    
    # Reasoning
    reasons: list[str] = field(default_factory=list)
    ai_explanation: str = ""
    
    # Risk/Reward
    risk_reward_ratio: float = 0.0
    expected_return_pct: float = 0.0
    max_loss_pct: float = 0.0
    
    # Timing
    hold_duration: str = "short"  # short (<1 day), medium (1-7 days), long (7+ days)
    urgency: str = "normal"  # low, normal, high, critical
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signal_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PortfolioRecommendation:
    """Detailed recommendation for a portfolio position."""
    symbol: str
    current_price: float
    avg_cost: float
    quantity: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Recommendation
    action: SignalType
    confidence: float
    
    # Targets
    target_price: float | None = None
    stop_loss: float | None = None
    
    # AI Analysis
    technical_score: float = 50.0
    fundamental_score: float = 50.0
    sentiment_score: float = 50.0
    ml_prediction: str = "neutral"
    
    # Reasoning (detailed explanation)
    summary: str = ""
    bullish_factors: list[str] = field(default_factory=list)
    bearish_factors: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    
    # Action items
    should_add: bool = False
    should_trim: bool = False
    should_hold: bool = True
    should_exit: bool = False
    
    # If adding, suggested size
    suggested_add_pct: float = 0.0
    suggested_trim_pct: float = 0.0


@dataclass
class PaperTrade:
    """A simulated paper trade."""
    trade_id: str
    symbol: str
    side: str  # buy, sell
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    signal_id: str = ""
    status: str = "open"  # open, closed


class MLSignalService:
    """
    Central service for ML-powered trading signals.
    
    Responsibilities:
    1. Run ML predictions on watchlist/portfolio symbols
    2. Generate trading signals with confidence levels
    3. Create alerts for high-confidence signals
    4. Track paper trades and calculate Sharpe ratio
    5. Provide portfolio recommendations with AI reasoning
    """
    
    def __init__(self):
        self.running = False
        self.check_interval = 300  # Check every 5 minutes
        self.signals: dict[str, TradingSignal] = {}  # symbol -> latest signal
        self.paper_trades: list[PaperTrade] = []
        self.daily_returns: list[float] = []  # For Sharpe calculation
        
        # Thresholds
        self.alert_confidence_threshold = 0.65  # Alert if confidence > 65%
        self.strong_signal_threshold = 0.75    # Strong buy/sell if confidence > 75%
        
    async def start_monitoring(self):
        """Start the ML signal monitoring loop."""
        if self.running:
            return
            
        self.running = True
        logger.info("🤖 Starting ML Signal Service...")
        
        # Load historical paper trades
        await self._load_paper_trades()
        
        while self.running:
            try:
                await self._run_prediction_cycle()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"ML Signal Service error: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False
        logger.info("⏹️ Stopping ML Signal Service...")
    
    async def _run_prediction_cycle(self):
        """Run one cycle of predictions on all tracked symbols."""
        symbols = await self._get_tracked_symbols()
        
        if not symbols:
            logger.debug("No symbols to track")
            return
        
        logger.info(f"Running ML predictions on {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                signal = await self.generate_signal(symbol)
                if signal:
                    self.signals[symbol] = signal
                    
                    # Check if we should create an alert
                    if signal.confidence >= self.alert_confidence_threshold:
                        await self._create_ml_alert(signal)
                    
                    # Check if we should simulate a paper trade
                    if signal.confidence >= self.strong_signal_threshold:
                        await self._handle_paper_trade(signal)
                        
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        # Update daily returns for Sharpe calculation
        await self._update_daily_returns()
    
    async def generate_signal(self, symbol: str) -> TradingSignal | None:
        """
        Generate a trading signal for a symbol using all available models.
        
        Combines:
        1. Order Flow Transformer prediction
        2. Technical analysis
        3. AI (Gemini) reasoning
        """
        try:
            # Get current price data
            price_data = await self._get_price_data(symbol)
            if not price_data or len(price_data) < 100:
                return None
            
            current_price = price_data[-1]['close']
            
            # 1. Get ML prediction from Order Flow Transformer
            ml_result = await self._get_orderflow_prediction(price_data)
            
            # 2. Get technical analysis
            tech_result = await self._get_technical_analysis(symbol)
            
            # 3. Combine signals
            combined_direction, combined_confidence = self._combine_signals(
                ml_result, tech_result
            )
            
            # 4. Calculate targets
            target_price, stop_loss = self._calculate_targets(
                current_price, combined_direction, price_data
            )
            
            # 5. Get AI explanation (async, can fail gracefully)
            ai_explanation = await self._get_ai_explanation(
                symbol, combined_direction, combined_confidence, tech_result
            )
            
            # Build signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=combined_direction,
                confidence=combined_confidence,
                source=SignalSource.COMBINED,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasons=self._build_reasons(ml_result, tech_result),
                ai_explanation=ai_explanation,
                risk_reward_ratio=self._calculate_rr(current_price, target_price, stop_loss),
                expected_return_pct=((target_price or current_price) - current_price) / current_price * 100 if target_price else 0,
                max_loss_pct=abs((stop_loss or current_price) - current_price) / current_price * 100 if stop_loss else 0,
                hold_duration=self._estimate_hold_duration(combined_direction, combined_confidence),
                urgency=self._determine_urgency(combined_confidence),
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def get_portfolio_recommendations(
        self, user_id: UUID
    ) -> list[PortfolioRecommendation]:
        """
        Get detailed recommendations for all positions in a user's portfolio.
        """
        recommendations = []
        
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                # Get user's positions
                positions = await conn.fetch("""
                    SELECT 
                        p.symbol,
                        p.quantity,
                        p.avg_cost,
                        p.market_value,
                        p.unrealized_pnl,
                        p.unrealized_pnl_pct,
                        p.current_price
                    FROM positions p
                    WHERE p.user_id = $1 AND p.quantity > 0
                    ORDER BY p.market_value DESC
                """, user_id)
                
                for pos in positions:
                    rec = await self._generate_position_recommendation(pos)
                    recommendations.append(rec)
                    
        except Exception as e:
            logger.error(f"Error getting portfolio recommendations: {e}")
        
        return recommendations
    
    async def _generate_position_recommendation(self, position: dict) -> PortfolioRecommendation:
        """Generate detailed recommendation for a single position."""
        symbol = position['symbol']
        current_price = float(position.get('current_price') or position.get('avg_cost') or 0)
        avg_cost = float(position.get('avg_cost') or 0)
        
        # Get or generate signal
        signal = self.signals.get(symbol)
        if not signal:
            signal = await self.generate_signal(symbol)
        
        # Get technical analysis
        tech = await self._get_technical_analysis(symbol)
        
        # Build recommendation
        rec = PortfolioRecommendation(
            symbol=symbol,
            current_price=current_price,
            avg_cost=avg_cost,
            quantity=float(position.get('quantity') or 0),
            market_value=float(position.get('market_value') or 0),
            unrealized_pnl=float(position.get('unrealized_pnl') or 0),
            unrealized_pnl_pct=float(position.get('unrealized_pnl_pct') or 0),
            action=signal.signal_type if signal else SignalType.HOLD,
            confidence=signal.confidence if signal else 0.5,
            target_price=signal.target_price if signal else None,
            stop_loss=signal.stop_loss if signal else None,
            technical_score=tech.get('score', 50) if tech else 50,
            fundamental_score=50,  # Would need fundamental data
            sentiment_score=50,    # Would need news sentiment
            ml_prediction=signal.signal_type.value if signal else "neutral",
            summary=signal.ai_explanation if signal else "",
            bullish_factors=signal.reasons[:3] if signal and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else [],
            bearish_factors=signal.reasons[:3] if signal and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] else [],
            key_risks=["Market volatility", "Sector rotation risk"],
        )
        
        # Determine action items
        pnl_pct = rec.unrealized_pnl_pct
        if signal:
            if signal.signal_type == SignalType.STRONG_BUY and signal.confidence > 0.7:
                rec.should_add = True
                rec.suggested_add_pct = min(30, (signal.confidence - 0.5) * 100)
            elif signal.signal_type == SignalType.STRONG_SELL and signal.confidence > 0.7:
                rec.should_exit = True
                rec.should_hold = False
            elif signal.signal_type == SignalType.SELL:
                rec.should_trim = True
                rec.suggested_trim_pct = min(50, signal.confidence * 50)
            
            # Profit taking
            if pnl_pct > 20 and signal.signal_type in [SignalType.HOLD, SignalType.SELL]:
                rec.should_trim = True
                rec.suggested_trim_pct = 25  # Take 25% profit
            
            # Cut losses
            if pnl_pct < -10 and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                rec.should_exit = True
                rec.should_hold = False
        
        return rec
    
    async def _get_orderflow_prediction(self, price_data: list[dict]) -> dict:
        """Get prediction from Order Flow Transformer."""
        try:
            from cift.ml.order_flow_predictor import OrderFlowPredictor
            
            predictor = OrderFlowPredictor()
            if predictor.model is None:
                return {"direction": "neutral", "confidence": 0.33}
            
            prices = np.array([d['close'] for d in price_data[-100:]])
            volumes = np.array([d['volume'] for d in price_data[-100:]])
            
            result = predictor.predict_from_ohlcv(prices, volumes)
            
            return {
                "direction": result.direction,
                "confidence": result.confidence,
                "probs": result.direction_probs,
            }
        except Exception as e:
            logger.warning(f"OrderFlow prediction failed: {e}")
            return {"direction": "neutral", "confidence": 0.33}
    
    async def _get_technical_analysis(self, symbol: str) -> dict | None:
        """Get technical analysis for a symbol."""
        try:
            from cift.services.stock_analyzer import stock_analyzer
            
            analysis = await stock_analyzer.analyze_stock(symbol)
            if analysis:
                return {
                    "score": analysis.overall_score,
                    "signal": analysis.rating,
                    "technical": analysis.technical,
                    "momentum": analysis.momentum,
                    "bullish_factors": analysis.bullish_factors,
                    "bearish_factors": analysis.bearish_factors,
                }
        except Exception as e:
            logger.warning(f"Technical analysis failed for {symbol}: {e}")
        return None
    
    async def _get_ai_explanation(
        self, symbol: str, direction: SignalType, confidence: float, tech: dict | None
    ) -> str:
        """Get AI-powered explanation for the signal."""
        try:
            from cift.services.ai_analysis_service import ai_analysis_service
            
            context = f"""
            Symbol: {symbol}
            Signal: {direction.value}
            Confidence: {confidence:.1%}
            Technical Score: {tech.get('score', 'N/A') if tech else 'N/A'}
            """
            
            if tech:
                bullish = tech.get('bullish_factors', [])
                bearish = tech.get('bearish_factors', [])
                if bullish:
                    context += f"\nBullish Factors: {', '.join(bullish[:3])}"
                if bearish:
                    context += f"\nBearish Factors: {', '.join(bearish[:3])}"
            
            prompt = f"""
            Given this trading signal for {symbol}:
            {context}
            
            Provide a brief 2-3 sentence explanation of why this signal makes sense,
            what the trader should watch for, and any key risks to be aware of.
            Be concise and actionable.
            """
            
            result = await ai_analysis_service.analyze_with_llm(prompt, symbol)
            return result.analysis if result else ""
            
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
            return ""
    
    def _combine_signals(self, ml_result: dict, tech_result: dict | None) -> tuple[SignalType, float]:
        """Combine ML and technical signals into one."""
        ml_dir = ml_result.get('direction', 'neutral')
        ml_conf = ml_result.get('confidence', 0.33)
        
        tech_score = tech_result.get('score', 50) if tech_result else 50
        tech_signal = tech_result.get('signal', 'HOLD') if tech_result else 'HOLD'
        
        # Convert to numeric
        ml_score = {'up': 75, 'neutral': 50, 'down': 25}.get(ml_dir, 50)
        tech_score_normalized = tech_score  # Already 0-100
        
        # Weighted average (60% ML, 40% Technical)
        combined_score = ml_score * 0.6 + tech_score_normalized * 0.4
        combined_confidence = (ml_conf * 0.6 + (abs(tech_score - 50) / 50) * 0.4)
        
        # Map to signal type
        if combined_score >= 70:
            if combined_confidence >= 0.75:
                return SignalType.STRONG_BUY, combined_confidence
            return SignalType.BUY, combined_confidence
        elif combined_score >= 55:
            return SignalType.BUY, combined_confidence * 0.8
        elif combined_score <= 30:
            if combined_confidence >= 0.75:
                return SignalType.STRONG_SELL, combined_confidence
            return SignalType.SELL, combined_confidence
        elif combined_score <= 45:
            return SignalType.SELL, combined_confidence * 0.8
        else:
            return SignalType.HOLD, combined_confidence * 0.5
    
    def _calculate_targets(
        self, current_price: float, direction: SignalType, price_data: list[dict]
    ) -> tuple[float | None, float | None]:
        """Calculate price targets and stop loss."""
        # Calculate ATR for volatility-based targets
        highs = [d['high'] for d in price_data[-14:]]
        lows = [d['low'] for d in price_data[-14:]]
        closes = [d['close'] for d in price_data[-14:]]
        
        atr = self._calculate_atr(highs, lows, closes)
        
        if direction in [SignalType.BUY, SignalType.STRONG_BUY]:
            target = current_price + (atr * 2.5)  # 2.5 ATR target
            stop = current_price - (atr * 1.5)    # 1.5 ATR stop
            return target, stop
        elif direction in [SignalType.SELL, SignalType.STRONG_SELL]:
            target = current_price - (atr * 2.5)
            stop = current_price + (atr * 1.5)
            return target, stop
        else:
            return None, None
    
    def _calculate_atr(self, highs: list, lows: list, closes: list) -> float:
        """Calculate Average True Range."""
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        return np.mean(tr_list) if tr_list else 0.0
    
    def _calculate_rr(self, entry: float, target: float | None, stop: float | None) -> float:
        """Calculate risk/reward ratio."""
        if not target or not stop:
            return 0.0
        reward = abs(target - entry)
        risk = abs(entry - stop)
        return reward / risk if risk > 0 else 0.0
    
    def _build_reasons(self, ml_result: dict, tech_result: dict | None) -> list[str]:
        """Build list of reasons for the signal."""
        reasons = []
        
        ml_dir = ml_result.get('direction', 'neutral')
        ml_conf = ml_result.get('confidence', 0)
        
        if ml_conf > 0.5:
            reasons.append(f"ML model predicts {ml_dir} with {ml_conf:.0%} confidence")
        
        if tech_result:
            tech_score = tech_result.get('score', 50)
            if tech_score >= 70:
                reasons.append(f"Strong technical score: {tech_score}/100")
            elif tech_score <= 30:
                reasons.append(f"Weak technical score: {tech_score}/100")
            
            bullish = tech_result.get('bullish_factors', [])
            bearish = tech_result.get('bearish_factors', [])
            reasons.extend(bullish[:2])
            reasons.extend(bearish[:2])
        
        return reasons[:5]  # Max 5 reasons
    
    def _estimate_hold_duration(self, direction: SignalType, confidence: float) -> str:
        """Estimate how long to hold the position."""
        if confidence >= 0.8:
            return "short"  # High confidence = quick scalp
        elif confidence >= 0.6:
            return "medium"  # Medium confidence = swing trade
        else:
            return "long"  # Lower confidence = longer hold for mean reversion
    
    def _determine_urgency(self, confidence: float) -> str:
        """Determine signal urgency."""
        if confidence >= 0.85:
            return "critical"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "normal"
        else:
            return "low"
    
    async def _create_ml_alert(self, signal: TradingSignal):
        """Create an alert for an ML signal."""
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                # Get users watching this symbol
                watchers = await conn.fetch("""
                    SELECT DISTINCT user_id FROM (
                        SELECT user_id FROM watchlist_items 
                        WHERE symbol = $1
                        UNION
                        SELECT user_id FROM positions 
                        WHERE symbol = $1 AND quantity > 0
                    ) AS combined
                """, signal.symbol)
                
                action_text = {
                    SignalType.STRONG_BUY: "🚀 STRONG BUY",
                    SignalType.BUY: "📈 BUY",
                    SignalType.HOLD: "⏸️ HOLD",
                    SignalType.SELL: "📉 SELL",
                    SignalType.STRONG_SELL: "🔻 STRONG SELL",
                }.get(signal.signal_type, "SIGNAL")
                
                for watcher in watchers:
                    await conn.execute("""
                        INSERT INTO alerts (
                            id, user_id, alert_type, symbol, 
                            severity, message, metadata, 
                            created_at, is_read
                        ) VALUES (
                            $1, $2, 'ml_signal', $3,
                            $4, $5, $6,
                            NOW(), false
                        )
                    """,
                        uuid4(),
                        watcher['user_id'],
                        signal.symbol,
                        'high' if signal.confidence >= 0.75 else 'medium',
                        f"{action_text}: {signal.symbol} @ ${signal.entry_price:.2f} ({signal.confidence:.0%} confidence)",
                        json.dumps({
                            "signal_type": signal.signal_type.value,
                            "confidence": signal.confidence,
                            "target_price": signal.target_price,
                            "stop_loss": signal.stop_loss,
                            "reasons": signal.reasons[:3],
                            "ai_explanation": signal.ai_explanation[:200] if signal.ai_explanation else "",
                            "risk_reward": signal.risk_reward_ratio,
                            "urgency": signal.urgency,
                        })
                    )
                
                logger.info(f"Created ML alerts for {signal.symbol} to {len(watchers)} users")
                
        except Exception as e:
            logger.error(f"Error creating ML alert: {e}")
    
    async def _handle_paper_trade(self, signal: TradingSignal):
        """Handle paper trading for high-confidence signals."""
        try:
            # Check if we already have an open trade for this symbol
            existing = [t for t in self.paper_trades if t.symbol == signal.symbol and t.status == "open"]
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if not existing:
                    # Open new paper long
                    trade = PaperTrade(
                        trade_id=str(uuid4()),
                        symbol=signal.symbol,
                        side="buy",
                        quantity=1000 / signal.entry_price,  # $1000 per trade
                        entry_price=signal.entry_price,
                        entry_time=datetime.utcnow(),
                        signal_id=signal.signal_id,
                    )
                    self.paper_trades.append(trade)
                    await self._save_paper_trade(trade)
                    logger.info(f"📝 Paper BUY: {signal.symbol} @ ${signal.entry_price:.2f}")
                    
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # Close any open longs
                for trade in existing:
                    if trade.side == "buy":
                        trade.exit_price = signal.entry_price
                        trade.exit_time = datetime.utcnow()
                        trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                        trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
                        trade.status = "closed"
                        await self._save_paper_trade(trade)
                        logger.info(f"📝 Paper SELL: {signal.symbol} @ ${signal.entry_price:.2f} (PnL: ${trade.pnl:.2f})")
                        
        except Exception as e:
            logger.error(f"Paper trade error: {e}")
    
    async def _get_tracked_symbols(self) -> list[str]:
        """Get all symbols to track (from watchlists and positions)."""
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT symbol FROM (
                        SELECT symbol FROM watchlist_items
                        UNION
                        SELECT symbol FROM positions WHERE quantity > 0
                        UNION
                        SELECT symbol FROM market_data WHERE symbol IN (
                            'BTCUSDT', 'ETHUSDT', 'AAPL', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'MSFT'
                        )
                    ) AS all_symbols
                    LIMIT 50
                """)
                return [row['symbol'] for row in rows]
        except Exception as e:
            logger.error(f"Error getting tracked symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'AAPL', 'TSLA', 'SPY']  # Defaults
    
    async def _get_price_data(self, symbol: str) -> list[dict] | None:
        """Get recent price data for a symbol."""
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = $1
                    ORDER BY timestamp DESC
                    LIMIT 200
                """, symbol)
                
                if not rows:
                    return None
                
                return [
                    {
                        'timestamp': row['timestamp'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                    }
                    for row in reversed(rows)
                ]
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            return None
    
    async def _load_paper_trades(self):
        """Load paper trades from database."""
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM paper_trades
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    ORDER BY created_at DESC
                """)
                
                for row in rows:
                    trade = PaperTrade(
                        trade_id=str(row['id']),
                        symbol=row['symbol'],
                        side=row['side'],
                        quantity=float(row['quantity']),
                        entry_price=float(row['entry_price']),
                        entry_time=row['entry_time'],
                        exit_price=float(row['exit_price']) if row['exit_price'] else None,
                        exit_time=row['exit_time'],
                        pnl=float(row['pnl']) if row['pnl'] else 0,
                        pnl_pct=float(row['pnl_pct']) if row['pnl_pct'] else 0,
                        signal_id=str(row.get('signal_id', '')),
                        status=row['status'],
                    )
                    self.paper_trades.append(trade)
                    
        except Exception as e:
            logger.warning(f"Could not load paper trades: {e}")
    
    async def _save_paper_trade(self, trade: PaperTrade):
        """Save paper trade to database."""
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO paper_trades (
                        id, symbol, side, quantity, entry_price, entry_time,
                        exit_price, exit_time, pnl, pnl_pct, signal_id, status,
                        created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW()
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        exit_price = EXCLUDED.exit_price,
                        exit_time = EXCLUDED.exit_time,
                        pnl = EXCLUDED.pnl,
                        pnl_pct = EXCLUDED.pnl_pct,
                        status = EXCLUDED.status
                """,
                    UUID(trade.trade_id),
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.entry_price,
                    trade.entry_time,
                    trade.exit_price,
                    trade.exit_time,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.signal_id,
                    trade.status,
                )
        except Exception as e:
            logger.error(f"Error saving paper trade: {e}")
    
    async def _update_daily_returns(self):
        """Update daily returns for Sharpe calculation."""
        closed_trades = [t for t in self.paper_trades if t.status == "closed"]
        if not closed_trades:
            return
        
        # Group by day
        daily_pnl: dict[str, float] = {}
        for trade in closed_trades:
            if trade.exit_time:
                day = trade.exit_time.strftime("%Y-%m-%d")
                daily_pnl[day] = daily_pnl.get(day, 0) + trade.pnl
        
        # Convert to returns (assuming $10000 portfolio)
        portfolio_value = 10000
        self.daily_returns = [pnl / portfolio_value for pnl in daily_pnl.values()]
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(self.daily_returns) < 5:
            return 0.0
        
        returns = np.array(self.daily_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)
    
    def get_paper_trading_stats(self) -> dict:
        """Get paper trading statistics."""
        closed_trades = [t for t in self.paper_trades if t.status == "closed"]
        open_trades = [t for t in self.paper_trades if t.status == "open"]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        winners = [t for t in closed_trades if t.pnl > 0]
        losers = [t for t in closed_trades if t.pnl < 0]
        
        return {
            "total_trades": len(closed_trades),
            "open_trades": len(open_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(closed_trades) * 100 if closed_trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed_trades) if closed_trades else 0,
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "best_trade": max(t.pnl for t in closed_trades) if closed_trades else 0,
            "worst_trade": min(t.pnl for t in closed_trades) if closed_trades else 0,
        }


# Singleton instance
ml_signal_service = MLSignalService()


def get_ml_signal_service() -> MLSignalService:
    """Get the ML signal service singleton."""
    return ml_signal_service
