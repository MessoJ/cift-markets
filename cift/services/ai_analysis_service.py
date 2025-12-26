"""
CIFT Markets - AI-Powered Analysis Service

Integrates Large Language Models for enhanced stock analysis:
- News sentiment analysis using Google Gemini
- Market narrative synthesis
- Trade thesis generation
- Risk assessment explanations

HONEST DISCLAIMER:
- AI improves analysis by ~5-15%, NOT 100%
- This is a tool to ASSIST decision making, not replace it
- No AI can predict markets with certainty
- Past analysis performance != future results

Cost Efficiency (Gemini 2.0 Flash):
- Input: $0.10/1M tokens (or FREE tier)
- Output: $0.40/1M tokens
- Average analysis: ~500 tokens = practically free
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx
from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================

# Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")  # Latest Gemini 3 model
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Alternative: Use local model via Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1:8b")

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class AISentimentResult:
    """AI-powered sentiment analysis result."""
    
    # Core sentiment
    sentiment_score: float  # -1 to 1 (bearish to bullish)
    sentiment_label: Literal["very_bearish", "bearish", "neutral", "bullish", "very_bullish"]
    confidence: float  # 0-1 confidence in the assessment
    
    # Detailed breakdown
    news_summary: str  # 1-2 sentence summary of news
    key_catalysts: list[str]  # Important events/factors
    risk_factors: list[str]  # Identified risks
    
    # Market narrative
    narrative: str  # The "story" the market is telling
    market_positioning: str  # How institutions might be positioned
    
    # AI reasoning
    reasoning: str  # Why the AI reached this conclusion
    
    # Meta
    model_used: str
    tokens_used: int
    latency_ms: float


@dataclass
class AITradeThesis:
    """AI-generated trade thesis."""
    
    # Direction
    direction: Literal["long", "short", "neutral"]
    conviction: Literal["high", "medium", "low"]
    
    # Thesis
    thesis: str  # 2-3 sentence bull/bear case
    entry_rationale: str  # Why enter now
    
    # Risk management
    key_risk: str  # Biggest risk to the trade
    invalidation: str  # What would invalidate the thesis
    
    # Time horizon
    time_horizon: Literal["day", "swing", "position"]
    
    # AI warning
    disclaimer: str = "This is AI-generated analysis for educational purposes. Not financial advice."


@dataclass 
class AIAnalysisResult:
    """Complete AI analysis result."""
    
    symbol: str
    timestamp: datetime
    
    sentiment: AISentimentResult
    trade_thesis: AITradeThesis | None
    
    # Combined score adjustment
    ai_score_adjustment: float  # -20 to +20 adjustment to base score
    
    # Raw response
    raw_response: str
    
    # Performance
    total_latency_ms: float


# ============================================================================
# AI ANALYSIS SERVICE
# ============================================================================


class AIAnalysisService:
    """
    AI-powered analysis using LLMs.
    
    Supports:
    - Google Gemini 2.0 Flash (default, fast, generous free tier)
    - Local models via Ollama (free, private)
    
    Uses structured prompts for consistent output.
    """
    
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize HTTP client."""
        if self._initialized:
            return
            
        self._client = httpx.AsyncClient(timeout=30.0)
        self._initialized = True
        
        if GEMINI_API_KEY:
            logger.info(f"AI Analysis Service initialized with Google Gemini ({GEMINI_MODEL})")
        elif USE_LOCAL_MODEL:
            logger.info(f"AI Analysis Service initialized with local model ({LOCAL_MODEL})")
        else:
            logger.warning("AI Analysis Service: No API key configured - using rule-based fallback")
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def analyze_sentiment(
        self, 
        symbol: str, 
        news_headlines: list[str],
        current_price: float,
        price_change_pct: float,
    ) -> AISentimentResult:
        """
        Analyze news sentiment using AI.
        
        Args:
            symbol: Stock symbol
            news_headlines: Recent news headlines
            current_price: Current stock price
            price_change_pct: Today's price change %
            
        Returns:
            AISentimentResult with AI-powered sentiment analysis
        """
        start_time = datetime.utcnow()
        
        if not GEMINI_API_KEY and not USE_LOCAL_MODEL:
            return self._fallback_sentiment(news_headlines)
        
        # Prepare news for analysis
        news_text = "\n".join([f"- {h}" for h in news_headlines[:10]])  # Limit to 10 headlines
        
        prompt = f"""Analyze sentiment for {symbol} stock:

{news_text}

Price: ${current_price:.2f} ({price_change_pct:+.2f}%)

Return JSON:
{{
    "sentiment_score": <float -1.0 to 1.0>,
    "sentiment_label": "<very_bearish|bearish|neutral|bullish|very_bullish>",
    "confidence": <float 0.0 to 1.0>,
    "news_summary": "<brief summary>",
    "key_catalysts": ["<catalyst1>", "<catalyst2>"],
    "risk_factors": ["<risk1>", "<risk2>"],
    "narrative": "<one sentence market story>",
    "market_positioning": "<institutional positioning>",
    "reasoning": "<brief reasoning>"
}}"""

        try:
            response, tokens = await self._call_llm(prompt)
            result = json.loads(response)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AISentimentResult(
                sentiment_score=float(result.get("sentiment_score", 0)),
                sentiment_label=result.get("sentiment_label", "neutral"),
                confidence=float(result.get("confidence", 0.5)),
                news_summary=result.get("news_summary", ""),
                key_catalysts=result.get("key_catalysts", []),
                risk_factors=result.get("risk_factors", []),
                narrative=result.get("narrative", ""),
                market_positioning=result.get("market_positioning", ""),
                reasoning=result.get("reasoning", ""),
                model_used=GEMINI_MODEL if GEMINI_API_KEY else LOCAL_MODEL,
                tokens_used=tokens,
                latency_ms=latency,
            )
            
        except Exception as e:
            logger.error(f"AI sentiment analysis failed: {e}")
            return self._fallback_sentiment(news_headlines)
    
    async def generate_trade_thesis(
        self,
        symbol: str,
        technical_summary: str,
        fundamental_summary: str,
        sentiment_summary: str,
        overall_score: float,
    ) -> AITradeThesis:
        """
        Generate AI trade thesis based on analysis.
        """
        if not GEMINI_API_KEY and not USE_LOCAL_MODEL:
            return self._fallback_thesis(overall_score)
        
        prompt = f"""Generate a trade thesis for {symbol} based on this analysis:

TECHNICAL: {technical_summary}
FUNDAMENTAL: {fundamental_summary}  
SENTIMENT: {sentiment_summary}
OVERALL SCORE: {overall_score}/100

Respond in this exact JSON format:
{{
    "direction": "<long|short|neutral>",
    "conviction": "<high|medium|low>",
    "thesis": "<2-3 sentence bull or bear case>",
    "entry_rationale": "<why enter now vs wait>",
    "key_risk": "<single biggest risk>",
    "invalidation": "<what would prove the thesis wrong>",
    "time_horizon": "<day|swing|position>"
}}

Be balanced and acknowledge uncertainty. This is for educational purposes."""

        try:
            response, _ = await self._call_llm(prompt)
            result = json.loads(response)
            
            return AITradeThesis(
                direction=result.get("direction", "neutral"),
                conviction=result.get("conviction", "low"),
                thesis=result.get("thesis", ""),
                entry_rationale=result.get("entry_rationale", ""),
                key_risk=result.get("key_risk", ""),
                invalidation=result.get("invalidation", ""),
                time_horizon=result.get("time_horizon", "swing"),
            )
            
        except Exception as e:
            logger.error(f"AI thesis generation failed: {e}")
            return self._fallback_thesis(overall_score)
    
    async def _call_llm(self, prompt: str) -> tuple[str, int]:
        """
        Call LLM API (Gemini or local).
        
        Returns: (response_text, tokens_used)
        """
        if not self._client:
            await self.initialize()
        
        if GEMINI_API_KEY:
            return await self._call_gemini(prompt)
        elif USE_LOCAL_MODEL:
            return await self._call_ollama(prompt)
        else:
            raise ValueError("No LLM configured")
    
    async def _call_gemini(self, prompt: str) -> tuple[str, int]:
        """Call Google Gemini API."""
        url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are a professional financial analyst. Respond only with valid JSON, no markdown code blocks.\n\n{prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4000,
                "responseMimeType": "application/json"
            }
        }
        
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from Gemini response
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Clean up response - remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Get token count if available
        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
        
        return content, tokens
    
    async def _call_ollama(self, prompt: str) -> tuple[str, int]:
        """Call local Ollama model."""
        payload = {
            "model": LOCAL_MODEL,
            "prompt": f"Respond only with valid JSON.\n\n{prompt}",
            "stream": False,
            "format": "json",
        }
        
        response = await self._client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        content = data.get("response", "{}")
        # Ollama doesn't return token count easily, estimate
        tokens = len(prompt.split()) + len(content.split())
        
        return content, tokens
    
    def _fallback_sentiment(self, headlines: list[str]) -> AISentimentResult:
        """Rule-based fallback when AI is unavailable."""
        positive = ["surge", "jump", "gain", "beat", "upgrade", "buy", "bullish", "strong", "growth", "profit", "soar", "rally"]
        negative = ["drop", "fall", "miss", "downgrade", "sell", "bearish", "weak", "loss", "decline", "cut", "crash", "plunge"]
        
        score = 0.0
        for headline in headlines:
            headline_lower = headline.lower()
            pos = sum(1 for w in positive if w in headline_lower)
            neg = sum(1 for w in negative if w in headline_lower)
            score += (pos - neg)
        
        normalized = max(-1, min(1, score / 10))
        
        if normalized > 0.3:
            label = "bullish" if normalized < 0.6 else "very_bullish"
        elif normalized < -0.3:
            label = "bearish" if normalized > -0.6 else "very_bearish"
        else:
            label = "neutral"
        
        return AISentimentResult(
            sentiment_score=normalized,
            sentiment_label=label,
            confidence=0.5,  # Lower confidence for rule-based
            news_summary="Rule-based analysis (AI unavailable)",
            key_catalysts=[],
            risk_factors=[],
            narrative="",
            market_positioning="",
            reasoning="Simple keyword matching",
            model_used="rule-based",
            tokens_used=0,
            latency_ms=0,
        )
    
    def _fallback_thesis(self, score: float) -> AITradeThesis:
        """Rule-based thesis when AI unavailable."""
        if score >= 65:
            direction = "long"
            conviction = "medium" if score < 75 else "high"
        elif score <= 35:
            direction = "short"
            conviction = "medium" if score > 25 else "high"
        else:
            direction = "neutral"
            conviction = "low"
        
        return AITradeThesis(
            direction=direction,
            conviction=conviction,
            thesis="Based on quantitative factor analysis.",
            entry_rationale="Score-based signal.",
            key_risk="Market regime change.",
            invalidation="Score reversal below threshold.",
            time_horizon="swing",
        )


# Global instance
ai_analysis_service = AIAnalysisService()
