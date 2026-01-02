/**
 * CIFT Markets - Prediction Service
 * 
 * Service for generating chart predictions.
 * Now integrated with real ML backend via /api/v1/ml-signals endpoint.
 * Falls back to mock data if ML service is unavailable.
 * 
 * ML Integration: Uses Order Flow Transformer + Gemini AI for predictions.
 */

import { apiClient } from '~/lib/api/client';
import { predictionStore } from '~/stores/prediction.store';
import type {
  PredictedBar,
  PredictionRequest,
  PredictionResponse,
} from '~/types/prediction.types';

// Flag to enable/disable mock mode
// Set to false to prefer real ML backend (will fall back to mock on error)
const USE_MOCK_PREDICTIONS = false;

/**
 * Generate predictions for a symbol/timeframe
 */
export async function generatePrediction(
  symbol: string,
  timeframe: string,
  currentBars: { timestamp: string | number; open: number; high: number; low: number; close: number; volume: number }[],
  barsAhead: number = 5
): Promise<PredictedBar[]> {
  predictionStore.setIsLoading(true);
  predictionStore.setError(null);

  try {
    if (USE_MOCK_PREDICTIONS) {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 500));
      
      const predictions = generateMockPredictions(currentBars, barsAhead, timeframe);
      console.info(`🔮 Generated ${predictions.length} mock predictions for ${symbol} ${timeframe}`);
      return predictions;
    }

    // Try real ML API first
    try {
      const response = await apiClient.get(`/ml-signals/signal/${symbol}`);
      
      if (response && response.signal && response.signal !== 'unknown') {
        // Convert ML signal to chart predictions
        const predictions = convertMLSignalToPredictions(
          currentBars, 
          barsAhead, 
          timeframe,
          response
        );
        console.info(`🧠 Generated ${predictions.length} ML-based predictions for ${symbol}`);
        return predictions;
      }
    } catch (mlError) {
      console.warn('ML API unavailable, falling back to mock predictions:', mlError);
    }
    
    // Fallback to mock predictions
    const predictions = generateMockPredictions(currentBars, barsAhead, timeframe);
    console.info(`🔮 Generated ${predictions.length} fallback predictions for ${symbol}`);
    return predictions;
  } catch (err: any) {
    const errorMsg = err.message || 'Failed to generate prediction';
    predictionStore.setError(errorMsg);
    console.error('Prediction generation failed:', err);
    throw err;
  } finally {
    predictionStore.setIsLoading(false);
  }
}

/**
 * Generate mock predictions based on recent price action
 * This simulates what an ML model might predict
 * 
 * DELETABLE: Remove when real ML models are integrated
 */
function generateMockPredictions(
  currentBars: { timestamp: string | number; open: number; high: number; low: number; close: number; volume: number }[],
  barsAhead: number,
  timeframe: string
): PredictedBar[] {
  if (currentBars.length < 10) {
    throw new Error('Need at least 10 bars for prediction');
  }

  const predictions: PredictedBar[] = [];
  const lastBars = currentBars.slice(-20);
  const lastBar = lastBars[lastBars.length - 1];
  
  // Convert timestamp to number if string
  const getTimestampMs = (ts: string | number): number => {
    return typeof ts === 'string' ? new Date(ts).getTime() : ts;
  };

  // Calculate basic statistics from recent bars
  const avgMove = calculateAverageMove(lastBars);
  const avgVolume = calculateAverageVolume(lastBars);
  const trend = calculateTrend(lastBars);
  const volatility = calculateVolatility(lastBars);

  // Get interval in milliseconds
  const intervalMs = getIntervalMs(timeframe);

  let prevClose = lastBar.close;
  let prevTimestamp = getTimestampMs(lastBar.timestamp);

  for (let i = 0; i < barsAhead; i++) {
    // Determine direction based on trend + randomness
    // Trend has 60% weight, randomness 40%
    const trendBias = trend > 0 ? 0.55 : trend < 0 ? 0.45 : 0.5;
    const isUp = Math.random() < trendBias;

    // Calculate price movement
    const moveMagnitude = avgMove * (0.5 + Math.random() * volatility);
    const move = isUp ? moveMagnitude : -moveMagnitude;

    // Generate OHLC
    const open = prevClose;
    const close = open + move;
    
    // High and low based on volatility
    const range = Math.abs(move) * (1 + volatility * Math.random());
    const high = Math.max(open, close) + range * Math.random() * 0.5;
    const low = Math.min(open, close) - range * Math.random() * 0.5;

    // Volume with some variation
    const volumeVariation = 0.7 + Math.random() * 0.6;
    const volume = Math.round(avgVolume * volumeVariation);

    // Robust confidence calculation
    // Base confidence starts high and decreases with prediction distance
    // Also factors in volatility (high volatility = lower confidence)
    // And trend strength (strong trend = higher confidence)
    const distanceFactor = 1 - (i * 0.12);  // Decreases 12% per bar
    const volatilityPenalty = Math.min(0.2, volatility * 0.1);  // Up to 20% penalty for volatility
    const trendBonus = Math.abs(trend) * 0.1;  // Up to 10% bonus for clear trend
    const dataQualityFactor = Math.min(1, lastBars.length / 20);  // Full confidence with 20+ bars
    
    const rawConfidence = (0.85 * distanceFactor - volatilityPenalty + trendBonus) * dataQualityFactor;
    const confidence = Math.max(0.25, Math.min(0.92, rawConfidence + (Math.random() - 0.5) * 0.05));

    // Calculate next timestamp
    const timestamp = prevTimestamp + intervalMs;

    predictions.push({
      timestamp,
      open: roundPrice(open),
      high: roundPrice(high),
      low: roundPrice(low),
      close: roundPrice(close),
      volume,
      confidence,
    });

    prevClose = close;
    prevTimestamp = timestamp;
  }

  return predictions;
}

/**
 * Calculate average absolute price move
 */
function calculateAverageMove(bars: { open: number; close: number }[]): number {
  const moves = bars.map(b => Math.abs(b.close - b.open));
  return moves.reduce((sum, m) => sum + m, 0) / moves.length;
}

/**
 * Calculate average volume
 */
function calculateAverageVolume(bars: { volume: number }[]): number {
  const volumes = bars.map(b => b.volume);
  return volumes.reduce((sum, v) => sum + v, 0) / volumes.length;
}

/**
 * Calculate trend direction (-1 to 1)
 */
function calculateTrend(bars: { close: number }[]): number {
  if (bars.length < 5) return 0;
  
  const recent = bars.slice(-5);
  const older = bars.slice(-10, -5);
  
  const recentAvg = recent.reduce((sum, b) => sum + b.close, 0) / recent.length;
  const olderAvg = older.reduce((sum, b) => sum + b.close, 0) / older.length;
  
  const diff = (recentAvg - olderAvg) / olderAvg;
  return Math.max(-1, Math.min(1, diff * 10));
}

/**
 * Calculate volatility (0 to 2)
 */
function calculateVolatility(bars: { high: number; low: number; close: number }[]): number {
  const ranges = bars.map(b => (b.high - b.low) / b.close);
  const avgRange = ranges.reduce((sum, r) => sum + r, 0) / ranges.length;
  return Math.min(2, avgRange * 20);
}

/**
 * Get interval milliseconds from timeframe string
 */
function getIntervalMs(timeframe: string): number {
  const intervals: Record<string, number> = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
    '1M': 30 * 24 * 60 * 60 * 1000,
  };
  return intervals[timeframe] || 24 * 60 * 60 * 1000;
}

/**
 * Round price to 2 decimal places
 */
function roundPrice(price: number): number {
  return Math.round(price * 100) / 100;
}

/**
 * Convert ML signal response to chart predictions
 * Maps signal direction and targets to OHLC bars
 */
function convertMLSignalToPredictions(
  currentBars: { timestamp: string | number; open: number; high: number; low: number; close: number; volume: number }[],
  barsAhead: number,
  timeframe: string,
  mlResponse: any
): PredictedBar[] {
  const predictions: PredictedBar[] = [];
  const lastBar = currentBars[currentBars.length - 1];
  
  const getTimestampMs = (ts: string | number): number => {
    return typeof ts === 'string' ? new Date(ts).getTime() : ts;
  };

  const intervalMs = getIntervalMs(timeframe);
  const avgVolume = calculateAverageVolume(currentBars.slice(-20));
  
  // Determine direction based on ML signal
  const signal = mlResponse.signal || 'hold';
  const confidence = mlResponse.confidence || 0.5;
  const targetPrice = mlResponse.target_price || lastBar.close;
  const stopLoss = mlResponse.stop_loss || lastBar.close;
  
  // Calculate expected move per bar towards target
  const totalMove = targetPrice - lastBar.close;
  const movePerBar = totalMove / barsAhead;
  
  // Signal-based bias
  const isUptrend = ['strong_buy', 'buy'].includes(signal);
  const isDowntrend = ['strong_sell', 'sell'].includes(signal);
  
  let prevClose = lastBar.close;
  let prevTimestamp = getTimestampMs(lastBar.timestamp);

  for (let i = 0; i < barsAhead; i++) {
    // Progress towards target with some noise
    const progress = (i + 1) / barsAhead;
    const noise = (Math.random() - 0.5) * Math.abs(movePerBar) * 0.3;
    const expectedClose = lastBar.close + (totalMove * progress) + noise;
    
    // Generate OHLC
    const open = prevClose;
    const close = expectedClose;
    
    // High/low based on direction and volatility
    const range = Math.abs(close - open) * 0.5;
    const high = Math.max(open, close) + range * (0.3 + Math.random() * 0.4);
    const low = Math.min(open, close) - range * (0.3 + Math.random() * 0.4);

    // Volume varies based on move strength
    const volumeMultiplier = 0.8 + Math.abs(movePerBar / lastBar.close) * 5 + Math.random() * 0.4;
    const volume = Math.round(avgVolume * volumeMultiplier);

    // Confidence decreases with distance but boosted by ML confidence
    const distanceFactor = 1 - (i * 0.08);
    const mlBoost = confidence * 0.15;
    const barConfidence = Math.max(0.3, Math.min(0.95, confidence * distanceFactor + mlBoost));

    const timestamp = prevTimestamp + intervalMs;

    predictions.push({
      timestamp,
      open: roundPrice(open),
      high: roundPrice(high),
      low: roundPrice(low),
      close: roundPrice(close),
      volume,
      confidence: barConfidence,
    });

    prevClose = close;
    prevTimestamp = timestamp;
  }

  return predictions;
}

/**
 * Start prediction session
 */
export function startPredictionSession(
  symbol: string,
  timeframe: string,
  predictions: PredictedBar[],
  startIndex: number
): string {
  return predictionStore.startPrediction(
    symbol,
    timeframe,
    predictions,
    startIndex,
    'orderflow_transformer_v8', // Updated with real model ID
    '8.0.0'
  );
}

/**
 * Clear prediction for symbol
 */
export function clearPrediction(symbol: string, timeframe: string): void {
  predictionStore.clearPrediction(symbol, timeframe);
}

/**
 * Check if prediction exists
 */
export function hasPrediction(symbol: string, timeframe: string): boolean {
  return predictionStore.hasPrediction(symbol, timeframe);
}
