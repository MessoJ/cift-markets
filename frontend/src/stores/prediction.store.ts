/**
 * CIFT Markets - Prediction Store
 * 
 * Centralized state management for chart predictions.
 * Uses SolidJS signals for reactive state.
 */

import { createSignal, createRoot } from 'solid-js';
import type {
  PredictionSession,
  PredictedBar,
  ActualBar,
  PredictionAccuracy,
  PredictionHistoryEntry,
} from '~/types/prediction.types';

function createPredictionStore() {
  // Current active prediction session per symbol+timeframe
  const [activeSessions, setActiveSessions] = createSignal<Map<string, PredictionSession>>(new Map());
  
  // Historical predictions for comparison
  const [history, setHistory] = createSignal<PredictionHistoryEntry[]>([]);
  
  // Loading state
  const [isLoading, setIsLoading] = createSignal(false);
  
  // Error state
  const [error, setError] = createSignal<string | null>(null);

  /**
   * Get session key for symbol+timeframe combination
   */
  const getSessionKey = (symbol: string, timeframe: string): string => {
    return `${symbol}:${timeframe}`;
  };

  /**
   * Get active prediction session for a symbol/timeframe
   */
  const getActiveSession = (symbol: string, timeframe: string): PredictionSession | undefined => {
    return activeSessions().get(getSessionKey(symbol, timeframe));
  };

  /**
   * Check if prediction is active for symbol/timeframe
   */
  const hasPrediction = (symbol: string, timeframe: string): boolean => {
    const session = getActiveSession(symbol, timeframe);
    return session?.status === 'active';
  };

  /**
   * Start a new prediction session
   */
  const startPrediction = (
    symbol: string,
    timeframe: string,
    predictions: PredictedBar[],
    startIndex: number,
    modelId: string,
    modelVersion: string
  ): string => {
    const sessionId = `pred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const session: PredictionSession = {
      id: sessionId,
      symbol,
      timeframe,
      startedAt: Date.now(),
      predictionStartIndex: startIndex,
      predictedBars: predictions,
      actualBars: [],
      status: 'active',
      modelId,
      modelVersion,
    };

    setActiveSessions(prev => {
      const newMap = new Map(prev);
      newMap.set(getSessionKey(symbol, timeframe), session);
      return newMap;
    });

    console.info(`🔮 Prediction started: ${symbol} ${timeframe} - ${predictions.length} bars ahead`);
    return sessionId;
  };

  /**
   * Add actual bar data for comparison
   */
  const addActualBar = (symbol: string, timeframe: string, bar: ActualBar): void => {
    const key = getSessionKey(symbol, timeframe);
    const session = activeSessions().get(key);
    
    if (!session || session.status !== 'active') return;

    // Check if we've reached the predicted period
    const predictedTimestamps = session.predictedBars.map(b => b.timestamp);
    
    if (predictedTimestamps.includes(bar.timestamp)) {
      setActiveSessions(prev => {
        const newMap = new Map(prev);
        const updatedSession = { ...session };
        updatedSession.actualBars = [...updatedSession.actualBars, bar];
        
        // If all predictions have actual data, calculate accuracy
        if (updatedSession.actualBars.length >= updatedSession.predictedBars.length) {
          updatedSession.accuracy = calculateAccuracy(
            updatedSession.predictedBars,
            updatedSession.actualBars
          );
          updatedSession.status = 'completed';
          
          // Add to history
          addToHistory(updatedSession);
          
          console.info(`✅ Prediction completed: ${symbol} ${timeframe} - Score: ${updatedSession.accuracy.overallScore.toFixed(1)}`);
        }
        
        newMap.set(key, updatedSession);
        return newMap;
      });
    }
  };

  /**
   * Calculate prediction accuracy metrics
   */
  const calculateAccuracy = (
    predicted: PredictedBar[],
    actual: ActualBar[]
  ): PredictionAccuracy => {
    if (predicted.length === 0 || actual.length === 0) {
      return {
        directional: 0,
        priceRMSE: Infinity,
        highLowAccuracy: 0,
        volumeAccuracy: 0,
        overallScore: 0,
        barsCompared: 0,
      };
    }

    const barsCompared = Math.min(predicted.length, actual.length);
    let correctDirections = 0;
    let priceSquaredErrors = 0;
    let withinRange = 0;
    let volumeAccuracySum = 0;

    for (let i = 0; i < barsCompared; i++) {
      const pred = predicted[i];
      const act = actual[i];

      // Directional accuracy (did we predict up/down correctly?)
      const predictedDirection = pred.close >= pred.open ? 1 : -1;
      const actualDirection = act.close >= act.open ? 1 : -1;
      if (predictedDirection === actualDirection) {
        correctDirections++;
      }

      // Price RMSE (close price)
      priceSquaredErrors += Math.pow(pred.close - act.close, 2);

      // High-Low range accuracy (was actual within predicted range?)
      if (act.high <= pred.high && act.low >= pred.low) {
        withinRange++;
      }

      // Volume accuracy (within 50% is considered accurate)
      const volumeRatio = pred.volume > 0 ? act.volume / pred.volume : 0;
      if (volumeRatio >= 0.5 && volumeRatio <= 1.5) {
        volumeAccuracySum++;
      }
    }

    const directional = (correctDirections / barsCompared) * 100;
    const priceRMSE = Math.sqrt(priceSquaredErrors / barsCompared);
    const highLowAccuracy = (withinRange / barsCompared) * 100;
    const volumeAccuracy = (volumeAccuracySum / barsCompared) * 100;

    // Overall score: weighted combination
    // Directional (40%) + Price accuracy (30%) + Range (20%) + Volume (10%)
    const priceScore = Math.max(0, 100 - (priceRMSE / predicted[0].close) * 100);
    const overallScore = 
      directional * 0.4 +
      priceScore * 0.3 +
      highLowAccuracy * 0.2 +
      volumeAccuracy * 0.1;

    return {
      directional,
      priceRMSE,
      highLowAccuracy,
      volumeAccuracy,
      overallScore: Math.min(100, Math.max(0, overallScore)),
      barsCompared,
    };
  };

  /**
   * Add completed session to history
   */
  const addToHistory = (session: PredictionSession): void => {
    const entry: PredictionHistoryEntry = {
      sessionId: session.id,
      symbol: session.symbol,
      timeframe: session.timeframe,
      startedAt: session.startedAt,
      completedAt: Date.now(),
      barsPreicted: session.predictedBars.length,
      accuracy: session.accuracy,
    };

    setHistory(prev => [entry, ...prev].slice(0, 100)); // Keep last 100 entries
  };

  /**
   * Clear prediction for symbol/timeframe
   */
  const clearPrediction = (symbol: string, timeframe: string): void => {
    const key = getSessionKey(symbol, timeframe);
    setActiveSessions(prev => {
      const newMap = new Map(prev);
      newMap.delete(key);
      return newMap;
    });
    console.info(`🗑️ Prediction cleared: ${symbol} ${timeframe}`);
  };

  /**
   * Clear all predictions
   */
  const clearAllPredictions = (): void => {
    setActiveSessions(new Map());
    console.info('🗑️ All predictions cleared');
  };

  /**
   * Get prediction history for a symbol
   */
  const getHistory = (symbol?: string): PredictionHistoryEntry[] => {
    if (!symbol) return history();
    return history().filter(h => h.symbol === symbol);
  };

  return {
    // State accessors
    activeSessions,
    history,
    isLoading,
    error,
    
    // Methods
    getActiveSession,
    hasPrediction,
    startPrediction,
    addActualBar,
    clearPrediction,
    clearAllPredictions,
    getHistory,
    calculateAccuracy,
    
    // State setters
    setIsLoading,
    setError,
  };
}

// Create singleton store
export const predictionStore = createRoot(createPredictionStore);
