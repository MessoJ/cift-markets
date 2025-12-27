/**
 * CIFT Markets - Prediction Types
 * 
 * Type definitions for the chart prediction feature.
 * Supports ML model predictions for price movements.
 */

export interface PredictedBar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  confidence: number; // 0-1 confidence score from ML model
}

export interface ActualBar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionSession {
  id: string;
  symbol: string;
  timeframe: string;
  startedAt: number; // Timestamp when prediction was initiated
  predictionStartIndex: number; // Index in chart where prediction line should appear
  predictedBars: PredictedBar[];
  actualBars: ActualBar[];
  status: 'active' | 'completed' | 'expired';
  accuracy?: PredictionAccuracy;
  modelId: string;
  modelVersion: string;
}

export interface PredictionAccuracy {
  directional: number; // % correct direction predictions
  priceRMSE: number; // Root Mean Square Error of price predictions
  highLowAccuracy: number; // % times actual was within predicted range
  volumeAccuracy: number; // % accuracy on volume prediction
  overallScore: number; // Combined score 0-100
  barsCompared: number;
}

export interface PredictionRequest {
  symbol: string;
  timeframe: string;
  barsAhead: number; // How many bars to predict
  modelId?: string; // Optional specific model to use
  includeConfidenceInterval?: boolean;
}

export interface PredictionResponse {
  sessionId: string;
  predictions: PredictedBar[];
  modelId: string;
  modelVersion: string;
  generatedAt: number;
  expiresAt: number;
}

export interface PredictionHistoryEntry {
  sessionId: string;
  symbol: string;
  timeframe: string;
  startedAt: number;
  completedAt?: number;
  barsPreicted: number;
  accuracy?: PredictionAccuracy;
}

// Visual styling for prediction display
export interface PredictionStyleConfig {
  predictedBarOpacity: number; // Default 0.4
  predictionLineColor: string; // Dotted line color
  predictionLineWidth: number;
  bullishPredictColor: string;
  bearishPredictColor: string;
  confidenceShowThreshold: number; // Show confidence if above this
}

export const DEFAULT_PREDICTION_STYLE: PredictionStyleConfig = {
  predictedBarOpacity: 0.45,
  predictionLineColor: '#f97316', // CIFT Accent Orange
  predictionLineWidth: 2,
  bullishPredictColor: 'rgba(34, 197, 94, 0.45)', // Green with opacity
  bearishPredictColor: 'rgba(239, 68, 68, 0.45)', // Red with opacity
  confidenceShowThreshold: 0.55, // Show confidence if above 55%
};
