/**
 * Chart Type Definitions
 * 
 * Advanced data structures for financial charting with ML integration support.
 * Designed for Hawkes process and other quantitative models.
 */

// ============================================================================
// OHLCV DATA STRUCTURES
// ============================================================================

export interface OHLCVBar {
  timestamp: string | number;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * ECharts-formatted candlestick data point
 * Format: [timestamp, open, close, low, high]
 * Note: ECharts uses [open, close, low, high] order (not OHLC)
 */
export type CandlestickDataPoint = [number, number, number, number, number];

/**
 * Volume bar data point
 * Format: [timestamp, volume, direction]
 * Direction: 1 for up (green), -1 for down (red)
 */
export type VolumeDataPoint = [number, number, number];

// ============================================================================
// CHART CONFIGURATION
// ============================================================================

export interface ChartTimeframe {
  value: string;
  label: string;
  minutes: number;
}

export const TIMEFRAMES: ChartTimeframe[] = [
  { value: '1m', label: '1m', minutes: 1 },
  { value: '5m', label: '5m', minutes: 5 },
  { value: '15m', label: '15m', minutes: 15 },
  { value: '30m', label: '30m', minutes: 30 },
  { value: '1h', label: '1H', minutes: 60 },
  { value: '4h', label: '4H', minutes: 240 },
  { value: '1d', label: '1D', minutes: 1440 },
];

export interface ChartSettings {
  symbol: string;
  timeframe: string;
  candleLimit: number;
  showVolume: boolean;
  showGrid: boolean;
  theme: 'dark' | 'light';
  colors: ChartColorScheme;
}

export interface ChartColorScheme {
  background: string;
  gridColor: string;
  textColor: string;
  bullish: string;
  bearish: string;
  volumeUp: string;
  volumeDown: string;
  ma20?: string;
  ma50?: string;
  ma200?: string;
}

export const DARK_THEME: ChartColorScheme = {
  background: '#1e222d',        // TradingView dark background
  gridColor: '#363c4e',         // Subtle grid lines
  textColor: '#d1d4dc',         // Light gray text
  bullish: '#26a69a',           // TradingView teal-green
  bearish: '#ef5350',           // TradingView coral-red
  volumeUp: 'rgba(38, 166, 154, 0.5)',   // Transparent teal
  volumeDown: 'rgba(239, 83, 80, 0.5)',  // Transparent coral
  ma20: '#2196F3',              // Blue
  ma50: '#FF9800',              // Orange
  ma200: '#9C27B0',             // Purple
};

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

export interface IndicatorData {
  name: string;
  data: [number, number][]; // [timestamp, value]
  color: string;
  lineWidth?: number;
  lineStyle?: 'solid' | 'dashed' | 'dotted';
}

export interface TechnicalIndicators {
  ma20?: number[];
  ma50?: number[];
  ma200?: number[];
  rsi?: number[];
  macd?: {
    macd: number[];
    signal: number[];
    histogram: number[];
  };
  bollinger?: {
    upper: number[];
    middle: number[];
    lower: number[];
  };
}

// ============================================================================
// ML MODEL INTEGRATION (Hawkes Process Support)
// ============================================================================

/**
 * Hawkes process event data for order flow prediction
 */
export interface HawkesEvent {
  timestamp: number;
  intensity: number;
  type: 'buy' | 'sell';
  predicted?: boolean;
}

/**
 * ML model prediction overlay
 */
export interface ModelPrediction {
  timestamp: number;
  predictedPrice: number;
  confidence: number;
  upper_bound?: number;
  lower_bound?: number;
}

/**
 * Order flow intensity (from Hawkes process)
 */
export interface OrderFlowIntensity {
  timestamp: number;
  buyIntensity: number;
  sellIntensity: number;
  netIntensity: number;
}

// ============================================================================
// CHART STATE
// ============================================================================

export interface ChartState {
  isLoading: boolean;
  error: string | null;
  lastUpdate: number;
  dataRange: {
    start: number;
    end: number;
  };
  zoomLevel: number;
}

// ============================================================================
// WEBSOCKET DATA STRUCTURES
// ============================================================================

export interface TickUpdate {
  type: 'tick' | 'price';
  symbol: string;
  price: number;
  volume?: number;
  timestamp: string;
  bid?: number;
  ask?: number;
}

export interface CandleUpdate {
  type: 'candle_update';
  symbol: string;
  timeframe: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  is_closed: boolean;
}
