/**
 * Chart Utility Functions
 * 
 * High-performance data transformation and calculation utilities for financial charting.
 * Optimized for real-time updates and large datasets.
 */

import type {
  OHLCVBar,
  CandlestickDataPoint,
  VolumeDataPoint,
} from '~/types/chart.types';

// ============================================================================
// DATA TRANSFORMATION (Zero-copy where possible)
// ============================================================================

/**
 * Transform backend OHLCV data to ECharts candlestick format
 * ECharts format: [timestamp, open, close, low, high]
 * Note: ECharts uses close before low/high
 */
export function transformToEChartsData(bars: OHLCVBar[]): CandlestickDataPoint[] {
  return bars.map((bar) => {
    const timestamp = typeof bar.timestamp === 'string' 
      ? new Date(bar.timestamp).getTime() 
      : bar.timestamp;
    
    return [timestamp, bar.open, bar.close, bar.low, bar.high];
  });
}

/**
 * Transform OHLCV data to volume bars with direction
 * Direction: 1 for up candles (close > open), -1 for down
 */
export function transformToVolumeData(bars: OHLCVBar[]): VolumeDataPoint[] {
  return bars.map((bar) => {
    const timestamp = typeof bar.timestamp === 'string' 
      ? new Date(bar.timestamp).getTime() 
      : bar.timestamp;
    
    const direction = bar.close >= bar.open ? 1 : -1;
    
    return [timestamp, bar.volume, direction];
  });
}

/**
 * Transform OHLCV data to Heikin Ashi candles
 */
export function transformToHeikinAshi(bars: OHLCVBar[]): OHLCVBar[] {
  if (bars.length === 0) return [];

  const haBars: OHLCVBar[] = [];
  
  // First bar is same as regular candle
  let prevHaOpen = bars[0].open;
  let prevHaClose = bars[0].close;
  
  haBars.push({
    ...bars[0],
    open: prevHaOpen,
    close: prevHaClose,
    high: bars[0].high,
    low: bars[0].low,
  });

  for (let i = 1; i < bars.length; i++) {
    const bar = bars[i];
    
    // HA Close = (Open + High + Low + Close) / 4
    const haClose = (bar.open + bar.high + bar.low + bar.close) / 4;
    
    // HA Open = (Prev HA Open + Prev HA Close) / 2
    const haOpen = (prevHaOpen + prevHaClose) / 2;
    
    // HA High = Max(High, HA Open, HA Close)
    const haHigh = Math.max(bar.high, haOpen, haClose);
    
    // HA Low = Min(Low, HA Open, HA Close)
    const haLow = Math.min(bar.low, haOpen, haClose);
    
    haBars.push({
      ...bar,
      open: haOpen,
      close: haClose,
      high: haHigh,
      low: haLow,
    });
    
    prevHaOpen = haOpen;
    prevHaClose = haClose;
  }
  
  return haBars;
}

// ============================================================================
// TECHNICAL CALCULATIONS (Simple Moving Average for Phase 1)
// ============================================================================

/**
 * Calculate Simple Moving Average (SMA)
 * Backend will handle this in Phase 3 using Polars for better performance
 */
export function calculateSMA(data: number[], period: number): (number | null)[] {
  if (data.length < period) {
    return new Array(data.length).fill(null);
  }

  const result: (number | null)[] = new Array(period - 1).fill(null);
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const sum = slice.reduce((acc, val) => acc + val, 0);
    result.push(sum / period);
  }
  
  return result;
}

/**
 * Extract close prices from OHLCV bars
 */
export function extractClosePrices(bars: OHLCVBar[]): number[] {
  return bars.map((bar) => bar.close);
}

// ============================================================================
// PRICE FORMATTING
// ============================================================================

/**
 * Format price with appropriate decimal places based on value
 */
export function formatPrice(price: number | null | undefined): string {
  if (price === null || price === undefined || isNaN(price)) {
    return '0.00';
  }
  
  if (price >= 1000) {
    return price.toFixed(2);
  } else if (price >= 1) {
    return price.toFixed(4);
  } else {
    return price.toFixed(6);
  }
}

/**
 * Format volume with K, M, B suffixes
 */
export function formatVolume(volume: number | null | undefined): string {
  if (volume === null || volume === undefined || isNaN(volume)) {
    return '0';
  }
  
  if (volume >= 1_000_000_000) {
    return (volume / 1_000_000_000).toFixed(2) + 'B';
  } else if (volume >= 1_000_000) {
    return (volume / 1_000_000).toFixed(2) + 'M';
  } else if (volume >= 1_000) {
    return (volume / 1_000).toFixed(2) + 'K';
  }
  return volume.toString();
}

/**
 * Format percentage change
 */
export function formatPercentage(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.00%';
  }
  
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

// ============================================================================
// TIME UTILITIES
// ============================================================================

/**
 * Format timestamp for chart axis
 */
export function formatChartTime(timestamp: number, timeframe: string): string {
  const date = new Date(timestamp);
  
  if (timeframe === '1m' || timeframe === '5m' || timeframe === '15m') {
    // Show time for intraday
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  } else if (timeframe === '1h' || timeframe === '4h') {
    // Show date and time
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  } else {
    // Show date only for daily+
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  }
}

// ============================================================================
// DATA VALIDATION
// ============================================================================

/**
 * Validate OHLCV bar data integrity
 */
export function isValidBar(bar: OHLCVBar): boolean {
  return (
    bar.high >= bar.low &&
    bar.high >= bar.open &&
    bar.high >= bar.close &&
    bar.low <= bar.open &&
    bar.low <= bar.close &&
    bar.volume >= 0 &&
    !isNaN(bar.open) &&
    !isNaN(bar.high) &&
    !isNaN(bar.low) &&
    !isNaN(bar.close)
  );
}

/**
 * Filter out invalid bars and log errors
 */
export function validateAndFilterBars(bars: OHLCVBar[]): OHLCVBar[] {
  const valid = bars.filter((bar) => {
    const valid = isValidBar(bar);
    if (!valid) {
      console.warn('Invalid OHLCV bar detected:', bar);
    }
    return valid;
  });
  
  if (valid.length !== bars.length) {
    console.warn(`Filtered ${bars.length - valid.length} invalid bars`);
  }
  
  return valid;
}

// ============================================================================
// PERFORMANCE METRICS
// ============================================================================

/**
 * Calculate price change and percentage for display
 */
export function calculatePriceChange(bars: OHLCVBar[]): {
  change: number;
  changePercent: number;
  direction: 'up' | 'down' | 'neutral';
} {
  if (bars.length < 2) {
    return { change: 0, changePercent: 0, direction: 'neutral' };
  }

  const latest = bars[bars.length - 1];
  const previous = bars[bars.length - 2];
  
  // Null-safe calculation
  if (!latest?.close || !previous?.close) {
    return { change: 0, changePercent: 0, direction: 'neutral' };
  }
  
  const change = latest.close - previous.close;
  const changePercent = previous.close !== 0 ? (change / previous.close) * 100 : 0;
  
  const direction = change > 0 ? 'up' : change < 0 ? 'down' : 'neutral';
  
  return { change, changePercent, direction };
}

/**
 * Get latest bar info for display
 */
export function getLatestBarInfo(bars: OHLCVBar[]) {
  if (bars.length === 0) {
    return null;
  }

  const latest = bars[bars.length - 1];
  const { change, changePercent, direction } = calculatePriceChange(bars);

  return {
    symbol: latest.symbol,
    price: latest.close,
    open: latest.open,
    high: latest.high,
    low: latest.low,
    volume: latest.volume,
    change,
    changePercent,
    direction,
    timestamp: latest.timestamp,
  };
}

// ============================================================================
// CHART RANGE CALCULATIONS
// ============================================================================

/**
 * Calculate appropriate price range for Y-axis
 * Adds padding for better visualization
 */
export function calculatePriceRange(bars: OHLCVBar[], paddingPercent = 5): [number, number] {
  if (bars.length === 0) {
    return [0, 100];
  }

  const highs = bars.map((b) => b.high);
  const lows = bars.map((b) => b.low);
  
  const maxHigh = Math.max(...highs);
  const minLow = Math.min(...lows);
  
  const range = maxHigh - minLow;
  const padding = range * (paddingPercent / 100);
  
  return [minLow - padding, maxHigh + padding];
}
