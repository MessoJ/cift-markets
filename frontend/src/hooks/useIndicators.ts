/**
 * useIndicators Hook
 * 
 * Fetch and manage technical indicators from backend (Polars-calculated).
 * Handles caching, refetching, and error states.
 */

import { createSignal, createEffect, on } from 'solid-js';
import { apiClient } from '~/lib/api/client';

export interface IndicatorData {
  timestamp: string;
  symbol: string;
  close: number;
  [key: string]: any; // Dynamic indicator values
}

export interface UseIndicatorsOptions {
  symbol: string;
  timeframe: string;
  limit?: number;
  indicators: string[];
  enabled?: boolean;
}

export function useIndicators(options: UseIndicatorsOptions) {
  const [data, setData] = createSignal<IndicatorData[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  /**
   * Fetch indicators from backend
   */
  const fetchIndicators = async () => {
    if (!options.enabled || options.indicators.length === 0) {
      setData([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      console.info(
        `📊 Fetching indicators for ${options.symbol}: ${options.indicators.join(', ')}`
      );

      // Build query string with multiple indicators
      const params = new URLSearchParams({
        timeframe: options.timeframe,
        limit: String(options.limit || 100),
      });
      
      // Add each indicator as a separate query parameter
      options.indicators.forEach((ind) => {
        params.append('indicators', ind);
      });

      // Call backend API
      // Use relative path to leverage Vite proxy
      const response = await fetch(
        `/api/v1/market-data/indicators/${options.symbol}?${params.toString()}`,
        {
          credentials: 'include',
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const indicatorData: IndicatorData[] = await response.json();

      if (!indicatorData || indicatorData.length === 0) {
        throw new Error('No indicator data returned');
      }

      setData(indicatorData);
      console.info(`✅ Loaded ${indicatorData.length} indicator data points`);
    } catch (err: any) {
      console.error('❌ Indicator fetch failed:', err);
      console.error('Indicator error details:', {
        message: err.message,
        stack: err.stack,
        indicators: options.indicators,
      });
      setError(err.message || 'Failed to load indicators');
    } finally {
      setLoading(false);
    }
  };

  // Fetch when dependencies change
  createEffect(
    on(
      () => [options.symbol, options.timeframe, options.indicators.join(','), options.enabled],
      () => {
        fetchIndicators();
      }
    )
  );

  return {
    data,
    loading,
    error,
    refetch: fetchIndicators,
  };
}
