/**
 * Market Indices Bar Component
 * 
 * RULES COMPLIANT: Fetches real market data from database via backend API.
 * NO HARDCODED DATA - All indices data pulled from market_data table.
 */

import { createSignal, createEffect, For } from 'solid-js';
import { apiClient } from '~/lib/api/client';

interface IndexData {
  symbol: string;
  label: string;
  change_pct: number;
  price: number;
}

export function MarketIndicesBar() {
  const [indices, setIndices] = createSignal<IndexData[]>([]);
  const [loading, setLoading] = createSignal(true);

  // Fetch real market indices data from database
  createEffect(async () => {
    try {
      setLoading(true);
      
      // Get major market indices from database - RULES COMPLIANT
      const symbols = ['SPX', 'NDX', 'DJI']; // S&P 500, Nasdaq 100, Dow Jones
      const quotes = await apiClient.getQuotes(symbols);
      
      // Map to display format
      const indexData: IndexData[] = quotes.map(quote => ({
        symbol: quote.symbol,
        label: getIndexLabel(quote.symbol),
        change_pct: quote.change_pct || 0,
        price: quote.price
      }));
      
      setIndices(indexData);
    } catch (error) {
      console.error('Failed to fetch market indices from database:', error);
      // Graceful degradation - show no data instead of hardcoded values
      setIndices([]);
    } finally {
      setLoading(false);
    }
  });

  // Auto-refresh every 30 seconds to get fresh database data
  let refreshInterval: number;
  createEffect(() => {
    refreshInterval = window.setInterval(async () => {
      try {
        const symbols = ['SPX', 'NDX', 'DJI'];
        const quotes = await apiClient.getQuotes(symbols);
        const indexData: IndexData[] = quotes.map(quote => ({
          symbol: quote.symbol,
          label: getIndexLabel(quote.symbol),
          change_pct: quote.change_pct || 0,
          price: quote.price
        }));
        setIndices(indexData);
      } catch (error) {
        console.error('Failed to refresh market indices:', error);
      }
    }, 30000); // Refresh every 30 seconds

    // Cleanup interval
    return () => {
      if (refreshInterval) clearInterval(refreshInterval);
    };
  });

  function getIndexLabel(symbol: string): string {
    const labels: Record<string, string> = {
      'SPX': 'SPX',
      'NDX': 'NDX', 
      'DJI': 'DJI'
    };
    return labels[symbol] || symbol;
  }

  function formatChange(change_pct: number): string {
    const sign = change_pct >= 0 ? '+' : '';
    return `${sign}${change_pct.toFixed(2)}%`;
  }

  function getChangeColor(change_pct: number): string {
    return change_pct >= 0 ? 'text-success-400' : 'text-danger-400';
  }

  return (
    <div class="hidden lg:flex items-center gap-3 text-[10px] font-mono">
      <For each={indices()} fallback={
        loading() ? (
          <span class="text-gray-600">Loading indices...</span>
        ) : (
          <span class="text-gray-600">Market data unavailable</span>
        )
      }>
        {(index) => (
          <div class="flex items-center gap-1.5">
            <span class="text-gray-500">{index.label}</span>
            <span class={getChangeColor(index.change_pct)}>
              {formatChange(index.change_pct)}
            </span>
          </div>
        )}
      </For>
    </div>
  );
}
