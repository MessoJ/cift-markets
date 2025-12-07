/**
 * useAssetData Hook
 * Fetches and manages asset location data for globe visualization
 * No mock data - all from API/database
 */

import { createSignal, createEffect, onCleanup } from 'solid-js';

export interface AssetLocation {
  id: string;
  code: string;
  name: string;
  asset_type: 'central_bank' | 'commodity_market' | 'government' | 'tech_hq' | 'energy';
  country: string;
  country_code: string;
  city: string;
  flag: string;
  lat: number;
  lng: number;
  timezone: string;
  description: string;
  importance_score: number;
  website: string;
  icon_url: string;
  current_status: 'operational' | 'unknown' | 'issue';
  sentiment_score: number;
  news_count: number;
  last_news_at: string | null;
  categories: string[];
  latest_articles: Array<{
    id: string;
    title: string;
    summary: string;
    published_at: string;
    sentiment: string;
    category: string;
  }>;
}

export interface AssetFilters {
  timeframe?: '1h' | '24h' | '7d' | '30d';
  asset_type?: 'central_bank' | 'commodity_market' | 'government' | 'tech_hq' | 'energy' | 'all';
  status?: 'operational' | 'unknown' | 'issue' | 'all';
  min_importance?: number;
}

export interface AssetStatusSummary {
  operational: number;
  unknown: number;
  issue: number;
}

console.log('✅ useAssetData.ts file loaded');

export function useAssetData(initialFilters: AssetFilters = {}) {
  console.log('🏛️ useAssetData hook initialized with filters:', initialFilters);
  
  const [assets, setAssets] = createSignal<AssetLocation[]>([]);
  const [statusSummary, setStatusSummary] = createSignal<AssetStatusSummary>({
    operational: 0,
    unknown: 0,
    issue: 0,
  });
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [lastUpdated, setLastUpdated] = createSignal<string>('');
  
  const [filters, setFilters] = createSignal<AssetFilters>({
    timeframe: '24h',
    asset_type: 'all',
    status: 'all',
    min_importance: 0,
    ...initialFilters,
  });

  async function fetchAssets() {
    console.log('📡 Fetching asset data from API...');
    setLoading(true);
    setError(null);

    try {
      const currentFilters = filters();
      const params = new URLSearchParams();
      
      if (currentFilters.timeframe) params.append('timeframe', currentFilters.timeframe);
      if (currentFilters.asset_type) params.append('asset_type', currentFilters.asset_type);
      if (currentFilters.status) params.append('status', currentFilters.status);
      if (currentFilters.min_importance !== undefined) {
        params.append('min_importance', currentFilters.min_importance.toString());
      }

      const url = `/api/v1/globe/assets/?${params.toString()}`;
      console.log('🌐 Fetching from:', url);

      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ Asset data received:', {
        total: data.total_count,
        status: data.status_summary,
        timeframe: data.timeframe,
      });

      setAssets(data.assets || []);
      setStatusSummary(data.status_summary || { operational: 0, unknown: 0, issue: 0 });
      setLastUpdated(data.last_updated || new Date().toISOString());
      setLoading(false);

    } catch (err) {
      console.error('❌ Error fetching asset data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch asset data');
      setAssets([]);
      setLoading(false);
    }
  }

  // Fetch on mount and when filters change
  createEffect(() => {
    console.log('🔄 Filters changed, refetching assets...', filters());
    fetchAssets();
  });

  // Auto-refresh every 5 minutes
  const refreshInterval = setInterval(() => {
    console.log('🔄 Auto-refreshing asset data...');
    fetchAssets();
  }, 5 * 60 * 1000);

  onCleanup(() => {
    console.log('🧹 Cleaning up useAssetData');
    clearInterval(refreshInterval);
  });

  return {
    assets,
    statusSummary,
    loading,
    error,
    lastUpdated,
    filters,
    setFilters,
    refetch: fetchAssets,
  };
}
