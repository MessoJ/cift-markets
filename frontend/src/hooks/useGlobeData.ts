/**
 * Globe Data Hook
 * Fetches exchange markers, arcs, and boundaries from API
 */

import { createSignal, createEffect, onCleanup } from 'solid-js';

console.log('✅ useGlobeData.ts file loaded');

export interface GlobeExchange {
  id: string;
  code: string;
  name: string;
  country: string;
  country_code: string;
  flag: string;
  lat: number;
  lng: number;
  timezone: string;
  market_cap_usd: number;
  website: string;
  icon_url: string;
  news_count: number;
  sentiment_score: number;
  categories: string[];
  latest_articles: any[];
}

export interface GlobeArc {
  id: string;
  source: {
    code: string;
    name: string;
    lat: number;
    lng: number;
  };
  target: {
    code: string;
    name: string;
    lat: number;
    lng: number;
  };
  article_count: number;
  connection_type: string;
  strength: number;
  color: [string, string];
  articles: any[];
}

export interface GlobeCountry {
  country_code: string;
  name: string;
  flag: string;
  article_count: number;
  sentiment_score: number;
  top_categories: string[];
  exchanges: string[];
}

export interface GlobeFilters {
  timeframe?: string;
  query?: string;
  exchanges?: string[];
  countries?: string[];
  categories?: string[];
  sentiment?: string;
  min_articles?: number;
  min_strength?: number;
  connection_type?: string;
}

export function useGlobeData(initialFilters: GlobeFilters = {}) {
  console.log('🌍 useGlobeData hook initialized with filters:', initialFilters);
  const [exchanges, setExchanges] = createSignal<GlobeExchange[]>([]);
  const [arcs, setArcs] = createSignal<GlobeArc[]>([]);
  const [boundaries, setBoundaries] = createSignal<GlobeCountry[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [filters, setFilters] = createSignal<GlobeFilters>(initialFilters);
  
  let abortController: AbortController | null = null;

  const fetchGlobeData = async () => {
    // Cancel previous request
    if (abortController) {
      abortController.abort();
    }
    abortController = new AbortController();

    setLoading(true);
    setError(null);

    try {
      const currentFilters = filters();
      const timeframe = currentFilters.timeframe || '24h';
      
      // Build query params for exchanges
      const exchangeParams = new URLSearchParams({
        timeframe,
        min_articles: String(currentFilters.min_articles || 0),
      });

      // Build query params for arcs
      const arcParams = new URLSearchParams({
        timeframe,
        min_strength: String(currentFilters.min_strength || 0.3),
      });
      if (currentFilters.connection_type && currentFilters.connection_type !== 'all') {
        arcParams.append('connection_type', currentFilters.connection_type);
      }

      // Build query params for boundaries
      const boundaryParams = new URLSearchParams({
        timeframe,
      });

      // Fetch all data in parallel
      console.log('🔄 Fetching globe data from API...');
      console.log('Exchange URL:', `/api/v1/globe/exchanges?${exchangeParams}`);
      
      const [exchangesRes, arcsRes, boundariesRes] = await Promise.all([
        fetch(`/api/v1/globe/exchanges?${exchangeParams}`, {
          signal: abortController.signal,
        }).then(r => r.json()),
        fetch(`/api/v1/globe/arcs?${arcParams}`, {
          signal: abortController.signal,
        }).then(r => r.json()),
        fetch(`/api/v1/globe/boundaries?${boundaryParams}`, {
          signal: abortController.signal,
        }).then(r => r.json()),
      ]);

      console.log('✅ API Response:', {
        exchanges: exchangesRes.exchanges?.length || 0,
        arcs: arcsRes.arcs?.length || 0,
        boundaries: boundariesRes.countries?.length || 0
      });
      console.log('Sample exchange:', exchangesRes.exchanges?.[0]);

      // Apply additional client-side filters if needed
      let filteredExchanges = exchangesRes.exchanges || [];
      console.log('📍 Raw exchanges from API:', filteredExchanges.length);
      
      if (currentFilters.query) {
        const query = currentFilters.query.toLowerCase();
        filteredExchanges = filteredExchanges.filter((ex: GlobeExchange) =>
          ex.name.toLowerCase().includes(query) ||
          ex.code.toLowerCase().includes(query) ||
          ex.country.toLowerCase().includes(query)
        );
      }

      if (currentFilters.exchanges && currentFilters.exchanges.length > 0) {
        filteredExchanges = filteredExchanges.filter((ex: GlobeExchange) =>
          currentFilters.exchanges!.includes(ex.code)
        );
      }

      if (currentFilters.countries && currentFilters.countries.length > 0) {
        filteredExchanges = filteredExchanges.filter((ex: GlobeExchange) =>
          currentFilters.countries!.includes(ex.country_code)
        );
      }

      if (currentFilters.sentiment) {
        filteredExchanges = filteredExchanges.filter((ex: GlobeExchange) => {
          if (currentFilters.sentiment === 'positive') return ex.sentiment_score > 0.2;
          if (currentFilters.sentiment === 'negative') return ex.sentiment_score < -0.2;
          return ex.sentiment_score >= -0.2 && ex.sentiment_score <= 0.2;
        });
      }

      console.log('💾 Setting exchanges state:', filteredExchanges.length);
      setExchanges(filteredExchanges);
      setArcs(arcsRes.arcs || []);
      setBoundaries(boundariesRes.countries || []);
      setLoading(false);
      
      console.log('🎯 Globe data state updated:', {
        exchangesCount: filteredExchanges.length,
        arcsCount: arcsRes.arcs?.length || 0,
        loading: false
      });
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // Request was cancelled, ignore
        return;
      }
      
      console.error('Error fetching globe data:', err);
      setError(err.message || 'Failed to fetch globe data');
      setLoading(false);
    }
  };

  const updateFilters = (newFilters: Partial<GlobeFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  const resetFilters = () => {
    setFilters(initialFilters);
  };

  // Fetch data when filters change
  createEffect(() => {
    filters(); // Track dependency
    fetchGlobeData();
  });

  // Cleanup on unmount
  onCleanup(() => {
    if (abortController) {
      abortController.abort();
    }
  });

  return {
    exchanges,
    arcs,
    boundaries,
    loading,
    error,
    filters,
    updateFilters,
    resetFilters,
    refetch: fetchGlobeData,
  };
}
