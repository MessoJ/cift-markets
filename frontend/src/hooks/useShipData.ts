/**
 * Ship Tracking Data Hook
 * Fetches real-time positions of tracked vessels
 */

import { createSignal, createEffect, onCleanup } from 'solid-js';

export interface TrackedShip {
  id: string;
  mmsi: string;
  imo: string;
  ship_name: string;
  ship_type: 'oil_tanker' | 'lng_carrier' | 'container' | 'bulk_carrier' | 'chemical_tanker';
  flag_country: string;
  flag_country_code: string;
  deadweight_tonnage: number;
  current_lat: number;
  current_lng: number;
  current_speed: number;
  current_course: number;
  current_status: string;
  destination: string;
  eta: string;
  cargo_type: string;
  cargo_value_usd: number;
  importance_score: number;
  news_count: number;
  avg_sentiment: number;
  last_updated: string;
}

export interface ShipFilters {
  ship_type?: 'all' | 'oil_tanker' | 'lng_carrier' | 'container' | 'bulk_carrier' | 'chemical_tanker';
  min_importance?: number;
  status?: 'all' | 'underway' | 'at_anchor' | 'moored';
}

export function useShipData(filters: ShipFilters = {}) {
  const [ships, setShips] = createSignal<TrackedShip[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);

  let refreshInterval: number | undefined;

  const fetchShips = async () => {
    setLoading(true);
    setError(null);

    try {
      // Build query params
      const params = new URLSearchParams();
      
      if (filters.ship_type && filters.ship_type !== 'all') {
        params.append('ship_type', filters.ship_type);
      }
      
      if (filters.min_importance !== undefined) {
        params.append('min_importance', String(filters.min_importance));
      }
      
      if (filters.status && filters.status !== 'all') {
        params.append('status', filters.status);
      }

      const url = `/api/v1/globe/ships?${params}`;
      console.log('🚢 Fetching ships from:', url);

      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      console.log(`✅ Loaded ${data.ships?.length || 0} ships from API`);
      setShips(data.ships || []);
      setLoading(false);
      
    } catch (err: any) {
      console.error('❌ Error fetching ship data:', err);
      setError(err.message || 'Failed to fetch ship data');
      setLoading(false);
    }
  };

  // Initial fetch
  createEffect(() => {
    fetchShips();
  });

  // Auto-refresh every 2 minutes (ships move slowly)
  createEffect(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }

    refreshInterval = window.setInterval(() => {
      console.log('🔄 Auto-refreshing ship positions...');
      fetchShips();
    }, 120000); // 2 minutes

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  });

  onCleanup(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  return {
    ships,
    loading,
    error,
    refetch: fetchShips,
  };
}
