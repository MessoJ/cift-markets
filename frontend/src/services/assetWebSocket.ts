/**
 * Real-time Asset & News WebSocket Service
 * 
 * Connects to backend WebSocket for:
 * - Live asset price updates
 * - Market-moving news events
 * - Asset status changes (halts, volatility alerts)
 * - Event classification and severity updates
 */

import { createSignal } from 'solid-js';
import { AssetMarkerData, NewsEvent, MarketMovingEvent, calculateEventStatus } from '../config/assetColors';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://20.250.40.67:8000/api/v1';

export type AssetUpdateCallback = (asset: AssetMarkerData) => void;
export type NewsUpdateCallback = (news: NewsEvent) => void;
export type EventCallback = (event: MarketMovingEvent) => void;

class AssetWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: number | null = null;
  private assetCallbacks: Set<AssetUpdateCallback> = new Set();
  private newsCallbacks: Set<NewsUpdateCallback> = new Set();
  private eventCallbacks: Set<EventCallback> = new Set();
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  connect(token?: string) {
    const url = token 
      ? `${WS_BASE_URL}/globe/stream?token=${token}`
      : `${WS_BASE_URL}/globe/stream`;

    console.log('🔌 Connecting to asset WebSocket:', url);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('✅ Asset WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Subscribe to asset updates
        this.send({
          action: 'subscribe',
          channels: ['assets', 'news', 'events'],
        });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ Asset WebSocket error:', error);
      };

      this.ws.onclose = () => {
        console.log('🔌 Asset WebSocket disconnected');
        this.isConnected = false;
        this.scheduleReconnect();
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  private handleMessage(data: any) {
    switch (data.type) {
      case 'asset_update':
        this.handleAssetUpdate(data.payload);
        break;
      case 'news_event':
        this.handleNewsEvent(data.payload);
        break;
      case 'market_event':
        this.handleMarketEvent(data.payload);
        break;
      case 'bulk_update':
        // Handle batch updates for efficiency
        data.payload?.assets?.forEach((asset: any) => this.handleAssetUpdate(asset));
        data.payload?.news?.forEach((news: any) => this.handleNewsEvent(news));
        break;
      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  }

  private handleAssetUpdate(payload: any) {
    const asset: AssetMarkerData = {
      id: payload.id,
      name: payload.name,
      symbol: payload.symbol,
      category: payload.category,
      lat: payload.lat,
      lng: payload.lng,
      value: payload.value,
      change: payload.change,
      change_pct: payload.change_pct,
      eventStatus: calculateEventStatus(payload.change_pct, payload.news, payload.isHalted),
      news: payload.news,
      lastUpdate: new Date(payload.timestamp || Date.now()),
    };

    this.assetCallbacks.forEach(cb => cb(asset));
  }

  private handleNewsEvent(payload: any) {
    const news: NewsEvent = {
      id: payload.id,
      headline: payload.headline,
      summary: payload.summary,
      sentiment: payload.sentiment || 'neutral',
      importance: payload.importance || 'medium',
      timestamp: new Date(payload.timestamp),
      source: payload.source,
      url: payload.url,
      relatedAssets: payload.relatedAssets || [],
      impactScore: payload.impactScore,
    };

    this.newsCallbacks.forEach(cb => cb(news));
  }

  private handleMarketEvent(payload: any) {
    const event: MarketMovingEvent = {
      assetId: payload.assetId,
      eventType: payload.eventType,
      severity: payload.severity,
      description: payload.description,
      triggeredAt: new Date(payload.triggeredAt),
      relatedNews: payload.relatedNews || [],
    };

    this.eventCallbacks.forEach(cb => cb(event));
  }

  private scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached. Giving up.');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimeout = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  private send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  onAssetUpdate(callback: AssetUpdateCallback) {
    this.assetCallbacks.add(callback);
    return () => this.assetCallbacks.delete(callback);
  }

  onNewsUpdate(callback: NewsUpdateCallback) {
    this.newsCallbacks.add(callback);
    return () => this.newsCallbacks.delete(callback);
  }

  onMarketEvent(callback: EventCallback) {
    this.eventCallbacks.add(callback);
    return () => this.eventCallbacks.delete(callback);
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnected = false;
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }
}

export const assetWebSocket = new AssetWebSocketService();

/**
 * Solid.js hook for real-time asset updates
 */
export function useAssetStream() {
  const [assets, setAssets] = createSignal<Map<string, AssetMarkerData>>(new Map());
  const [latestNews, setLatestNews] = createSignal<NewsEvent[]>([]);
  const [events, setEvents] = createSignal<MarketMovingEvent[]>([]);

  const unsubscribeAsset = assetWebSocket.onAssetUpdate((asset) => {
    setAssets(prev => {
      const updated = new Map(prev);
      updated.set(asset.id, asset);
      return updated;
    });
  });

  const unsubscribeNews = assetWebSocket.onNewsUpdate((news) => {
    setLatestNews(prev => [news, ...prev].slice(0, 50)); // Keep latest 50
  });

  const unsubscribeEvent = assetWebSocket.onMarketEvent((event) => {
    setEvents(prev => [event, ...prev].slice(0, 20)); // Keep latest 20
  });

  // Cleanup on component unmount
  const cleanup = () => {
    unsubscribeAsset();
    unsubscribeNews();
    unsubscribeEvent();
  };

  return {
    assets,
    latestNews,
    events,
    cleanup,
  };
}
