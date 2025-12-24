/**
 * Market Data WebSocket Hook
 * 
 * Advanced WebSocket integration for real-time market data streaming.
 * 
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Subscription management
 * - Connection status tracking
 * - Type-safe message handling
 * - Error recovery
 */

import { createSignal, onCleanup, createEffect } from 'solid-js';
import type { TickUpdate, CandleUpdate } from '~/types/chart.types';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export interface MarketDataSubscription {
  symbols: string[];
  onTick?: (tick: TickUpdate) => void;
  onCandle?: (candle: CandleUpdate) => void;
  onError?: (error: Error) => void;
}

export interface UseMarketDataWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectDelay?: number;
  maxReconnectDelay?: number;
  reconnectDecay?: number;
}

const getWebSocketUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}/api/v1/market-data/ws/stream`;
};

const DEFAULT_OPTIONS: Required<UseMarketDataWebSocketOptions> = {
  url: getWebSocketUrl(),
  autoConnect: true,
  reconnectDelay: 1000,
  maxReconnectDelay: 30000,
  reconnectDecay: 1.5,
};

export function useMarketDataWebSocket(options: UseMarketDataWebSocketOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [status, setStatus] = createSignal<ConnectionStatus>('disconnected');
  const [error, setError] = createSignal<string | null>(null);
  const [subscribedSymbols, setSubscribedSymbols] = createSignal<string[]>([]);
  
  let ws: WebSocket | null = null;
  let reconnectTimeout: number | null = null;
  let currentReconnectDelay = opts.reconnectDelay;
  let intentionalClose = false;
  
  // Callbacks storage
  const callbacks: {
    onTick: Set<(tick: TickUpdate) => void>;
    onCandle: Set<(candle: CandleUpdate) => void>;
    onError: Set<(error: Error) => void>;
  } = {
    onTick: new Set(),
    onCandle: new Set(),
    onError: new Set(),
  };

  /**
   * Connect to WebSocket server
   */
  const connect = () => {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
      console.warn('WebSocket already connected or connecting');
      return;
    }

    try {
      setStatus('connecting');
      setError(null);
      
      console.info(`🔌 Connecting to WebSocket: ${opts.url}`);
      ws = new WebSocket(opts.url);

      ws.onopen = () => {
        console.info('✅ WebSocket connected');
        setStatus('connected');
        setError(null);
        currentReconnectDelay = opts.reconnectDelay;
        
        // Resubscribe to symbols if any
        const symbols = subscribedSymbols();
        if (symbols.length > 0) {
          subscribe(symbols);
        }
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
          const error = new Error('Invalid message format');
          callbacks.onError.forEach((cb) => cb(error));
        }
      };

      ws.onerror = (event) => {
        console.error('❌ WebSocket error:', event);
        setStatus('error');
        setError('WebSocket connection error');
        
        const error = new Error('WebSocket connection error');
        callbacks.onError.forEach((cb) => cb(error));
      };

      ws.onclose = (event) => {
        console.info(`🔌 WebSocket closed: ${event.code} - ${event.reason}`);
        setStatus('disconnected');
        ws = null;

        // Attempt reconnection if not intentional close
        if (!intentionalClose) {
          scheduleReconnect();
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Connection failed');
      scheduleReconnect();
    }
  };

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = () => {
    intentionalClose = true;
    
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }

    if (ws) {
      ws.close(1000, 'Client disconnect');
      ws = null;
    }

    setStatus('disconnected');
    setSubscribedSymbols([]);
  };

  /**
   * Schedule reconnection with exponential backoff
   */
  const scheduleReconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
    }

    console.info(`⏱️  Reconnecting in ${currentReconnectDelay}ms...`);
    
    reconnectTimeout = window.setTimeout(() => {
      currentReconnectDelay = Math.min(
        currentReconnectDelay * opts.reconnectDecay,
        opts.maxReconnectDelay
      );
      connect();
    }, currentReconnectDelay);
  };

  /**
   * Subscribe to symbols
   */
  const subscribe = (symbols: string[]) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot subscribe: WebSocket not connected');
      return;
    }

    const message = {
      action: 'subscribe',
      symbols: symbols,
    };

    ws.send(JSON.stringify(message));
    setSubscribedSymbols((prev) => [...new Set([...prev, ...symbols])]);
    
    console.info(`📊 Subscribed to: ${symbols.join(', ')}`);
  };

  /**
   * Unsubscribe from symbols
   */
  const unsubscribe = (symbols: string[]) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot unsubscribe: WebSocket not connected');
      return;
    }

    const message = {
      action: 'unsubscribe',
      symbols: symbols,
    };

    ws.send(JSON.stringify(message));
    setSubscribedSymbols((prev) => prev.filter((s) => !symbols.includes(s)));
    
    console.info(`📊 Unsubscribed from: ${symbols.join(', ')}`);
  };

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'tick':
      case 'price':
        const tick: TickUpdate = {
          type: message.type as 'tick' | 'price',
          symbol: message.symbol,
          price: message.price,
          volume: message.volume,
          timestamp: message.timestamp,
          bid: message.bid,
          ask: message.ask,
        };
        callbacks.onTick.forEach((cb) => cb(tick));
        break;

      case 'candle_update':
        const candle: CandleUpdate = {
          type: 'candle_update',
          symbol: message.symbol,
          timeframe: message.timeframe,
          timestamp: message.timestamp,
          open: message.open,
          high: message.high,
          low: message.low,
          close: message.close,
          volume: message.volume,
          is_closed: message.is_closed,
        };
        callbacks.onCandle.forEach((cb) => cb(candle));
        break;

      case 'subscribed':
        console.info(`✅ Subscription confirmed: ${message.symbols?.join(', ')}`);
        break;

      case 'unsubscribed':
        console.info(`✅ Unsubscription confirmed: ${message.symbols?.join(', ')}`);
        break;

      case 'pong':
        // Heartbeat response
        break;

      case 'error':
        console.error('WebSocket server error:', message.message);
        const error = new Error(message.message || 'Server error');
        callbacks.onError.forEach((cb) => cb(error));
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  };

  /**
   * Register callback for tick updates
   */
  const onTick = (callback: (tick: TickUpdate) => void) => {
    callbacks.onTick.add(callback);
    
    // Return cleanup function
    return () => {
      callbacks.onTick.delete(callback);
    };
  };

  /**
   * Register callback for candle updates
   */
  const onCandle = (callback: (candle: CandleUpdate) => void) => {
    callbacks.onCandle.add(callback);
    
    return () => {
      callbacks.onCandle.delete(callback);
    };
  };

  /**
   * Register callback for errors
   */
  const onError = (callback: (error: Error) => void) => {
    callbacks.onError.add(callback);
    
    return () => {
      callbacks.onError.delete(callback);
    };
  };

  /**
   * Send ping to keep connection alive
   */
  const ping = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'ping' }));
    }
  };

  // Auto-connect if enabled
  if (opts.autoConnect) {
    connect();
  }

  // Cleanup on unmount
  onCleanup(() => {
    intentionalClose = true;
    disconnect();
  });

  // Heartbeat interval
  const heartbeatInterval = setInterval(() => {
    if (status() === 'connected') {
      ping();
    }
  }, 30000); // 30 seconds

  onCleanup(() => {
    clearInterval(heartbeatInterval);
  });

  return {
    status,
    error,
    subscribedSymbols,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    onTick,
    onCandle,
    onError,
    isConnected: () => status() === 'connected',
  };
}
