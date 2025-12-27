import { createStore, produce } from "solid-js/store";

// Types
export interface MarketTicker {
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  change?: number;
  changePercent?: number;
  volume?: number;
  timestamp: string;
}

interface MarketDataState {
  tickers: Record<string, MarketTicker>;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  subscriptions: string[];
}

// Store
const [state, setState] = createStore<MarketDataState>({
  tickers: {},
  connectionStatus: 'disconnected',
  subscriptions: []
});

// WebSocket instance
let socket: WebSocket | null = null;
let reconnectTimer: any = null;
let heartbeatTimer: any = null;

export const marketStore = {
  // Getters
  getTicker: (symbol: string) => state.tickers[symbol],
  getAllTickers: () => state.tickers,
  getStatus: () => state.connectionStatus,
  
  // Actions
  connect: () => {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;

    setState('connectionStatus', 'connecting');
    
    try {
      // Construct WebSocket URL (handle relative paths if needed)
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const url = `${protocol}//${host}/api/v1/market-data/ws/stream`;
      
      socket = new WebSocket(url);

      socket.onopen = () => {
        console.log('[MarketData] Connected');
        setState('connectionStatus', 'connected');
        
        // Resubscribe to previous symbols
        if (state.subscriptions.length > 0) {
          marketStore.subscribe(state.subscriptions);
        }
        
        // Start heartbeat
        startHeartbeat();
      };

      socket.onclose = () => {
        console.log('[MarketData] Disconnected');
        setState('connectionStatus', 'disconnected');
        stopHeartbeat();
        
        // Auto-reconnect
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => marketStore.connect(), 5000);
      };

      socket.onerror = (err) => {
        console.error('[MarketData] Error:', err);
        socket?.close();
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'price' || message.type === 'tick') {
            const { symbol, price, bid, ask, change, change_pct, volume, timestamp } = message;
            
            setState(produce((s) => {
              // Update or create ticker
              if (!s.tickers[symbol]) {
                s.tickers[symbol] = { symbol, price: 0, timestamp: '' };
              }
              
              const ticker = s.tickers[symbol];
              ticker.price = price;
              if (bid) ticker.bid = bid;
              if (ask) ticker.ask = ask;
              if (change) ticker.change = change;
              if (change_pct) ticker.changePercent = change_pct;
              if (volume) ticker.volume = volume;
              ticker.timestamp = timestamp;
            }));
          }
        } catch (e) {
          console.warn('[MarketData] Failed to parse message', e);
        }
      };
    } catch (e) {
      console.error('[MarketData] Connection failed', e);
      setState('connectionStatus', 'disconnected');
    }
  },

  disconnect: () => {
    if (socket) {
      socket.close();
      socket = null;
    }
    stopHeartbeat();
  },

  subscribe: (symbols: string[]) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    
    // Add to local state
    setState(produce(s => {
      symbols.forEach(sym => {
        if (!s.subscriptions.includes(sym)) s.subscriptions.push(sym);
      });
    }));

    // Send to server
    socket.send(JSON.stringify({
      action: 'subscribe',
      symbols
    }));
  },

  unsubscribe: (symbols: string[]) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    // Remove from local state
    setState(produce(s => {
      s.subscriptions = s.subscriptions.filter(sub => !symbols.includes(sub));
    }));

    // Send to server
    socket.send(JSON.stringify({
      action: 'unsubscribe',
      symbols
    }));
  }
};

function startHeartbeat() {
  stopHeartbeat();
  heartbeatTimer = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'ping' }));
    }
  }, 30000);
}

function stopHeartbeat() {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
}
