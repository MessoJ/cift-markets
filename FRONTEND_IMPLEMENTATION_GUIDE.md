# CIFT Markets - Frontend Implementation Guide (Phase 2)

**Ready to Start**: Frontend with SolidJS  
**Date**: 2025-01-08  
**Tech Stack**: SolidJS + TailwindCSS + shadcn/ui + Vite

---

## 🎯 Objectives

Build a **blazing-fast trading dashboard** using:
- **SolidJS** - 8x faster than React
- **TailwindCSS** - Utility-first CSS
- **shadcn/ui** - Beautiful component library
- **Vite** - Lightning-fast dev server
- **WebSocket** - Real-time market data
- **TanStack Query** - Server state management

---

## 📊 Architecture Overview

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── charts/          # Trading charts
│   │   ├── trading/         # Order entry, positions
│   │   └── layout/          # Nav, sidebar, header
│   ├── pages/
│   │   ├── Dashboard.tsx    # Main trading view
│   │   ├── Login.tsx        # Authentication
│   │   ├── Portfolio.tsx    # Positions & P&L
│   │   └── History.tsx      # Order history
│   ├── hooks/
│   │   ├── useWebSocket.ts  # Real-time data
│   │   ├── useAuth.ts       # Authentication
│   │   └── useMarketData.ts # Market data queries
│   ├── lib/
│   │   ├── api.ts           # API client
│   │   ├── websocket.ts     # WebSocket manager
│   │   └── auth.ts          # Auth utilities
│   ├── stores/
│   │   ├── authStore.ts     # Auth state
│   │   ├── marketStore.ts   # Market data state
│   │   └── portfolioStore.ts # Portfolio state
│   ├── types/
│   │   └── api.ts           # TypeScript types
│   ├── App.tsx
│   └── main.tsx
├── public/
├── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── README.md
```

---

## 🚀 Project Setup

### **1. Initialize SolidJS Project**

```bash
cd frontend

# Create Vite + SolidJS project
npm create vite@latest . -- --template solid-ts

# Install dependencies
npm install

# Install TailwindCSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Install UI libraries
npm install @kobalte/core clsx tailwind-merge
npm install lucide-solid  # Icons (Lucide)

# Install state management & utilities
npm install @tanstack/solid-query
npm install solid-router
npm install zod  # Schema validation
```

### **2. Configure TailwindCSS**

**`tailwind.config.js`**:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        // Add more colors...
      },
    },
  },
  plugins: [],
}
```

### **3. Setup API Client**

**`src/lib/api.ts`**:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

class ApiClient {
  private token: string | null = null;

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  getToken(): string | null {
    return this.token || localStorage.getItem('access_token');
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('access_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const token = this.getToken();
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }

  // Authentication
  async login(email: string, password: string) {
    return this.request<{ access_token: string; refresh_token: string }>(
      '/auth/login',
      {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      }
    );
  }

  async register(email: string, username: string, password: string) {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, username, password }),
    });
  }

  async getMe() {
    return this.request('/auth/me');
  }

  // Market Data
  async getQuote(symbol: string) {
    return this.request(`/market-data/quote/${symbol}`);
  }

  async getQuotes(symbols: string[]) {
    return this.request(`/market-data/quotes?symbols=${symbols.join(',')}`);
  }

  // Trading
  async submitOrder(order: OrderRequest) {
    return this.request('/trading/orders', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  async getPositions() {
    return this.request('/trading/positions');
  }

  async getPortfolio() {
    return this.request('/trading/portfolio');
  }
}

export const api = new ApiClient();
```

### **4. WebSocket Manager**

**`src/lib/websocket.ts`**:
```typescript
type MessageHandler = (data: any) => void;

class WebSocketManager {
  private ws: WebSocket | null = null;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect() {
    const wsUrl = 'ws://localhost:8000/api/v1/market-data/ws/stream';
    
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const handlers = this.handlers.get(data.type);
      
      if (handlers) {
        handlers.forEach(handler => handler(data));
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.reconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => this.connect(), 1000 * this.reconnectAttempts);
    }
  }

  subscribe(symbols: string[]) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        action: 'subscribe',
        symbols,
      }));
    }
  }

  on(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: string, handler: MessageHandler) {
    this.handlers.get(type)?.delete(handler);
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }
}

export const wsManager = new WebSocketManager();
```

---

## 🎨 UI Components

### **Key Components to Build**

1. **Order Entry Panel**
   - Symbol input with autocomplete
   - Order type selector (market/limit)
   - Quantity input
   - Price input (for limit orders)
   - Buy/Sell buttons (green/red)
   - Risk validation display

2. **Position Table**
   - Symbol, Quantity, Avg Cost, Current Price
   - Unrealized P&L (with color coding)
   - % Change
   - Actions (close position)

3. **Order Book** (Level 2)
   - Bid/Ask ladder
   - Volume bars
   - Spread indicator
   - Real-time updates via WebSocket

4. **Price Chart**
   - Candlestick/line chart
   - Volume bars
   - Technical indicators overlay
   - Multiple timeframes

5. **Portfolio Summary Cards**
   - Total Value
   - Cash Available
   - Buying Power
   - Day P&L
   - Total P&L

---

## 📱 Pages/Views

### **1. Login Page**
- Email/password form
- JWT token storage
- Redirect to dashboard

### **2. Dashboard (Main View)**
- **Top Bar**: Portfolio summary cards
- **Left Panel**: Watchlist + order entry
- **Center**: Price chart + order book
- **Right Panel**: Positions table
- **Bottom**: Order history/fills

### **3. Portfolio Page**
- Detailed position list
- P&L charts (day/week/month/year)
- Performance metrics
- Transaction history

### **4. History Page**
- Order history table
- Fill details
- Filters (symbol, date, status)
- Export functionality

---

## 🔌 Real-Time Features

### **WebSocket Integration**

```typescript
// In Dashboard component
import { createEffect, onCleanup } from 'solid-js';
import { wsManager } from '@/lib/websocket';

export function Dashboard() {
  createEffect(() => {
    // Connect WebSocket
    wsManager.connect();
    
    // Subscribe to symbols
    wsManager.subscribe(['AAPL', 'GOOGL', 'MSFT']);
    
    // Handle price updates
    const handlePrice = (data: any) => {
      console.log('Price update:', data);
      // Update store/state
    };
    
    wsManager.on('price', handlePrice);
    
    // Cleanup
    onCleanup(() => {
      wsManager.off('price', handlePrice);
      wsManager.disconnect();
    });
  });

  return (
    <div>Dashboard</div>
  );
}
```

---

## 🎨 Design System

### **Color Scheme**
- **Background**: Dark theme (hsl(222.2 84% 4.9%))
- **Primary**: Blue (hsl(221.2 83.2% 53.3%))
- **Success**: Green (#10b981) - Profit, Buy button
- **Danger**: Red (#ef4444) - Loss, Sell button
- **Warning**: Yellow (#f59e0b) - Alerts
- **Muted**: Gray for secondary text

### **Typography**
- **Font**: Inter or Geist Sans
- **Headings**: Bold, larger sizes
- **Data**: Monospace for prices/quantities
- **Body**: Regular weight, 14-16px

### **Components Style**
- Rounded corners (border-radius: 0.5rem)
- Subtle shadows
- Smooth transitions (duration-200)
- Hover states on interactive elements

---

## 📊 State Management

### **Auth Store**
```typescript
import { createSignal, createEffect } from 'solid-js';

const [user, setUser] = createSignal(null);
const [isAuthenticated, setIsAuthenticated] = createSignal(false);

export function useAuth() {
  return {
    user,
    isAuthenticated,
    login: async (email, password) => {
      const data = await api.login(email, password);
      api.setToken(data.access_token);
      const userData = await api.getMe();
      setUser(userData);
      setIsAuthenticated(true);
    },
    logout: () => {
      api.clearToken();
      setUser(null);
      setIsAuthenticated(false);
    },
  };
}
```

### **Market Data Store**
```typescript
import { createSignal, createEffect } from 'solid-js';

const [prices, setPrices] = createSignal<Map<string, number>>(new Map());

export function useMarketData() {
  createEffect(() => {
    wsManager.on('price', (data) => {
      setPrices((prev) => {
        const updated = new Map(prev);
        updated.set(data.symbol, data.price);
        return updated;
      });
    });
  });

  return { prices };
}
```

---

## 🚀 Performance Optimizations

1. **Code Splitting**
   - Lazy load routes
   - Dynamic imports for heavy components

2. **Memoization**
   - Use `createMemo` for derived state
   - Avoid unnecessary recalculations

3. **Virtual Scrolling**
   - For large lists (order history)
   - Use `@tanstack/solid-virtual`

4. **Debouncing**
   - Search inputs
   - Real-time validation

5. **WebSocket Throttling**
   - Limit update frequency (60fps max)
   - Batch updates

---

## 📦 Package.json Example

```json
{
  "name": "cift-markets-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "solid-js": "^1.8.0",
    "@solidjs/router": "^0.13.0",
    "@tanstack/solid-query": "^5.0.0",
    "@kobalte/core": "^0.13.0",
    "lucide-solid": "^0.300.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "vite-plugin-solid": "^2.10.0",
    "typescript": "^5.3.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
```

---

## ✅ Implementation Checklist

### **Setup Phase**
- [ ] Initialize Vite + SolidJS project
- [ ] Configure TailwindCSS
- [ ] Setup shadcn/ui components
- [ ] Create API client
- [ ] Setup WebSocket manager
- [ ] Configure routing

### **Core Features**
- [ ] Login/register pages
- [ ] Protected routes
- [ ] Dashboard layout
- [ ] Order entry form
- [ ] Position table
- [ ] Portfolio summary
- [ ] WebSocket integration

### **Advanced Features**
- [ ] Price charts (TradingView)
- [ ] Order book visualization
- [ ] Real-time P&L updates
- [ ] Order history table
- [ ] Performance analytics

### **Polish**
- [ ] Dark/light theme toggle
- [ ] Responsive design
- [ ] Loading states
- [ ] Error handling
- [ ] Toast notifications

---

## 🎓 SolidJS Best Practices

1. **Use Signals for Reactive State**
   ```typescript
   const [count, setCount] = createSignal(0);
   ```

2. **Memoize Derived Values**
   ```typescript
   const doubled = createMemo(() => count() * 2);
   ```

3. **Effects for Side Effects**
   ```typescript
   createEffect(() => {
     console.log('Count changed:', count());
   });
   ```

4. **Cleanup in onCleanup**
   ```typescript
   onCleanup(() => {
     // Cleanup subscriptions
   });
   ```

5. **Use Stores for Complex State**
   ```typescript
   const [store, setStore] = createStore({ ... });
   ```

---

**Status**: ✅ **READY TO START FRONTEND IMPLEMENTATION**

**Performance Target**: <100ms page load, 60fps animations, <16ms state updates

**Next Session**: Begin SolidJS project setup and build core dashboard

