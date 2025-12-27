/**
 * ORDER BOOK DEPTH CHART v1.0
 * 
 * Professional-grade Level 2 market depth visualization.
 * This is a TradingView Pro / Bloomberg Terminal feature.
 * 
 * Features:
 * - Real-time bid/ask depth visualization
 * - Cumulative volume stacking
 * - Spread indicator with basis points
 * - Price levels with order counts
 * - Animated updates
 * - Imbalance detection
 * 
 * Data from /api/v1/market-data/orderbook/{symbol}
 */

import { createSignal, createEffect, Show, For, onMount, onCleanup } from 'solid-js';
import { Maximize2, Minimize2, RefreshCw } from 'lucide-solid';

interface OrderBookLevel {
  price: number;
  size: number;
  orders?: number;
}

interface OrderBookData {
  symbol: string;
  timestamp: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  spread_bps: number;
  midpoint: number;
  _simulated?: boolean;
}

interface OrderBookDepthChartProps {
  symbol: string;
  levels?: number;
  height?: string;
  onSpreadChange?: (spread: number, spreadBps: number) => void;
}

export default function OrderBookDepthChart(props: OrderBookDepthChartProps) {
  const [data, setData] = createSignal<OrderBookData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [expanded, setExpanded] = createSignal(false);

  const levels = () => props.levels || 10;

  const fetchOrderBook = async () => {
    try {
      const response = await fetch(
        `/api/v1/market-data/orderbook/${props.symbol}?levels=${levels()}`,
        { credentials: 'include' }
      );

      if (response.ok) {
        const orderBook = await response.json();
        setData(orderBook);
        
        if (props.onSpreadChange) {
          props.onSpreadChange(orderBook.spread, orderBook.spread_bps);
        }
      } else {
        setError('Failed to fetch order book');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch and refresh every 2 seconds
  onMount(() => {
    fetchOrderBook();
    const interval = setInterval(fetchOrderBook, 2000);
    onCleanup(() => clearInterval(interval));
  });

  // Refetch on symbol change
  createEffect(() => {
    props.symbol;
    setLoading(true);
    fetchOrderBook();
  });

  // Calculate max cumulative volume for scaling
  const maxCumulativeVolume = () => {
    if (!data()) return 1;
    
    let bidTotal = 0;
    let askTotal = 0;
    
    data()!.bids.forEach(level => bidTotal += level.size);
    data()!.asks.forEach(level => askTotal += level.size);
    
    return Math.max(bidTotal, askTotal);
  };

  // Calculate bid/ask imbalance
  const imbalance = () => {
    if (!data()) return 0;
    
    const bidTotal = data()!.bids.reduce((sum, l) => sum + l.size, 0);
    const askTotal = data()!.asks.reduce((sum, l) => sum + l.size, 0);
    const total = bidTotal + askTotal;
    
    if (total === 0) return 0;
    return ((bidTotal - askTotal) / total) * 100;
  };

  const imbalanceColor = () => {
    const ib = imbalance();
    if (ib > 20) return 'text-green-500';
    if (ib < -20) return 'text-red-500';
    return 'text-gray-400';
  };

  const formatSize = (size: number): string => {
    if (size >= 1000000) return `${(size / 1000000).toFixed(1)}M`;
    if (size >= 1000) return `${(size / 1000).toFixed(1)}K`;
    return size.toLocaleString();
  };

  return (
    <div 
      class="bg-terminal-900 rounded-lg border border-terminal-750 overflow-hidden"
      style={{ height: expanded() ? '400px' : props.height || '250px' }}
    >
      {/* Header */}
      <div class="flex items-center justify-between px-3 py-2 border-b border-terminal-750 bg-terminal-950/50">
        <div class="flex items-center gap-2">
          <h3 class="text-sm font-semibold text-white">Order Book</h3>
          <Show when={data()?._simulated}>
            <span 
              class="text-[10px] px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded cursor-help border border-yellow-500/30"
              title="Simulated Data: Real Level 2 order book data requires expensive exchange feeds. This visualization uses simulated data based on current spread."
            >
              SIMULATED
            </span>
          </Show>
        </div>
        
        <div class="flex items-center gap-2">
          {/* Spread Display */}
          <Show when={data()}>
            <div class="text-xs font-mono">
              <span class="text-gray-500">Spread: </span>
              <span class="text-white">${data()!.spread.toFixed(4)}</span>
              <span class="text-gray-500 ml-1">({data()!.spread_bps.toFixed(1)} bps)</span>
            </div>
          </Show>
          
          <button
            onClick={() => fetchOrderBook()}
            class="p-1 text-gray-500 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
          
          <button
            onClick={() => setExpanded(prev => !prev)}
            class="p-1 text-gray-500 hover:text-white transition-colors"
            title={expanded() ? 'Collapse' : 'Expand'}
          >
            {expanded() ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      {/* Content */}
      <div class="flex-1 overflow-hidden">
        <Show when={loading()} fallback={
          <Show when={error()} fallback={
            <Show when={data()}>
              <div class="flex h-full">
                {/* Bids Side */}
                <div class="flex-1 flex flex-col">
                  <div class="flex items-center justify-between px-2 py-1 text-[10px] text-gray-500 border-b border-terminal-750">
                    <span>BIDS</span>
                    <span>SIZE</span>
                  </div>
                  <div class="flex-1 overflow-y-auto">
                    <For each={data()!.bids}>
                      {(level, index) => {
                        // Calculate cumulative volume up to this level
                        const cumulative = data()!.bids
                          .slice(0, index() + 1)
                          .reduce((sum, l) => sum + l.size, 0);
                        const width = (cumulative / maxCumulativeVolume()) * 100;
                        
                        return (
                          <div class="relative flex items-center justify-between px-2 py-1.5 hover:bg-terminal-800/50 text-xs">
                            {/* Depth bar (right-aligned for bids) */}
                            <div 
                              class="absolute right-0 top-0 bottom-0 bg-green-500/20"
                              style={{ width: `${width}%` }}
                            />
                            {/* Price */}
                            <span class="relative z-10 font-mono text-green-400">
                              ${level.price.toFixed(2)}
                            </span>
                            {/* Size */}
                            <span class="relative z-10 font-mono text-white">
                              {formatSize(level.size)}
                            </span>
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </div>

                {/* Center - Midpoint & Imbalance */}
                <div class="w-24 flex flex-col items-center justify-center border-x border-terminal-750 bg-terminal-950/30">
                  <span class="text-[10px] text-gray-500 mb-1">MIDPOINT</span>
                  <span class="text-sm font-mono font-bold text-white">
                    ${data()!.midpoint.toFixed(2)}
                  </span>
                  
                  <div class="mt-3">
                    <span class="text-[10px] text-gray-500">IMBALANCE</span>
                    <div class={`text-sm font-mono font-bold ${imbalanceColor()}`}>
                      {imbalance() > 0 ? '+' : ''}{imbalance().toFixed(1)}%
                    </div>
                  </div>
                  
                  {/* Imbalance Bar */}
                  <div class="w-12 h-1.5 bg-terminal-750 rounded-full mt-2 overflow-hidden">
                    <div 
                      class="h-full transition-all duration-300"
                      classList={{
                        'bg-green-500': imbalance() > 0,
                        'bg-red-500': imbalance() < 0,
                        'bg-gray-500': imbalance() === 0,
                      }}
                      style={{ 
                        width: `${Math.abs(imbalance())}%`,
                        'margin-left': imbalance() >= 0 ? '50%' : `${50 + imbalance()}%`
                      }}
                    />
                  </div>
                </div>

                {/* Asks Side */}
                <div class="flex-1 flex flex-col">
                  <div class="flex items-center justify-between px-2 py-1 text-[10px] text-gray-500 border-b border-terminal-750">
                    <span>SIZE</span>
                    <span>ASKS</span>
                  </div>
                  <div class="flex-1 overflow-y-auto">
                    <For each={data()!.asks}>
                      {(level, index) => {
                        // Calculate cumulative volume up to this level
                        const cumulative = data()!.asks
                          .slice(0, index() + 1)
                          .reduce((sum, l) => sum + l.size, 0);
                        const width = (cumulative / maxCumulativeVolume()) * 100;
                        
                        return (
                          <div class="relative flex items-center justify-between px-2 py-1.5 hover:bg-terminal-800/50 text-xs">
                            {/* Depth bar (left-aligned for asks) */}
                            <div 
                              class="absolute left-0 top-0 bottom-0 bg-red-500/20"
                              style={{ width: `${width}%` }}
                            />
                            {/* Size */}
                            <span class="relative z-10 font-mono text-white">
                              {formatSize(level.size)}
                            </span>
                            {/* Price */}
                            <span class="relative z-10 font-mono text-red-400">
                              ${level.price.toFixed(2)}
                            </span>
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </div>
              </div>
            </Show>
          }>
            <div class="flex items-center justify-center h-full text-sm text-red-400">
              {error()}
            </div>
          </Show>
        }>
          <div class="flex items-center justify-center h-full">
            <div class="animate-spin w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full" />
          </div>
        </Show>
      </div>
    </div>
  );
}
