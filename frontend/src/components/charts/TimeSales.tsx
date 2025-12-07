/**
 * TIME & SALES v1.0
 * 
 * Real-time trade tape showing recent executions.
 * Professional traders use this for order flow analysis.
 * 
 * Features:
 * - Real-time trade feed with color coding
 * - Size highlighting for large trades
 * - Exchange attribution
 * - Trade direction (buy/sell)
 * - Auto-scroll with pause on hover
 * 
 * Data from /api/v1/market-data/timesales/{symbol}
 */

import { createSignal, createEffect, Show, For, onMount, onCleanup } from 'solid-js';
import { ArrowUp, ArrowDown, Pause, Play, RefreshCw } from 'lucide-solid';

interface Trade {
  time: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  exchange?: string;
}

interface TimeSalesData {
  symbol: string;
  trades: Trade[];
  count: number;
  last_price: number;
  _simulated?: boolean;
}

interface TimeSalesProps {
  symbol: string;
  limit?: number;
  height?: string;
}

export default function TimeSales(props: TimeSalesProps) {
  const [data, setData] = createSignal<TimeSalesData | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [paused, setPaused] = createSignal(false);
  const [hovering, setHovering] = createSignal(false);
  
  let scrollContainer: HTMLDivElement | undefined;

  const limit = () => props.limit || 50;

  const fetchTrades = async () => {
    if (paused()) return;
    
    try {
      const response = await fetch(
        `/api/v1/market-data/timesales/${props.symbol}?limit=${limit()}`,
        { credentials: 'include' }
      );

      if (response.ok) {
        const trades = await response.json();
        setData(trades);
        
        // Auto-scroll to top unless hovering
        if (scrollContainer && !hovering()) {
          scrollContainer.scrollTop = 0;
        }
      }
    } catch (err) {
      console.error('Failed to fetch time & sales:', err);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch and refresh every 1 second
  onMount(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, 1000);
    onCleanup(() => clearInterval(interval));
  });

  // Refetch on symbol change
  createEffect(() => {
    props.symbol;
    setLoading(true);
    fetchTrades();
  });

  const formatTime = (isoTime: string): string => {
    const date = new Date(isoTime);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  const formatSize = (size: number): string => {
    if (size >= 10000) return `${(size / 1000).toFixed(1)}K`;
    return size.toLocaleString();
  };

  // Determine if trade is "large" (for highlighting)
  const isLargeTrade = (size: number): boolean => {
    return size >= 1000;
  };

  // Exchange badge colors
  const exchangeColor = (exchange?: string): string => {
    switch (exchange) {
      case 'NYSE': return 'bg-blue-500/30 text-blue-400';
      case 'NASDAQ': return 'bg-green-500/30 text-green-400';
      case 'ARCA': return 'bg-purple-500/30 text-purple-400';
      case 'BATS': return 'bg-orange-500/30 text-orange-400';
      default: return 'bg-gray-500/30 text-gray-400';
    }
  };

  return (
    <div 
      class="bg-terminal-900 rounded-lg border border-terminal-750 overflow-hidden flex flex-col"
      style={{ height: props.height || '300px' }}
    >
      {/* Header */}
      <div class="flex items-center justify-between px-3 py-2 border-b border-terminal-750 bg-terminal-950/50">
        <div class="flex items-center gap-2">
          <h3 class="text-sm font-semibold text-white">Time & Sales</h3>
          <Show when={data()?._simulated}>
            <span class="text-[10px] px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">SIM</span>
          </Show>
        </div>
        
        <div class="flex items-center gap-2">
          {/* Trade Count */}
          <span class="text-xs text-gray-500 font-mono">{data()?.count || 0} trades</span>
          
          {/* Pause/Play */}
          <button
            onClick={() => setPaused(prev => !prev)}
            class={`p-1 rounded transition-colors ${paused() ? 'bg-yellow-500/20 text-yellow-400' : 'text-gray-500 hover:text-white'}`}
            title={paused() ? 'Resume' : 'Pause'}
          >
            {paused() ? <Play size={14} /> : <Pause size={14} />}
          </button>
          
          <button
            onClick={() => { setPaused(false); fetchTrades(); }}
            class="p-1 text-gray-500 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Column Headers */}
      <div class="grid grid-cols-4 gap-2 px-3 py-1.5 text-[10px] text-gray-500 border-b border-terminal-750 bg-terminal-950/30">
        <span>TIME</span>
        <span class="text-center">PRICE</span>
        <span class="text-right">SIZE</span>
        <span class="text-right">EXCH</span>
      </div>

      {/* Trade List */}
      <div 
        ref={scrollContainer}
        class="flex-1 overflow-y-auto"
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
      >
        <Show when={!loading()} fallback={
          <div class="flex items-center justify-center h-full">
            <div class="animate-spin w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full" />
          </div>
        }>
          <Show when={data()?.trades.length}>
            <For each={data()!.trades}>
              {(trade) => (
                <div 
                  class="grid grid-cols-4 gap-2 px-3 py-1 text-xs hover:bg-terminal-800/50 transition-colors border-b border-terminal-800/30"
                  classList={{
                    'bg-green-500/5': trade.side === 'buy' && isLargeTrade(trade.size),
                    'bg-red-500/5': trade.side === 'sell' && isLargeTrade(trade.size),
                  }}
                >
                  {/* Time */}
                  <span class="text-gray-400 font-mono">
                    {formatTime(trade.time)}
                  </span>
                  
                  {/* Price with direction arrow */}
                  <div class="flex items-center justify-center gap-1">
                    <span class={`font-mono font-medium ${trade.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                      {trade.price.toFixed(2)}
                    </span>
                    {trade.side === 'buy' ? (
                      <ArrowUp size={10} class="text-green-500" />
                    ) : (
                      <ArrowDown size={10} class="text-red-500" />
                    )}
                  </div>
                  
                  {/* Size */}
                  <span 
                    class="text-right font-mono"
                    classList={{
                      'text-white font-bold': isLargeTrade(trade.size),
                      'text-gray-300': !isLargeTrade(trade.size),
                    }}
                  >
                    {formatSize(trade.size)}
                  </span>
                  
                  {/* Exchange */}
                  <div class="text-right">
                    <Show when={trade.exchange}>
                      <span class={`text-[9px] px-1 py-0.5 rounded ${exchangeColor(trade.exchange)}`}>
                        {trade.exchange}
                      </span>
                    </Show>
                  </div>
                </div>
              )}
            </For>
          </Show>
        </Show>
      </div>

      {/* Footer Stats */}
      <Show when={data()}>
        <div class="flex items-center justify-between px-3 py-1.5 border-t border-terminal-750 bg-terminal-950/30 text-[10px]">
          <span class="text-gray-500">
            Last: <span class="text-white font-mono">${data()!.last_price.toFixed(2)}</span>
          </span>
          <div class="flex items-center gap-3">
            <span class="text-green-400">
              <ArrowUp size={10} class="inline mr-0.5" />
              {data()!.trades.filter(t => t.side === 'buy').length}
            </span>
            <span class="text-red-400">
              <ArrowDown size={10} class="inline mr-0.5" />
              {data()!.trades.filter(t => t.side === 'sell').length}
            </span>
          </div>
        </div>
      </Show>
    </div>
  );
}
