/**
 * OrderBook Component
 * 
 * Level 2 order book visualization showing bid/ask depth.
 * Professional trading essential for price action analysis.
 * 
 * Design System: Bloomberg Terminal / IBKR style
 * 
 * Features:
 * - Real-time depth bars with cumulative visualization
 * - Order imbalance indicator (bid vs ask pressure)
 * - Large order highlighting (whale detection)
 * - Click-to-trade integration
 * - Animated price updates
 */

import { For, Show, createMemo } from 'solid-js';
import { formatCurrency } from '~/lib/utils/format';

export interface OrderBookLevel {
  price: number;
  size: number;
  total?: number;
  orders?: number;
}

export interface OrderBookData {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread?: number;
  spreadPercent?: number;
  midPrice?: number;
  lastUpdate?: number;
}

interface OrderBookProps {
  data: OrderBookData | null;
  maxLevels?: number;
  showDepthBars?: boolean;
  showOrders?: boolean;
  showImbalance?: boolean;
  highlightLargeOrders?: boolean;
  largeOrderThreshold?: number; // In shares - orders above this highlighted
  precision?: number;
  sizePrecision?: number;
  onPriceClick?: (price: number, side: 'bid' | 'ask') => void;
  className?: string;
}

export function OrderBook(props: OrderBookProps) {
  const maxLevels = () => props.maxLevels || 10;
  const precision = () => props.precision || 2;
  const sizePrecision = () => props.sizePrecision || 0;
  const largeOrderThreshold = () => props.largeOrderThreshold || 10000;
  
  // Calculate max size for depth bars
  const maxBidSize = () => {
    const bids = props.data?.bids || [];
    return Math.max(...bids.slice(0, maxLevels()).map(b => b.size), 1);
  };
  
  const maxAskSize = () => {
    const asks = props.data?.asks || [];
    return Math.max(...asks.slice(0, maxLevels()).map(a => a.size), 1);
  };
  
  const maxSize = () => Math.max(maxBidSize(), maxAskSize());
  
  // Calculate total cumulative size for each side
  const totalBidSize = createMemo(() => {
    return (props.data?.bids || []).slice(0, maxLevels()).reduce((sum, b) => sum + b.size, 0);
  });
  
  const totalAskSize = createMemo(() => {
    return (props.data?.asks || []).slice(0, maxLevels()).reduce((sum, a) => sum + a.size, 0);
  });
  
  // Order imbalance: positive = more buying pressure, negative = more selling pressure
  const imbalance = createMemo(() => {
    const bidTotal = totalBidSize();
    const askTotal = totalAskSize();
    const total = bidTotal + askTotal;
    if (total === 0) return 0;
    return ((bidTotal - askTotal) / total) * 100; // Returns -100 to +100
  });
  
  // Format size with abbreviation
  const formatSize = (size: number) => {
    if (size >= 1000000) return `${(size / 1000000).toFixed(1)}M`;
    if (size >= 1000) return `${(size / 1000).toFixed(1)}K`;
    return size.toFixed(sizePrecision());
  };
  
  // Check if order is large (whale)
  const isLargeOrder = (size: number) => {
    return props.highlightLargeOrders !== false && size >= largeOrderThreshold();
  };

  return (
    <div class={`flex flex-col bg-terminal-900 border border-terminal-750 ${props.className || ''}`}>
      {/* Header */}
      <div class="flex items-center justify-between px-3 py-2 border-b border-terminal-750 bg-terminal-850">
        <h3 class="text-xs font-mono font-bold text-gray-400 uppercase">Order Book</h3>
        <Show when={props.data?.spread !== undefined}>
          <div class="flex items-center gap-2 text-[10px] font-mono">
            <span class="text-gray-500">Spread:</span>
            <span class="text-white tabular-nums">
              {formatCurrency(props.data!.spread!)}
            </span>
            <span class="text-gray-500">
              ({props.data!.spreadPercent?.toFixed(3)}%)
            </span>
          </div>
        </Show>
      </div>
      
      {/* Order Imbalance Indicator */}
      <Show when={props.showImbalance !== false && props.data}>
        <div class="px-3 py-1.5 border-b border-terminal-750 bg-terminal-900">
          <div class="flex items-center justify-between text-[10px] font-mono mb-1">
            <span class="text-success-500">BID {totalBidSize().toLocaleString()}</span>
            <span class={`font-bold ${imbalance() > 10 ? 'text-success-400' : imbalance() < -10 ? 'text-danger-400' : 'text-gray-400'}`}>
              {imbalance() > 0 ? '+' : ''}{imbalance().toFixed(1)}%
            </span>
            <span class="text-danger-500">ASK {totalAskSize().toLocaleString()}</span>
          </div>
          {/* Imbalance bar */}
          <div class="h-1.5 bg-terminal-800 rounded-full overflow-hidden flex">
            <div 
              class="bg-success-500 transition-all duration-300"
              style={{ width: `${50 + imbalance() / 2}%` }}
            />
            <div 
              class="bg-danger-500 transition-all duration-300 flex-1"
            />
          </div>
        </div>
      </Show>
      
      {/* Column Headers */}
      <div class={`grid px-3 py-1.5 text-[10px] font-mono text-gray-500 uppercase border-b border-terminal-800 ${props.showOrders ? 'grid-cols-4' : 'grid-cols-3'}`}>
        <span class="text-left">Price</span>
        <span class="text-right">Size</span>
        <span class="text-right">Total</span>
        <Show when={props.showOrders}>
          <span class="text-right">#</span>
        </Show>
      </div>
      
      {/* Asks (Sell orders) - reversed so best ask is at bottom */}
      <div class="flex flex-col-reverse">
        <For each={(props.data?.asks || []).slice(0, maxLevels())}>
          {(level) => (
            <div
              class={`relative grid px-3 py-1 text-xs font-mono cursor-pointer transition-all group ${
                isLargeOrder(level.size) 
                  ? 'bg-danger-900/30 hover:bg-danger-900/50 border-l-2 border-danger-500' 
                  : 'hover:bg-terminal-850'
              } ${props.showOrders ? 'grid-cols-4' : 'grid-cols-3'}`}
              onClick={() => props.onPriceClick?.(level.price, 'ask')}
            >
              {/* Depth bar background */}
              <Show when={props.showDepthBars !== false}>
                <div
                  class="absolute inset-y-0 right-0 bg-danger-500/10 transition-all duration-300"
                  style={{ width: `${(level.size / maxSize()) * 100}%` }}
                />
              </Show>
              
              <span class={`relative z-10 tabular-nums transition-colors ${isLargeOrder(level.size) ? 'text-danger-300 font-bold' : 'text-danger-400'}`}>
                {level.price.toFixed(precision())}
                {/* Large order indicator */}
                <Show when={isLargeOrder(level.size)}>
                  <span class="ml-1 text-[8px] text-danger-300 animate-pulse">●</span>
                </Show>
              </span>
              <span class={`relative z-10 text-right tabular-nums ${isLargeOrder(level.size) ? 'text-white font-bold' : 'text-gray-300'}`}>
                {formatSize(level.size)}
              </span>
              <span class="relative z-10 text-right text-gray-500 tabular-nums">
                {formatSize(level.total || level.size)}
              </span>
              <Show when={props.showOrders}>
                <span class="relative z-10 text-right text-gray-600 tabular-nums">
                  {level.orders || 1}
                </span>
              </Show>
              
              {/* Hover tooltip */}
              <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 pointer-events-none z-50 transition-opacity">
                <div class="bg-terminal-950 border border-terminal-600 rounded px-2 py-1 text-[10px] font-mono whitespace-nowrap shadow-xl">
                  <span class="text-danger-400">Click to SELL @ {formatCurrency(level.price)}</span>
                </div>
              </div>
            </div>
          )}
        </For>
      </div>
      
      {/* Spread indicator */}
      <div class="flex items-center justify-center py-2 bg-terminal-850 border-y border-terminal-750">
        <Show when={props.data?.midPrice !== undefined}>
          <div class="flex items-center gap-3 text-xs font-mono">
            <span class="text-gray-500">Mid:</span>
            <span class="text-white font-bold tabular-nums">
              {formatCurrency(props.data!.midPrice!)}
            </span>
          </div>
        </Show>
        <Show when={!props.data?.midPrice && !props.data}>
          <span class="text-xs text-gray-600 font-mono">Loading...</span>
        </Show>
      </div>
      
      {/* Bids (Buy orders) */}
      <div class="flex flex-col">
        <For each={(props.data?.bids || []).slice(0, maxLevels())}>
          {(level) => (
            <div
              class={`relative grid px-3 py-1 text-xs font-mono cursor-pointer transition-all group ${
                isLargeOrder(level.size) 
                  ? 'bg-success-900/30 hover:bg-success-900/50 border-l-2 border-success-500' 
                  : 'hover:bg-terminal-850'
              } ${props.showOrders ? 'grid-cols-4' : 'grid-cols-3'}`}
              onClick={() => props.onPriceClick?.(level.price, 'bid')}
            >
              {/* Depth bar background */}
              <Show when={props.showDepthBars !== false}>
                <div
                  class="absolute inset-y-0 left-0 bg-success-500/10 transition-all duration-300"
                  style={{ width: `${(level.size / maxSize()) * 100}%` }}
                />
              </Show>
              
              <span class={`relative z-10 tabular-nums transition-colors ${isLargeOrder(level.size) ? 'text-success-300 font-bold' : 'text-success-400'}`}>
                {level.price.toFixed(precision())}
                {/* Large order indicator */}
                <Show when={isLargeOrder(level.size)}>
                  <span class="ml-1 text-[8px] text-success-300 animate-pulse">●</span>
                </Show>
              </span>
              <span class={`relative z-10 text-right tabular-nums ${isLargeOrder(level.size) ? 'text-white font-bold' : 'text-gray-300'}`}>
                {formatSize(level.size)}
              </span>
              <span class="relative z-10 text-right text-gray-500 tabular-nums">
                {formatSize(level.total || level.size)}
              </span>
              <Show when={props.showOrders}>
                <span class="relative z-10 text-right text-gray-600 tabular-nums">
                  {level.orders || 1}
                </span>
              </Show>
              
              {/* Hover tooltip */}
              <div class="absolute left-full ml-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 pointer-events-none z-50 transition-opacity">
                <div class="bg-terminal-950 border border-terminal-600 rounded px-2 py-1 text-[10px] font-mono whitespace-nowrap shadow-xl">
                  <span class="text-success-400">Click to BUY @ {formatCurrency(level.price)}</span>
                </div>
              </div>
            </div>
          )}
        </For>
      </div>
      
      {/* Empty state */}
      <Show when={!props.data || (props.data.bids.length === 0 && props.data.asks.length === 0)}>
        <div class="flex items-center justify-center py-8">
          <span class="text-xs text-gray-600 font-mono">No order book data</span>
        </div>
      </Show>
    </div>
  );
}

/**
 * CompactOrderBook - Minimal version for sidebars
 */
interface CompactOrderBookProps {
  data: OrderBookData | null;
  levels?: number;
  onPriceClick?: (price: number) => void;
  className?: string;
}

export function CompactOrderBook(props: CompactOrderBookProps) {
  const levels = () => props.levels || 5;
  
  return (
    <div class={`bg-terminal-900 border border-terminal-750 ${props.className || ''}`}>
      <div class="px-3 py-2 border-b border-terminal-750">
        <span class="text-xs font-mono text-gray-400 uppercase">L2 Depth</span>
      </div>
      
      <div class="p-2">
        {/* Asks */}
        <div class="flex flex-col-reverse mb-1">
          <For each={(props.data?.asks || []).slice(0, levels())}>
            {(level) => (
              <div 
                class="flex justify-between text-[10px] font-mono py-0.5 cursor-pointer hover:bg-terminal-850"
                onClick={() => props.onPriceClick?.(level.price)}
              >
                <span class="text-danger-400 tabular-nums">{level.price.toFixed(2)}</span>
                <span class="text-gray-500 tabular-nums">{level.size.toLocaleString()}</span>
              </div>
            )}
          </For>
        </div>
        
        {/* Spread */}
        <div class="flex justify-center py-1 border-y border-terminal-800 text-[10px] font-mono text-gray-500">
          {props.data?.spread?.toFixed(2) || '—'}
        </div>
        
        {/* Bids */}
        <div class="flex flex-col mt-1">
          <For each={(props.data?.bids || []).slice(0, levels())}>
            {(level) => (
              <div 
                class="flex justify-between text-[10px] font-mono py-0.5 cursor-pointer hover:bg-terminal-850"
                onClick={() => props.onPriceClick?.(level.price)}
              >
                <span class="text-success-400 tabular-nums">{level.price.toFixed(2)}</span>
                <span class="text-gray-500 tabular-nums">{level.size.toLocaleString()}</span>
              </div>
            )}
          </For>
        </div>
      </div>
    </div>
  );
}

export default OrderBook;
