/**
 * OrderBook Component
 * 
 * Level 2 order book visualization showing bid/ask depth.
 * Professional trading essential for price action analysis.
 * 
 * Design System: Bloomberg Terminal / IBKR style
 */

import { For, Show } from 'solid-js';
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
  precision?: number;
  sizePrecision?: number;
  onPriceClick?: (price: number, side: 'bid' | 'ask') => void;
  className?: string;
}

export function OrderBook(props: OrderBookProps) {
  const maxLevels = () => props.maxLevels || 10;
  const precision = () => props.precision || 2;
  const sizePrecision = () => props.sizePrecision || 0;
  
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
  
  // Format size with abbreviation
  const formatSize = (size: number) => {
    if (size >= 1000000) return `${(size / 1000000).toFixed(1)}M`;
    if (size >= 1000) return `${(size / 1000).toFixed(1)}K`;
    return size.toFixed(sizePrecision());
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
      
      {/* Column Headers */}
      <div class="grid grid-cols-4 px-3 py-1.5 text-[10px] font-mono text-gray-500 uppercase border-b border-terminal-800">
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
              class="relative grid grid-cols-4 px-3 py-1 text-xs font-mono hover:bg-terminal-850 cursor-pointer transition-colors"
              onClick={() => props.onPriceClick?.(level.price, 'ask')}
            >
              {/* Depth bar background */}
              <Show when={props.showDepthBars !== false}>
                <div
                  class="absolute inset-y-0 right-0 bg-danger-500/10 transition-all duration-300"
                  style={{ width: `${(level.size / maxSize()) * 100}%` }}
                />
              </Show>
              
              <span class="relative z-10 text-danger-400 tabular-nums">
                {level.price.toFixed(precision())}
              </span>
              <span class="relative z-10 text-right text-gray-300 tabular-nums">
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
              class="relative grid grid-cols-4 px-3 py-1 text-xs font-mono hover:bg-terminal-850 cursor-pointer transition-colors"
              onClick={() => props.onPriceClick?.(level.price, 'bid')}
            >
              {/* Depth bar background */}
              <Show when={props.showDepthBars !== false}>
                <div
                  class="absolute inset-y-0 left-0 bg-success-500/10 transition-all duration-300"
                  style={{ width: `${(level.size / maxSize()) * 100}%` }}
                />
              </Show>
              
              <span class="relative z-10 text-success-400 tabular-nums">
                {level.price.toFixed(precision())}
              </span>
              <span class="relative z-10 text-right text-gray-300 tabular-nums">
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
