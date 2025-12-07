/**
 * Time & Sales Component
 * 
 * Real-time trade tape showing recent executions.
 * Essential for scalping and reading price action.
 * 
 * Design System: Bloomberg Terminal style
 */

import { For, Show } from 'solid-js';
import { formatCurrency } from '~/lib/utils/format';

export interface TradeExecution {
  id: string;
  timestamp: number | string;
  price: number;
  size: number;
  side: 'buy' | 'sell' | 'unknown';
  exchange?: string;
}

interface TimeSalesProps {
  trades: TradeExecution[];
  maxTrades?: number;
  showExchange?: boolean;
  highlightLarge?: number; // Size threshold for highlighting
  onTradeClick?: (trade: TradeExecution) => void;
  className?: string;
}

export function TimeSales(props: TimeSalesProps) {
  const maxTrades = () => props.maxTrades || 50;
  const highlightThreshold = () => props.highlightLarge || 1000;
  
  const formatTime = (timestamp: number | string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };
  
  const formatSize = (size: number) => {
    if (size >= 1000000) return `${(size / 1000000).toFixed(1)}M`;
    if (size >= 1000) return `${(size / 1000).toFixed(1)}K`;
    return size.toString();
  };
  
  const getSideColor = (side: string) => {
    switch (side) {
      case 'buy': return 'text-success-400';
      case 'sell': return 'text-danger-400';
      default: return 'text-gray-400';
    }
  };
  
  const getSideIcon = (side: string) => {
    switch (side) {
      case 'buy': return '↑';
      case 'sell': return '↓';
      default: return '•';
    }
  };

  return (
    <div class={`flex flex-col bg-terminal-900 border border-terminal-750 ${props.className || ''}`}>
      {/* Header */}
      <div class="flex items-center justify-between px-3 py-2 border-b border-terminal-750 bg-terminal-850">
        <h3 class="text-xs font-mono font-bold text-gray-400 uppercase">Time & Sales</h3>
        <span class="text-[10px] font-mono text-gray-600">
          {props.trades.length} trades
        </span>
      </div>
      
      {/* Column Headers */}
      <div class={`grid px-3 py-1.5 text-[10px] font-mono text-gray-500 uppercase border-b border-terminal-800 ${props.showExchange ? 'grid-cols-5' : 'grid-cols-4'}`}>
        <span>Time</span>
        <span class="text-right">Price</span>
        <span class="text-right">Size</span>
        <span class="text-center">Side</span>
        <Show when={props.showExchange}>
          <span class="text-right">Exch</span>
        </Show>
      </div>
      
      {/* Trades list */}
      <div class="flex-1 overflow-y-auto max-h-80">
        <For each={props.trades.slice(0, maxTrades())}>
          {(trade, index) => {
            const isLarge = () => trade.size >= highlightThreshold();
            const isNew = () => index() < 3;
            
            return (
              <div
                class={`grid px-3 py-1 text-xs font-mono cursor-pointer transition-all duration-300
                  ${props.showExchange ? 'grid-cols-5' : 'grid-cols-4'}
                  ${isLarge() ? 'bg-accent-500/10' : ''}
                  ${isNew() ? 'animate-flash' : ''}
                  hover:bg-terminal-850`}
                onClick={() => props.onTradeClick?.(trade)}
              >
                <span class="text-gray-500 tabular-nums">
                  {formatTime(trade.timestamp)}
                </span>
                <span class={`text-right tabular-nums ${getSideColor(trade.side)}`}>
                  {formatCurrency(trade.price)}
                </span>
                <span class={`text-right tabular-nums ${isLarge() ? 'text-accent-400 font-semibold' : 'text-gray-300'}`}>
                  {formatSize(trade.size)}
                </span>
                <span class={`text-center ${getSideColor(trade.side)}`}>
                  {getSideIcon(trade.side)}
                </span>
                <Show when={props.showExchange}>
                  <span class="text-right text-gray-600 uppercase text-[10px]">
                    {trade.exchange || 'NYSE'}
                  </span>
                </Show>
              </div>
            );
          }}
        </For>
        
        {/* Empty state */}
        <Show when={props.trades.length === 0}>
          <div class="flex items-center justify-center py-8">
            <span class="text-xs text-gray-600 font-mono">No trades yet</span>
          </div>
        </Show>
      </div>
      
      {/* Footer with stats */}
      <Show when={props.trades.length > 0}>
        <div class="flex items-center justify-between px-3 py-2 border-t border-terminal-750 bg-terminal-850">
          <div class="flex items-center gap-4 text-[10px] font-mono">
            <span class="text-gray-500">
              Buys: <span class="text-success-400">{props.trades.filter(t => t.side === 'buy').length}</span>
            </span>
            <span class="text-gray-500">
              Sells: <span class="text-danger-400">{props.trades.filter(t => t.side === 'sell').length}</span>
            </span>
          </div>
          <div class="text-[10px] font-mono text-gray-500">
            Vol: {formatSize(props.trades.reduce((sum, t) => sum + t.size, 0))}
          </div>
        </div>
      </Show>
    </div>
  );
}

/**
 * CompactTimeSales - Minimal version for sidebars
 */
interface CompactTimeSalesProps {
  trades: TradeExecution[];
  maxTrades?: number;
  className?: string;
}

export function CompactTimeSales(props: CompactTimeSalesProps) {
  const maxTrades = () => props.maxTrades || 10;
  
  const formatTime = (timestamp: number | string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div class={`bg-terminal-900 border border-terminal-750 ${props.className || ''}`}>
      <div class="px-3 py-2 border-b border-terminal-750">
        <span class="text-xs font-mono text-gray-400 uppercase">Recent Trades</span>
      </div>
      
      <div class="p-2 space-y-0.5 max-h-48 overflow-y-auto">
        <For each={props.trades.slice(0, maxTrades())}>
          {(trade) => (
            <div class="flex items-center justify-between text-[10px] font-mono py-0.5">
              <span class={trade.side === 'buy' ? 'text-success-400' : 'text-danger-400'}>
                {trade.side === 'buy' ? '↑' : '↓'} {trade.price.toFixed(2)}
              </span>
              <span class="text-gray-500">{trade.size.toLocaleString()}</span>
              <span class="text-gray-600">{formatTime(trade.timestamp)}</span>
            </div>
          )}
        </For>
        
        <Show when={props.trades.length === 0}>
          <div class="text-center py-4 text-gray-600 text-[10px]">
            No recent trades
          </div>
        </Show>
      </div>
    </div>
  );
}

export default TimeSales;
