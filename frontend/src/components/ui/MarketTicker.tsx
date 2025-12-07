/**
 * MarketTicker Component
 * 
 * Scrolling ticker showing market movers and indices.
 * Bloomberg/CNBC style horizontal news ticker.
 * 
 * Design System: Professional financial display
 */

import { createSignal, For, Show } from 'solid-js';

export interface TickerItem {
  symbol: string;
  name?: string;
  price: number;
  change: number;
  changePercent: number;
  volume?: number;
  type?: 'stock' | 'index' | 'crypto' | 'fx';
}

interface MarketTickerProps {
  items: TickerItem[];
  speed?: number; // pixels per second
  pauseOnHover?: boolean;
  showVolume?: boolean;
  onItemClick?: (item: TickerItem) => void;
  className?: string;
}

export function MarketTicker(props: MarketTickerProps) {
  const [isPaused, setIsPaused] = createSignal(false);
  const speed = () => props.speed || 50;
  
  // Duplicate items for seamless loop
  const displayItems = () => [...props.items, ...props.items];
  
  const handleMouseEnter = () => {
    if (props.pauseOnHover !== false) {
      setIsPaused(true);
    }
  };
  
  const handleMouseLeave = () => {
    setIsPaused(false);
  };
  
  const formatVolume = (vol: number) => {
    if (vol >= 1000000000) return `${(vol / 1000000000).toFixed(1)}B`;
    if (vol >= 1000000) return `${(vol / 1000000).toFixed(1)}M`;
    if (vol >= 1000) return `${(vol / 1000).toFixed(1)}K`;
    return vol.toString();
  };
  
  const getTypeIcon = (type?: string) => {
    switch (type) {
      case 'index': return '📊';
      case 'crypto': return '₿';
      case 'fx': return '💱';
      default: return '';
    }
  };

  return (
    <div 
      class={`overflow-hidden bg-terminal-900 border-y border-terminal-750 ${props.className || ''}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div 
        class="flex items-center whitespace-nowrap"
        style={{
          animation: `ticker-scroll ${(displayItems().length * 200) / speed()}s linear infinite`,
          'animation-play-state': isPaused() ? 'paused' : 'running',
        }}
      >
        <For each={displayItems()}>
          {(item) => (
            <div
              class="inline-flex items-center gap-3 px-4 py-2 cursor-pointer hover:bg-terminal-850 transition-colors border-r border-terminal-800"
              onClick={() => props.onItemClick?.(item)}
            >
              {/* Symbol */}
              <div class="flex items-center gap-1.5">
                <Show when={getTypeIcon(item.type)}>
                  <span class="text-xs">{getTypeIcon(item.type)}</span>
                </Show>
                <span class="text-xs font-bold text-white">{item.symbol}</span>
              </div>
              
              {/* Price */}
              <span class="text-xs font-mono tabular-nums text-gray-300">
                ${item.price.toFixed(2)}
              </span>
              
              {/* Change */}
              <div class={`flex items-center gap-1 text-xs font-mono tabular-nums ${item.change >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                <span>{item.change >= 0 ? '▲' : '▼'}</span>
                <span>{Math.abs(item.changePercent).toFixed(2)}%</span>
              </div>
              
              {/* Volume */}
              <Show when={props.showVolume && item.volume}>
                <span class="text-[10px] text-gray-600 font-mono">
                  Vol: {formatVolume(item.volume!)}
                </span>
              </Show>
            </div>
          )}
        </For>
      </div>
      
      {/* Add animation keyframes */}
      <style>{`
        @keyframes ticker-scroll {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
      `}</style>
    </div>
  );
}

/**
 * StaticMarketBar - Non-scrolling market overview bar
 */
interface StaticMarketBarProps {
  items: TickerItem[];
  onItemClick?: (item: TickerItem) => void;
  className?: string;
}

export function StaticMarketBar(props: StaticMarketBarProps) {
  return (
    <div class={`flex items-center gap-1 px-2 py-1.5 bg-terminal-900 border-b border-terminal-750 overflow-x-auto ${props.className || ''}`}>
      <For each={props.items}>
        {(item) => (
          <div
            class="flex items-center gap-2 px-3 py-1 rounded hover:bg-terminal-850 cursor-pointer transition-colors flex-shrink-0"
            onClick={() => props.onItemClick?.(item)}
          >
            <span class="text-xs font-bold text-white">{item.symbol}</span>
            <span class="text-xs font-mono tabular-nums text-gray-400">
              {item.price.toFixed(2)}
            </span>
            <span class={`text-[10px] font-mono tabular-nums ${item.change >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
              {item.change >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
            </span>
          </div>
        )}
      </For>
    </div>
  );
}

/**
 * MarketMovers - Gainers/Losers widget
 */
interface MarketMoversProps {
  gainers: TickerItem[];
  losers: TickerItem[];
  maxItems?: number;
  onItemClick?: (item: TickerItem) => void;
  className?: string;
}

export function MarketMovers(props: MarketMoversProps) {
  const [activeTab, setActiveTab] = createSignal<'gainers' | 'losers'>('gainers');
  const maxItems = () => props.maxItems || 5;
  
  const activeItems = () => {
    const items = activeTab() === 'gainers' ? props.gainers : props.losers;
    return items.slice(0, maxItems());
  };

  return (
    <div class={`bg-terminal-900 border border-terminal-750 ${props.className || ''}`}>
      {/* Tabs */}
      <div class="flex border-b border-terminal-750">
        <button
          class={`flex-1 px-3 py-2 text-xs font-semibold transition-colors ${
            activeTab() === 'gainers'
              ? 'text-success-400 bg-success-500/10 border-b-2 border-success-500'
              : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('gainers')}
        >
          📈 Gainers
        </button>
        <button
          class={`flex-1 px-3 py-2 text-xs font-semibold transition-colors ${
            activeTab() === 'losers'
              ? 'text-danger-400 bg-danger-500/10 border-b-2 border-danger-500'
              : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('losers')}
        >
          📉 Losers
        </button>
      </div>
      
      {/* Items */}
      <div class="p-2 space-y-1">
        <For each={activeItems()}>
          {(item, index) => (
            <div
              class="flex items-center justify-between px-2 py-1.5 rounded hover:bg-terminal-850 cursor-pointer transition-colors"
              onClick={() => props.onItemClick?.(item)}
            >
              <div class="flex items-center gap-2">
                <span class="w-4 text-[10px] text-gray-600 font-mono">{index() + 1}</span>
                <span class="text-xs font-bold text-white">{item.symbol}</span>
              </div>
              <div class="flex items-center gap-3">
                <span class="text-xs font-mono tabular-nums text-gray-400">
                  ${item.price.toFixed(2)}
                </span>
                <span class={`text-xs font-mono tabular-nums font-semibold min-w-[60px] text-right ${
                  item.change >= 0 ? 'text-success-400' : 'text-danger-400'
                }`}>
                  {item.change >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
                </span>
              </div>
            </div>
          )}
        </For>
        
        <Show when={activeItems().length === 0}>
          <div class="text-center py-4 text-gray-600 text-xs">
            No data available
          </div>
        </Show>
      </div>
    </div>
  );
}

export default MarketTicker;
