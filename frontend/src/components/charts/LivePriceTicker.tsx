/**
 * Live Price Ticker v2.0
 * 
 * Bloomberg Terminal-grade real-time price display.
 * Features:
 * - Real-time price with visual flash effects
 * - Bid/Ask spread indicator
 * - OHLCV stats with color coding
 * - 52-week high/low range bar
 * - Market status indicator
 * - Responsive mobile-first design
 */

import { createSignal, createEffect, Show } from 'solid-js';
import { TrendingUp, TrendingDown, Activity } from 'lucide-solid';

export interface LivePriceTickerProps {
  symbol: string;
  price: number | null;
  open: number;
  high: number;
  low: number;
  volume: number;
  change: number;
  changePercent: number;
  bid?: number;
  ask?: number;
  prevClose?: number;
  high52w?: number;
  low52w?: number;
  marketCap?: number;
  avgVolume?: number;
}

export default function LivePriceTicker(props: LivePriceTickerProps) {
  const [priceDirection, setPriceDirection] = createSignal<'up' | 'down' | 'neutral'>('neutral');
  const [flashClass, setFlashClass] = createSignal('');
  const [tickCount, setTickCount] = createSignal(0);

  // Track price changes for flash effect
  createEffect((prevPrice?: number) => {
    const currentPrice = props.price;
    if (currentPrice !== null && prevPrice !== undefined && currentPrice !== prevPrice) {
      const direction = currentPrice > prevPrice ? 'up' : 'down';
      setPriceDirection(direction);
      setTickCount(prev => prev + 1);
      
      // Flash effect
      setFlashClass(direction === 'up' ? 'flash-green' : 'flash-red');
      setTimeout(() => setFlashClass(''), 300);
    }
    return currentPrice ?? prevPrice;
  });

  const isPositive = () => props.change >= 0;
  const spread = () => props.bid && props.ask ? ((props.ask - props.bid) / props.bid * 10000).toFixed(1) : null;
  
  // 52-week range position (0-100%)
  const rangePosition = () => {
    if (!props.price || !props.high52w || !props.low52w) return 50;
    const range = props.high52w - props.low52w;
    if (range <= 0) return 50;
    return Math.min(100, Math.max(0, ((props.price - props.low52w) / range) * 100));
  };

  return (
    <div class="bg-terminal-900 border-b border-terminal-750">
      {/* Main Price Row */}
      <div class="flex flex-wrap items-center gap-3 sm:gap-6 px-3 sm:px-4 py-2 sm:py-3">
        {/* Symbol & Direction */}
        <div class="flex items-center gap-2">
          <h2 class="text-lg sm:text-xl font-bold text-white font-mono">{props.symbol}</h2>
          <Show when={priceDirection() !== 'neutral'}>
            <div class={`p-1 rounded ${priceDirection() === 'up' ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
              {priceDirection() === 'up' ? (
                <TrendingUp size={16} class="text-green-500" />
              ) : (
                <TrendingDown size={16} class="text-red-500" />
              )}
            </div>
          </Show>
        </div>

        {/* Live Price (Large) */}
        <div class="flex flex-col">
          <Show when={props.price !== null} fallback={
            <div class="text-xl sm:text-2xl font-mono font-bold text-gray-500 animate-pulse">---.--</div>
          }>
            <div 
              class={`text-xl sm:text-2xl font-mono font-bold transition-all duration-150 ${flashClass()}`}
              classList={{
                'text-green-500': isPositive(),
                'text-red-500': !isPositive(),
              }}
            >
              ${props.price?.toFixed(2)}
            </div>
          </Show>
          <div class="text-[10px] text-gray-500 font-mono">LAST</div>
        </div>

        {/* Change */}
        <div class="flex flex-col">
          <div 
            class="text-sm font-mono font-semibold"
            classList={{
              'text-green-500': isPositive(),
              'text-red-500': !isPositive(),
            }}
          >
            {isPositive() ? '+' : ''}{props.change.toFixed(2)}
            <span class="ml-1 text-xs">({isPositive() ? '+' : ''}{props.changePercent.toFixed(2)}%)</span>
          </div>
          <div class="text-[10px] text-gray-500 font-mono">CHG</div>
        </div>

        {/* Bid/Ask Spread */}
        <Show when={props.bid && props.ask}>
          <div class="hidden sm:flex flex-col border-l border-terminal-750 pl-3">
            <div class="flex items-center gap-2 text-xs font-mono">
              <span class="text-green-500">${props.bid?.toFixed(2)}</span>
              <span class="text-gray-600">×</span>
              <span class="text-red-500">${props.ask?.toFixed(2)}</span>
            </div>
            <div class="text-[10px] text-gray-500">
              BID/ASK <Show when={spread()}><span class="text-gray-600">({spread()} bps)</span></Show>
            </div>
          </div>
        </Show>

        {/* OHLC Stats - Compact on mobile, expanded on desktop */}
        <div class="flex gap-2 sm:gap-4 text-xs border-l border-terminal-750 pl-2 sm:pl-3">
          <div class="flex flex-col">
            <span class="text-gray-500">O</span>
            <span class="text-white font-mono text-[11px] sm:text-xs">{props.open > 0 ? props.open.toFixed(2) : '--'}</span>
          </div>
          <div class="flex flex-col">
            <span class="text-gray-500">H</span>
            <span class="text-green-400 font-mono text-[11px] sm:text-xs">{props.high > 0 ? props.high.toFixed(2) : '--'}</span>
          </div>
          <div class="flex flex-col">
            <span class="text-gray-500">L</span>
            <span class="text-red-400 font-mono text-[11px] sm:text-xs">{props.low > 0 ? props.low.toFixed(2) : '--'}</span>
          </div>
          <div class="flex flex-col">
            <span class="text-gray-500">V</span>
            <span class="text-white font-mono text-[11px] sm:text-xs">{formatVolume(props.volume)}</span>
          </div>
        </div>

        {/* 52-Week Range Bar */}
        <Show when={props.high52w && props.low52w}>
          <div class="hidden lg:flex flex-col border-l border-terminal-750 pl-3 min-w-[120px]">
            <div class="flex justify-between text-[10px] text-gray-500 mb-1">
              <span>${props.low52w?.toFixed(0)}</span>
              <span>52W</span>
              <span>${props.high52w?.toFixed(0)}</span>
            </div>
            <div class="h-1.5 bg-terminal-800 rounded-full relative">
              <div 
                class="absolute top-0 h-full w-1 bg-primary-500 rounded-full"
                style={{ left: `${rangePosition()}%`, transform: 'translateX(-50%)' }}
              />
            </div>
          </div>
        </Show>

        {/* Real-time Status */}
        <div class="ml-auto flex items-center gap-3">
          {/* Tick Counter */}
          <div class="hidden sm:flex items-center gap-1.5 text-xs text-gray-500">
            <Activity size={12} class="text-green-500" />
            <span class="font-mono">{tickCount()}</span>
          </div>
          
          {/* Live Indicator */}
          <div class="flex items-center gap-1.5 px-2 py-1 bg-green-500/10 border border-green-500/30 rounded">
            <div class="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            <span class="text-[10px] text-green-500 font-semibold tracking-wider">LIVE</span>
          </div>
        </div>
      </div>
      
      {/* Mobile OHLC Row (visible on small screens) */}
      <div class="md:hidden flex items-center justify-between px-3 pb-2 text-xs">
        <div class="flex gap-3">
          <span><span class="text-gray-500">O:</span> <span class="text-white font-mono">{props.open > 0 ? props.open.toFixed(2) : '--'}</span></span>
          <span><span class="text-gray-500">H:</span> <span class="text-green-400 font-mono">{props.high > 0 ? props.high.toFixed(2) : '--'}</span></span>
          <span><span class="text-gray-500">L:</span> <span class="text-red-400 font-mono">{props.low > 0 ? props.low.toFixed(2) : '--'}</span></span>
          <span><span class="text-gray-500">V:</span> <span class="text-white font-mono">{formatVolume(props.volume)}</span></span>
        </div>
      </div>
    </div>
  );
}

function formatVolume(volume: number | undefined | null): string {
  if (volume === undefined || volume === null || volume === 0) return '--';
  if (volume >= 1_000_000_000) {
    return `${(volume / 1_000_000_000).toFixed(2)}B`;
  } else if (volume >= 1_000_000) {
    return `${(volume / 1_000_000).toFixed(2)}M`;
  } else if (volume >= 1_000) {
    return `${(volume / 1_000).toFixed(1)}K`;
  }
  return volume.toLocaleString();
}
