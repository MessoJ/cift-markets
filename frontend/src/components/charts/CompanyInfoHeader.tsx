/**
 * COMPANY INFO HEADER v1.0
 * 
 * Bloomberg/TradingView-grade company information display.
 * Shows fundamentals, earnings dates, and key metrics above the chart.
 * 
 * Features:
 * - Company logo and name with sector badge
 * - Market cap with dynamic formatting (B/M/T)
 * - P/E ratio and dividend yield
 * - 52-week range progress bar
 * - Next earnings countdown with bell icon
 * - Pre/post market prices
 * 
 * NO MOCK DATA - All data from /api/v1/company/{symbol}/summary
 */

import { createSignal, createEffect, Show, onMount, onCleanup, For } from 'solid-js';
import { 
  Building2, Calendar, TrendingUp, TrendingDown, 
  AlertCircle, Bell, ExternalLink, Info 
} from 'lucide-solid';

interface CompanyInfoProps {
  symbol: string;
  currentPrice?: number | null;
}

interface SymbolSummary {
  symbol: string;
  name: string;
  sector?: string;
  industry?: string;
  market_cap?: number;  // In millions
  logo_url?: string;
  pe_ratio?: number;
  // Quote
  price?: number;
  change?: number;
  change_pct?: number;
  high?: number;
  low?: number;
  volume?: number;
  prev_close?: number;
  high_52w?: number;
  low_52w?: number;
  pre_market?: number;
  post_market?: number;
  // Earnings
  next_earnings?: {
    date: string;
    eps_estimate?: number;
    time?: string;  // 'bmo' or 'amc'
  };
}

// Sector colors for visual badges
const SECTOR_COLORS: Record<string, string> = {
  'Technology': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  'Healthcare': 'bg-green-500/20 text-green-400 border-green-500/30',
  'Financial': 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  'Consumer Cyclical': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  'Consumer Defensive': 'bg-teal-500/20 text-teal-400 border-teal-500/30',
  'Energy': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  'Industrials': 'bg-gray-500/20 text-gray-400 border-gray-500/30',
  'Basic Materials': 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  'Communication Services': 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  'Utilities': 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  'Real Estate': 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
};

function formatMarketCap(marketCapMillions: number): string {
  if (marketCapMillions >= 1000000) {
    return `$${(marketCapMillions / 1000000).toFixed(2)}T`;
  }
  if (marketCapMillions >= 1000) {
    return `$${(marketCapMillions / 1000).toFixed(1)}B`;
  }
  return `$${marketCapMillions.toFixed(0)}M`;
}

function formatVolume(volume: number): string {
  if (volume >= 1000000) {
    return `${(volume / 1000000).toFixed(2)}M`;
  }
  if (volume >= 1000) {
    return `${(volume / 1000).toFixed(1)}K`;
  }
  return volume.toString();
}

function calculateDaysUntil(dateStr: string): number {
  const date = new Date(dateStr);
  const today = new Date();
  const diff = date.getTime() - today.getTime();
  return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

function calculate52WeekPosition(current: number, low52w: number, high52w: number): number {
  if (high52w <= low52w) return 50;
  return ((current - low52w) / (high52w - low52w)) * 100;
}

export default function CompanyInfoHeader(props: CompanyInfoProps) {
  const [summary, setSummary] = createSignal<SymbolSummary | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [showDetails, setShowDetails] = createSignal(false);

  const fetchSummary = async () => {
    if (!props.symbol) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/v1/company/${props.symbol}/summary`, {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        setSummary(data);
      } else {
        // Fallback: just use symbol
        setSummary({ symbol: props.symbol, name: props.symbol });
      }
    } catch (err) {
      console.error('Failed to fetch company summary:', err);
      setSummary({ symbol: props.symbol, name: props.symbol });
    } finally {
      setLoading(false);
    }
  };

  // Fetch on symbol change
  createEffect(() => {
    fetchSummary();
  });

  // Refresh periodically (every 30 seconds during market hours)
  onMount(() => {
    const interval = setInterval(fetchSummary, 30000);
    onCleanup(() => clearInterval(interval));
  });

  const currentPrice = () => props.currentPrice ?? summary()?.price;
  const priceChange = () => summary()?.change ?? 0;
  const priceChangePct = () => summary()?.change_pct ?? 0;
  const isPositive = () => priceChange() >= 0;

  return (
    <div class="bg-terminal-900 border-b border-terminal-750 px-3 py-2">
      <div class="flex flex-wrap items-center gap-3 lg:gap-6">
        {/* Logo + Name + Sector */}
        <div class="flex items-center gap-3 min-w-[200px]">
          <Show 
            when={summary()?.logo_url}
            fallback={
              <div class="w-10 h-10 bg-terminal-800 rounded-lg flex items-center justify-center text-gray-500">
                <Building2 size={20} />
              </div>
            }
          >
            <img 
              src={summary()!.logo_url!} 
              alt={summary()?.name}
              class="w-10 h-10 rounded-lg object-contain bg-white p-1"
              onError={(e) => {
                e.currentTarget.style.display = 'none';
              }}
            />
          </Show>
          
          <div class="flex flex-col">
            <div class="flex items-center gap-2">
              <span class="text-lg font-bold text-white">{props.symbol}</span>
              <Show when={summary()?.sector}>
                <span class={`text-xs px-2 py-0.5 rounded-full border ${
                  SECTOR_COLORS[summary()!.sector!] || 'bg-gray-500/20 text-gray-400 border-gray-500/30'
                }`}>
                  {summary()!.sector}
                </span>
              </Show>
            </div>
            <span class="text-sm text-gray-400 truncate max-w-[200px]">
              {summary()?.name || 'Loading...'}
            </span>
          </div>
        </div>

        {/* Market Cap */}
        <Show when={summary()?.market_cap}>
          <div class="flex flex-col items-center px-3 border-l border-terminal-750">
            <span class="text-xs text-gray-500 uppercase tracking-wider">Mkt Cap</span>
            <span class="text-sm font-mono font-semibold text-white">
              {formatMarketCap(summary()!.market_cap!)}
            </span>
          </div>
        </Show>

        {/* P/E Ratio */}
        <Show when={summary()?.pe_ratio}>
          <div class="flex flex-col items-center px-3 border-l border-terminal-750">
            <span class="text-xs text-gray-500 uppercase tracking-wider">P/E</span>
            <span class="text-sm font-mono font-semibold text-white">
              {summary()!.pe_ratio!.toFixed(2)}
            </span>
          </div>
        </Show>

        {/* Volume */}
        <Show when={summary()?.volume}>
          <div class="flex flex-col items-center px-3 border-l border-terminal-750">
            <span class="text-xs text-gray-500 uppercase tracking-wider">Volume</span>
            <span class="text-sm font-mono font-semibold text-white">
              {formatVolume(summary()!.volume!)}
            </span>
          </div>
        </Show>

        {/* 52-Week Range (Mini) */}
        <Show when={summary()?.high_52w && summary()?.low_52w}>
          <div class="flex flex-col gap-1 px-3 border-l border-terminal-750 min-w-[120px]">
            <span class="text-xs text-gray-500 uppercase tracking-wider text-center">52W Range</span>
            <div class="flex items-center gap-2 text-xs font-mono">
              <span class="text-red-400">${summary()!.low_52w!.toFixed(2)}</span>
              <div class="flex-1 h-1.5 bg-terminal-750 rounded-full overflow-hidden relative">
                <div 
                  class="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full"
                  style={{ width: '100%' }}
                />
                <div 
                  class="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-white rounded-full shadow-lg border border-terminal-750"
                  style={{ 
                    left: `${calculate52WeekPosition(
                      currentPrice() || 0, 
                      summary()!.low_52w!, 
                      summary()!.high_52w!
                    )}%`,
                    transform: 'translateX(-50%) translateY(-50%)'
                  }}
                />
              </div>
              <span class="text-green-400">${summary()!.high_52w!.toFixed(2)}</span>
            </div>
          </div>
        </Show>

        {/* Next Earnings */}
        <Show when={summary()?.next_earnings?.date}>
          {(() => {
            const daysUntil = calculateDaysUntil(summary()!.next_earnings!.date);
            const isUrgent = daysUntil <= 7;
            return (
              <div class={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
                isUrgent 
                  ? 'bg-yellow-500/10 border-yellow-500/30' 
                  : 'bg-terminal-800/50 border-terminal-750'
              }`}>
                <Bell size={14} class={isUrgent ? 'text-yellow-400' : 'text-gray-400'} />
                <div class="flex flex-col">
                  <span class={`text-xs ${isUrgent ? 'text-yellow-400' : 'text-gray-400'}`}>
                    Earnings {summary()?.next_earnings?.time === 'bmo' ? '(Before Open)' : '(After Close)'}
                  </span>
                  <span class="text-sm font-mono font-semibold text-white">
                    {daysUntil === 0 ? 'Today' : daysUntil === 1 ? 'Tomorrow' : `${daysUntil} days`}
                  </span>
                </div>
                <Show when={summary()?.next_earnings?.eps_estimate}>
                  <span class="text-xs text-gray-500 ml-1">
                    Est: ${summary()!.next_earnings!.eps_estimate!.toFixed(2)}
                  </span>
                </Show>
              </div>
            );
          })()}
        </Show>

        {/* Pre/Post Market */}
        <Show when={summary()?.pre_market || summary()?.post_market}>
          <div class="flex flex-col items-center px-3 border-l border-terminal-750">
            <span class="text-xs text-gray-500 uppercase tracking-wider">
              {summary()?.pre_market ? 'Pre-Mkt' : 'Post-Mkt'}
            </span>
            <span class="text-sm font-mono font-semibold text-blue-400">
              ${(summary()?.pre_market || summary()?.post_market)?.toFixed(2)}
            </span>
          </div>
        </Show>

        {/* Spacer */}
        <div class="flex-1" />

        {/* Info Toggle Button */}
        <button
          class="p-2 rounded-lg bg-terminal-800/50 hover:bg-terminal-750 text-gray-400 hover:text-white transition-colors"
          onClick={() => setShowDetails(prev => !prev)}
          title="Toggle Company Details"
        >
          <Info size={16} />
        </button>
      </div>

      {/* Expandable Details Panel */}
      <Show when={showDetails()}>
        <div class="mt-3 pt-3 border-t border-terminal-750 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
          <div>
            <span class="text-xs text-gray-500 block">Open</span>
            <span class="font-mono text-white">${summary()?.high?.toFixed(2) || '--'}</span>
          </div>
          <div>
            <span class="text-xs text-gray-500 block">High</span>
            <span class="font-mono text-green-400">${summary()?.high?.toFixed(2) || '--'}</span>
          </div>
          <div>
            <span class="text-xs text-gray-500 block">Low</span>
            <span class="font-mono text-red-400">${summary()?.low?.toFixed(2) || '--'}</span>
          </div>
          <div>
            <span class="text-xs text-gray-500 block">Prev Close</span>
            <span class="font-mono text-white">${summary()?.prev_close?.toFixed(2) || '--'}</span>
          </div>
          <div>
            <span class="text-xs text-gray-500 block">52W High</span>
            <span class="font-mono text-green-400">${summary()?.high_52w?.toFixed(2) || '--'}</span>
          </div>
          <div>
            <span class="text-xs text-gray-500 block">52W Low</span>
            <span class="font-mono text-red-400">${summary()?.low_52w?.toFixed(2) || '--'}</span>
          </div>
        </div>
      </Show>
    </div>
  );
}
