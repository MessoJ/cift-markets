/**
 * SymbolSearch Component
 * 
 * Autocomplete search for stock symbols with recent/favorites.
 * Essential for professional trading workflows.
 * 
 * Design System: Bloomberg Terminal style
 */

import { createSignal, createEffect, For, Show, onCleanup } from 'solid-js';
import { Search, Star, Clock, TrendingUp, X } from 'lucide-solid';

export interface SearchResult {
  symbol: string;
  name: string;
  exchange?: string;
  type?: 'stock' | 'etf' | 'crypto' | 'index' | 'fx';
  price?: number;
  change?: number;
  changePercent?: number;
}

interface SymbolSearchProps {
  value: string;
  onSelect: (result: SearchResult) => void;
  onSearch?: (query: string) => Promise<SearchResult[]>;
  recentSymbols?: string[];
  favoriteSymbols?: string[];
  placeholder?: string;
  showPrice?: boolean;
  autoFocus?: boolean;
  className?: string;
}

// Mock search results - in production, this would call an API
const mockSearch = async (query: string): Promise<SearchResult[]> => {
  if (!query || query.length < 1) return [];
  
  const allSymbols: SearchResult[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ', type: 'stock', price: 178.50, change: 2.30, changePercent: 1.31 },
    { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ', type: 'stock', price: 378.20, change: -1.50, changePercent: -0.40 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ', type: 'stock', price: 141.80, change: 0.90, changePercent: 0.64 },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ', type: 'stock', price: 178.25, change: 3.20, changePercent: 1.83 },
    { symbol: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ', type: 'stock', price: 248.50, change: -5.30, changePercent: -2.09 },
    { symbol: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ', type: 'stock', price: 505.75, change: 8.40, changePercent: 1.69 },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ', type: 'stock', price: 875.30, change: 15.20, changePercent: 1.77 },
    { symbol: 'SPY', name: 'SPDR S&P 500 ETF', exchange: 'NYSE', type: 'etf', price: 502.40, change: 1.80, changePercent: 0.36 },
    { symbol: 'QQQ', name: 'Invesco QQQ Trust', exchange: 'NASDAQ', type: 'etf', price: 432.15, change: 2.50, changePercent: 0.58 },
    { symbol: 'BTC-USD', name: 'Bitcoin USD', exchange: 'CRYPTO', type: 'crypto', price: 98500, change: 1200, changePercent: 1.23 },
    { symbol: 'ETH-USD', name: 'Ethereum USD', exchange: 'CRYPTO', type: 'crypto', price: 3450, change: -50, changePercent: -1.43 },
  ];
  
  const q = query.toUpperCase();
  return allSymbols.filter(s => 
    s.symbol.includes(q) || s.name.toUpperCase().includes(q)
  ).slice(0, 8);
};

export function SymbolSearch(props: SymbolSearchProps) {
  const [query, setQuery] = createSignal(props.value || '');
  const [results, setResults] = createSignal<SearchResult[]>([]);
  const [isOpen, setIsOpen] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  const [activeIndex, setActiveIndex] = createSignal(-1);
  
  let inputRef: HTMLInputElement | undefined;
  let containerRef: HTMLDivElement | undefined;
  
  // Search effect with debounce
  let searchTimeout: ReturnType<typeof setTimeout>;
  
  createEffect(() => {
    const q = query();
    
    clearTimeout(searchTimeout);
    
    if (!q || q.length < 1) {
      setResults([]);
      return;
    }
    
    searchTimeout = setTimeout(async () => {
      setLoading(true);
      try {
        const searchFn = props.onSearch || mockSearch;
        const searchResults = await searchFn(q);
        setResults(searchResults);
        setActiveIndex(-1);
      } finally {
        setLoading(false);
      }
    }, 150);
  });
  
  // Click outside to close
  const handleClickOutside = (e: MouseEvent) => {
    if (containerRef && !containerRef.contains(e.target as Node)) {
      setIsOpen(false);
    }
  };
  
  createEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    onCleanup(() => document.removeEventListener('mousedown', handleClickOutside));
  });
  
  // Keyboard navigation
  const handleKeyDown = (e: KeyboardEvent) => {
    const items = results();
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setActiveIndex(prev => Math.min(prev + 1, items.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveIndex(prev => Math.max(prev - 1, -1));
        break;
      case 'Enter':
        e.preventDefault();
        if (activeIndex() >= 0 && items[activeIndex()]) {
          handleSelect(items[activeIndex()]);
        } else if (items.length === 1) {
          handleSelect(items[0]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        inputRef?.blur();
        break;
    }
  };
  
  const handleSelect = (result: SearchResult) => {
    setQuery(result.symbol);
    setIsOpen(false);
    props.onSelect(result);
  };
  
  const getTypeIcon = (type?: string) => {
    switch (type) {
      case 'etf': return '📊';
      case 'crypto': return '₿';
      case 'index': return '📈';
      case 'fx': return '💱';
      default: return '';
    }
  };

  return (
    <div ref={containerRef} class={`relative ${props.className || ''}`}>
      {/* Input */}
      <div class="relative">
        <Search class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
        <input
          ref={inputRef}
          type="text"
          value={query()}
          onInput={(e) => {
            setQuery(e.currentTarget.value.toUpperCase());
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={props.placeholder || 'Search symbol...'}
          autofocus={props.autoFocus}
          class="w-full bg-terminal-850 border border-terminal-750 text-white font-mono text-sm pl-9 pr-8 py-2 
                 focus:outline-none focus:border-accent-500 transition-colors"
        />
        <Show when={query()}>
          <button
            class="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-terminal-700 rounded transition-colors"
            onClick={() => {
              setQuery('');
              setResults([]);
              inputRef?.focus();
            }}
          >
            <X class="w-3 h-3 text-gray-500" />
          </button>
        </Show>
      </div>
      
      {/* Dropdown */}
      <Show when={isOpen() && (results().length > 0 || props.recentSymbols?.length || props.favoriteSymbols?.length)}>
        <div class="absolute z-50 w-full mt-1 bg-terminal-900 border border-terminal-750 shadow-xl max-h-80 overflow-y-auto">
          {/* Favorites */}
          <Show when={!query() && props.favoriteSymbols?.length}>
            <div class="px-3 py-1.5 text-[10px] font-mono text-gray-500 uppercase bg-terminal-850 border-b border-terminal-800">
              <Star class="w-3 h-3 inline mr-1" /> Favorites
            </div>
            <div class="flex flex-wrap gap-1 p-2 border-b border-terminal-800">
              <For each={props.favoriteSymbols}>
                {(symbol) => (
                  <button
                    class="px-2 py-1 text-xs font-mono text-white bg-terminal-800 hover:bg-terminal-700 
                           rounded transition-colors"
                    onClick={() => handleSelect({ symbol, name: symbol })}
                  >
                    {symbol}
                  </button>
                )}
              </For>
            </div>
          </Show>
          
          {/* Recent */}
          <Show when={!query() && props.recentSymbols?.length}>
            <div class="px-3 py-1.5 text-[10px] font-mono text-gray-500 uppercase bg-terminal-850 border-b border-terminal-800">
              <Clock class="w-3 h-3 inline mr-1" /> Recent
            </div>
            <div class="flex flex-wrap gap-1 p-2 border-b border-terminal-800">
              <For each={props.recentSymbols}>
                {(symbol) => (
                  <button
                    class="px-2 py-1 text-xs font-mono text-gray-400 bg-terminal-850 hover:bg-terminal-800 
                           rounded transition-colors"
                    onClick={() => handleSelect({ symbol, name: symbol })}
                  >
                    {symbol}
                  </button>
                )}
              </For>
            </div>
          </Show>
          
          {/* Search Results */}
          <Show when={results().length > 0}>
            <div class="px-3 py-1.5 text-[10px] font-mono text-gray-500 uppercase bg-terminal-850 border-b border-terminal-800">
              <TrendingUp class="w-3 h-3 inline mr-1" /> Results
            </div>
            <For each={results()}>
              {(result, index) => (
                <div
                  class={`flex items-center justify-between px-3 py-2 cursor-pointer transition-colors
                    ${activeIndex() === index() ? 'bg-accent-500/20' : 'hover:bg-terminal-850'}`}
                  onClick={() => handleSelect(result)}
                  onMouseEnter={() => setActiveIndex(index())}
                >
                  <div class="flex items-center gap-3 min-w-0">
                    <div class="flex flex-col min-w-0">
                      <div class="flex items-center gap-2">
                        <span class="text-sm font-bold text-white font-mono">
                          {result.symbol}
                        </span>
                        <Show when={getTypeIcon(result.type)}>
                          <span class="text-xs">{getTypeIcon(result.type)}</span>
                        </Show>
                        <Show when={result.exchange}>
                          <span class="text-[10px] text-gray-600 font-mono">{result.exchange}</span>
                        </Show>
                      </div>
                      <span class="text-xs text-gray-500 truncate">{result.name}</span>
                    </div>
                  </div>
                  
                  <Show when={props.showPrice && result.price !== undefined}>
                    <div class="flex flex-col items-end ml-3 flex-shrink-0">
                      <span class="text-xs font-mono text-white tabular-nums">
                        ${result.price!.toFixed(2)}
                      </span>
                      <span class={`text-[10px] font-mono tabular-nums ${
                        (result.changePercent ?? 0) >= 0 ? 'text-success-400' : 'text-danger-400'
                      }`}>
                        {(result.changePercent ?? 0) >= 0 ? '+' : ''}{result.changePercent?.toFixed(2)}%
                      </span>
                    </div>
                  </Show>
                </div>
              )}
            </For>
          </Show>
          
          {/* Loading state */}
          <Show when={loading()}>
            <div class="px-3 py-4 text-center text-xs text-gray-500">
              <span class="animate-pulse">Searching...</span>
            </div>
          </Show>
          
          {/* No results */}
          <Show when={query() && !loading() && results().length === 0}>
            <div class="px-3 py-4 text-center text-xs text-gray-500">
              No results for "{query()}"
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}

export default SymbolSearch;
