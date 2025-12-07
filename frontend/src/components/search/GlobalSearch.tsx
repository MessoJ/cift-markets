/**
 * Global Search Component
 * 
 * Universal search with autocomplete dropdown.
 * Searches across symbols, orders, positions, watchlists, news, and assets.
 * RULES COMPLIANT: All data from database via API.
 */

import { createSignal, createEffect, Show, For, onCleanup } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { Search, X, TrendingUp, FileText, Briefcase, Star, Newspaper, MapPin, Clock } from 'lucide-solid';
import { apiClient, type SearchResult } from '~/lib/api/client';

export function GlobalSearch() {
  const navigate = useNavigate();
  
  // State
  const [query, setQuery] = createSignal('');
  const [results, setResults] = createSignal<SearchResult[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [showDropdown, setShowDropdown] = createSignal(false);
  const [selectedIndex, setSelectedIndex] = createSignal(0);
  const [searchTime, setSearchTime] = createSignal(0);
  
  let searchTimeout: number | undefined;
  let inputRef: HTMLInputElement | undefined;
  let dropdownRef: HTMLDivElement | undefined;
  
  // Debounced search
  const performSearch = async (searchQuery: string) => {
    if (!searchQuery || searchQuery.trim().length < 1) {
      setResults([]);
      setShowDropdown(false);
      return;
    }
    
    try {
      setLoading(true);
      const response = await apiClient.search(searchQuery.trim(), 15);
      setResults(response.results);
      setSearchTime(response.took_ms);
      setShowDropdown(true);
      setSelectedIndex(0);
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle input change with debounce
  const handleInputChange = (value: string) => {
    setQuery(value);
    
    // Clear previous timeout
    if (searchTimeout) {
      clearTimeout(searchTimeout);
    }
    
    // Debounce search - wait 300ms after user stops typing
    searchTimeout = window.setTimeout(() => {
      performSearch(value);
    }, 300);
  };
  
  // Handle result selection
  const selectResult = (result: SearchResult) => {
    setQuery('');
    setResults([]);
    setShowDropdown(false);
    navigate(result.link);
  };
  
  // Keyboard navigation
  const handleKeyDown = (e: KeyboardEvent) => {
    if (!showDropdown() || results().length === 0) return;
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(Math.min(selectedIndex() + 1, results().length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(Math.max(selectedIndex() - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (results()[selectedIndex()]) {
          selectResult(results()[selectedIndex()]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        setShowDropdown(false);
        inputRef?.blur();
        break;
    }
  };
  
  // Click outside to close
  const handleClickOutside = (e: MouseEvent) => {
    if (
      dropdownRef &&
      !dropdownRef.contains(e.target as Node) &&
      inputRef &&
      !inputRef.contains(e.target as Node)
    ) {
      setShowDropdown(false);
    }
  };
  
  createEffect(() => {
    document.addEventListener('click', handleClickOutside);
    onCleanup(() => {
      document.removeEventListener('click', handleClickOutside);
      if (searchTimeout) clearTimeout(searchTimeout);
    });
  });
  
  // Get icon for result type
  const getResultIcon = (result: SearchResult) => {
    if (result.icon) {
      return <span class="text-lg">{result.icon}</span>;
    }
    
    switch (result.type) {
      case 'symbol':
        return <TrendingUp class="w-4 h-4 text-accent-500" />;
      case 'order':
        return <FileText class="w-4 h-4 text-blue-500" />;
      case 'position':
        return <Briefcase class="w-4 h-4 text-green-500" />;
      case 'watchlist':
        return <Star class="w-4 h-4 text-yellow-500" />;
      case 'news':
        return <Newspaper class="w-4 h-4 text-purple-500" />;
      case 'asset':
        return <MapPin class="w-4 h-4 text-orange-500" />;
      default:
        return <Search class="w-4 h-4 text-gray-500" />;
    }
  };
  
  // Get badge color for result type
  const getTypeBadgeColor = (type: string) => {
    const colors: Record<string, string> = {
      symbol: 'bg-accent-500/20 text-accent-400 border-accent-500/30',
      order: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      position: 'bg-green-500/20 text-green-400 border-green-500/30',
      watchlist: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      news: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      asset: 'bg-orange-500/20 text-orange-400 border-orange-500/30'
    };
    return colors[type] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  };
  
  return (
    <div class="relative flex-1 max-w-md">
      {/* Search Input */}
      <div class="relative">
        <Search class="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={query()}
          onInput={(e) => handleInputChange(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => query() && setShowDropdown(true)}
          placeholder="Search symbols, orders, positions..."
          class="w-full bg-terminal-900 border border-terminal-750 text-xs text-gray-300 pl-8 pr-8 py-1.5 focus:outline-none focus:border-accent-500 focus:ring-1 focus:ring-accent-500/30 font-mono placeholder:text-gray-600 transition-all"
        />
        <Show when={query()}>
          <button
            onClick={() => {
              setQuery('');
              setResults([]);
              setShowDropdown(false);
            }}
            class="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
          >
            <X class="w-3.5 h-3.5" />
          </button>
        </Show>
        <Show when={loading()}>
          <div class="absolute right-2 top-1/2 -translate-y-1/2">
            <div class="w-3 h-3 border border-accent-500 border-t-transparent rounded-full animate-spin" />
          </div>
        </Show>
      </div>
      
      {/* Search Results Dropdown */}
      <Show when={showDropdown() && results().length > 0}>
        <div
          ref={dropdownRef}
          class="absolute top-full mt-1 left-0 right-0 bg-terminal-900 border border-terminal-750 shadow-2xl max-h-[400px] overflow-y-auto z-50 animate-fadeIn"
        >
          {/* Results Header */}
          <div class="px-3 py-2 border-b border-terminal-750 flex items-center justify-between bg-terminal-850">
            <span class="text-xs font-mono text-gray-400">
              {results().length} result{results().length !== 1 ? 's' : ''}
            </span>
            <Show when={searchTime() > 0}>
              <span class="text-[10px] font-mono text-gray-600 flex items-center gap-1">
                <Clock class="w-3 h-3" />
                {searchTime().toFixed(0)}ms
              </span>
            </Show>
          </div>
          
          {/* Results List */}
          <div class="py-1">
            <For each={results()}>
              {(result, index) => (
                <button
                  onClick={() => selectResult(result)}
                  class={`w-full px-3 py-2 flex items-start gap-3 hover:bg-terminal-800 transition-colors text-left ${
                    index() === selectedIndex() ? 'bg-terminal-800' : ''
                  }`}
                >
                  {/* Icon */}
                  <div class="flex-shrink-0 mt-0.5">
                    {getResultIcon(result)}
                  </div>
                  
                  {/* Content */}
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-0.5">
                      <span class="text-sm font-medium text-white truncate">
                        {result.title}
                      </span>
                      <span class={`text-[10px] px-1.5 py-0.5 border rounded font-mono ${getTypeBadgeColor(result.type)}`}>
                        {result.type.toUpperCase()}
                      </span>
                    </div>
                    <Show when={result.subtitle}>
                      <div class="text-xs text-gray-400 font-mono mb-0.5">
                        {result.subtitle}
                      </div>
                    </Show>
                    <Show when={result.description}>
                      <div class="text-xs text-gray-500 truncate">
                        {result.description}
                      </div>
                    </Show>
                  </div>
                </button>
              )}
            </For>
          </div>
          
          {/* Footer */}
          <div class="px-3 py-1.5 border-t border-terminal-750 bg-terminal-850">
            <div class="flex items-center justify-between text-[10px] font-mono text-gray-600">
              <span>↑↓ Navigate • Enter Select • Esc Close</span>
            </div>
          </div>
        </div>
      </Show>
      
      {/* No Results */}
      <Show when={showDropdown() && !loading() && query() && results().length === 0}>
        <div
          ref={dropdownRef}
          class="absolute top-full mt-1 left-0 right-0 bg-terminal-900 border border-terminal-750 shadow-2xl z-50"
        >
          <div class="px-4 py-6 text-center">
            <Search class="w-8 h-8 text-gray-600 mx-auto mb-2" />
            <p class="text-sm text-gray-400 font-mono mb-1">No results found</p>
            <p class="text-xs text-gray-600">Try searching for a symbol, order ID, or watchlist</p>
          </div>
        </div>
      </Show>
    </div>
  );
}
