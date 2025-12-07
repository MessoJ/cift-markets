import { createSignal, For, Show } from 'solid-js';

export interface SearchResult {
  type: 'country' | 'asset' | 'exchange' | 'city' | 'ship';
  name: string;
  code?: string;
  lat: number;
  lng: number;
  subtitle?: string;
  flag?: string;
}

interface GlobeSearchProps {
  onSelect: (result: SearchResult) => void;
  data: SearchResult[];
}

export function GlobeSearch(props: GlobeSearchProps) {
  const [query, setQuery] = createSignal('');
  const [showResults, setShowResults] = createSignal(false);
  const [selectedIndex, setSelectedIndex] = createSignal(0);

  const filteredResults = () => {
    const q = query().toLowerCase().trim();
    if (q.length < 2) return [];
    
    return props.data.filter(item => 
      item.name.toLowerCase().includes(q) ||
      item.code?.toLowerCase().includes(q) ||
      item.subtitle?.toLowerCase().includes(q)
    ).slice(0, 8); // Limit to 8 results
  };

  const handleInput = (e: Event) => {
    const value = (e.target as HTMLInputElement).value;
    setQuery(value);
    setShowResults(value.length >= 2);
    setSelectedIndex(0);
  };

  const handleSelect = (result: SearchResult) => {
    props.onSelect(result);
    setQuery('');
    setShowResults(false);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    const results = filteredResults();
    if (!showResults() || results.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((selectedIndex() + 1) % results.length);
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((selectedIndex() - 1 + results.length) % results.length);
        break;
      case 'Enter':
        e.preventDefault();
        if (results[selectedIndex()]) {
          handleSelect(results[selectedIndex()]);
        }
        break;
      case 'Escape':
        setShowResults(false);
        break;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'country': return '🌍';
      case 'city': return '🏙️';
      case 'asset': return '🏛️';
      case 'exchange': return '📈';
      case 'ship': return '🚢';
      default: return '📍';
    }
  };

  return (
    <div class="relative">
      {/* Search Input */}
      <div class="relative">
        <input
          type="text"
          value={query()}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          onFocus={() => query().length >= 2 && setShowResults(true)}
          onBlur={() => setTimeout(() => setShowResults(false), 200)}
          placeholder="Search countries, cities, assets..."
          class="w-full bg-terminal-800/80 backdrop-blur-sm border border-terminal-600 rounded-lg px-4 py-2.5 pl-10 text-white placeholder-gray-400 focus:outline-none focus:border-accent-500 focus:ring-1 focus:ring-accent-500 transition-colors text-sm"
        />
        <svg 
          class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      </div>

      {/* Results Dropdown */}
      <Show when={showResults() && filteredResults().length > 0}>
        <div class="absolute top-full mt-2 w-full bg-terminal-900/95 backdrop-blur-xl border border-terminal-600 rounded-lg shadow-2xl z-50 max-h-80 overflow-y-auto">
          <For each={filteredResults()}>
            {(result, index) => (
              <button
                onClick={() => handleSelect(result)}
                class={`w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-terminal-800 transition-colors border-b border-terminal-700 last:border-0 ${
                  index() === selectedIndex() ? 'bg-terminal-800' : ''
                }`}
              >
                <span class="text-2xl">{result.flag || getTypeIcon(result.type)}</span>
                <div class="flex-1 min-w-0">
                  <p class="text-white font-medium truncate">{result.name}</p>
                  <Show when={result.subtitle}>
                    <p class="text-gray-400 text-xs truncate">{result.subtitle}</p>
                  </Show>
                </div>
                <Show when={result.code}>
                  <span class="text-gray-500 text-xs font-mono">{result.code}</span>
                </Show>
                <span class="text-gray-500 text-xs capitalize">{result.type}</span>
              </button>
            )}
          </For>
        </div>
      </Show>

      {/* No Results */}
      <Show when={showResults() && query().length >= 2 && filteredResults().length === 0}>
        <div class="absolute top-full mt-2 w-full bg-terminal-900/95 backdrop-blur-xl border border-terminal-600 rounded-lg shadow-2xl z-50 px-4 py-3">
          <p class="text-gray-400 text-sm text-center">No results found for "{query()}"</p>
        </div>
      </Show>
    </div>
  );
}
