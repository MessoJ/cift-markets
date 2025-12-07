/**
 * Globe Search Panel
 * Advanced filtering for globe data
 */

import { createSignal, For } from 'solid-js';
import type { GlobeFilters } from '~/hooks/useGlobeData';

interface GlobeSearchPanelProps {
  filters: GlobeFilters;
  onFilterChange: (filters: Partial<GlobeFilters>) => void;
  onReset: () => void;
}

const EXCHANGE_OPTIONS = [
  { code: 'NYSE', name: 'New York' },
  { code: 'NASDAQ', name: 'NASDAQ' },
  { code: 'LSE', name: 'London' },
  { code: 'SSE', name: 'Shanghai' },
  { code: 'TSE', name: 'Tokyo' },
  { code: 'HKEX', name: 'Hong Kong' },
  { code: 'ENX', name: 'Euronext' },
  { code: 'BSE', name: 'Mumbai' },
];

const TIMEFRAME_OPTIONS = [
  { value: '1h', label: 'Last Hour' },
  { value: '24h', label: 'Last 24 Hours' },
  { value: '7d', label: 'Last 7 Days' },
  { value: '30d', label: 'Last 30 Days' },
];

const SENTIMENT_OPTIONS = [
  { value: 'all', label: 'All Sentiment' },
  { value: 'positive', label: 'Positive' },
  { value: 'neutral', label: 'Neutral' },
  { value: 'negative', label: 'Negative' },
];

const CONNECTION_TYPE_OPTIONS = [
  { value: 'all', label: 'All Types' },
  { value: 'trade', label: 'Trade' },
  { value: 'impact', label: 'Impact' },
  { value: 'correlation', label: 'Correlation' },
];

export function GlobeSearchPanel(props: GlobeSearchPanelProps) {
  const [query, setQuery] = createSignal(props.filters.query || '');
  const [selectedExchanges, setSelectedExchanges] = createSignal<string[]>(props.filters.exchanges || []);
  const [isExpanded, setIsExpanded] = createSignal(false);

  const handleSearch = () => {
    props.onFilterChange({
      query: query() || undefined,
      exchanges: selectedExchanges().length > 0 ? selectedExchanges() : undefined,
    });
  };

  const toggleExchange = (code: string) => {
    const current = selectedExchanges();
    const newSelection = current.includes(code)
      ? current.filter(e => e !== code)
      : [...current, code];
    
    setSelectedExchanges(newSelection);
    props.onFilterChange({
      exchanges: newSelection.length > 0 ? newSelection : undefined,
    });
  };

  const handleTimeframeChange = (timeframe: string) => {
    props.onFilterChange({ timeframe });
  };

  const handleSentimentChange = (sentiment: string) => {
    props.onFilterChange({
      sentiment: sentiment === 'all' ? undefined : sentiment,
    });
  };

  const handleConnectionTypeChange = (type: string) => {
    props.onFilterChange({
      connection_type: type === 'all' ? undefined : type,
    });
  };

  const handleReset = () => {
    setQuery('');
    setSelectedExchanges([]);
    props.onReset();
  };

  const activeFilterCount = () => {
    let count = 0;
    if (props.filters.query) count++;
    if (props.filters.exchanges && props.filters.exchanges.length > 0) count += props.filters.exchanges.length;
    if (props.filters.sentiment) count++;
    if (props.filters.connection_type && props.filters.connection_type !== 'all') count++;
    return count;
  };

  return (
    <div class="bg-terminal-900/95 backdrop-blur-lg border border-terminal-750 rounded-lg p-4 space-y-4">
      {/* Search Bar */}
      <div class="relative">
        <input
          type="text"
          placeholder="Search exchanges, countries, or news..."
          value={query()}
          onInput={(e) => setQuery(e.currentTarget.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSearch();
            }
          }}
          class="w-full bg-terminal-850 border border-terminal-750 rounded-lg px-4 py-3 pr-10 text-white placeholder-gray-500 focus:outline-none focus:border-accent-500 transition-colors"
        />
        <button
          onClick={handleSearch}
          class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
        >
          🔍
        </button>
      </div>

      {/* Timeframe Selection */}
      <div>
        <label class="block text-sm font-medium text-gray-400 mb-2">Timeframe</label>
        <div class="grid grid-cols-4 gap-2">
          <For each={TIMEFRAME_OPTIONS}>
            {(option) => (
              <button
                onClick={() => handleTimeframeChange(option.value)}
                class={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  props.filters.timeframe === option.value
                    ? 'bg-accent-500 text-white'
                    : 'bg-terminal-850 text-gray-400 hover:bg-terminal-800 hover:text-white'
                }`}
              >
                {option.label}
              </button>
            )}
          </For>
        </div>
      </div>

      {/* Exchange Quick Filters */}
      <div>
        <label class="block text-sm font-medium text-gray-400 mb-2">Quick Exchanges</label>
        <div class="flex flex-wrap gap-2">
          <For each={EXCHANGE_OPTIONS}>
            {(exchange) => (
              <button
                onClick={() => toggleExchange(exchange.code)}
                class={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  selectedExchanges().includes(exchange.code)
                    ? 'bg-accent-500 text-white'
                    : 'bg-terminal-850 text-gray-400 hover:bg-terminal-800 hover:text-white'
                }`}
              >
                {exchange.code}
              </button>
            )}
          </For>
        </div>
      </div>

      {/* Advanced Filters Toggle */}
      <button
        onClick={() => setIsExpanded(!isExpanded())}
        class="w-full flex items-center justify-between text-sm text-gray-400 hover:text-white transition-colors"
      >
        <span class="font-medium">Advanced Filters</span>
        <span class="transform transition-transform" style={{ transform: isExpanded() ? 'rotate(180deg)' : 'rotate(0deg)' }}>
          ▼
        </span>
      </button>

      {/* Advanced Filters */}
      {isExpanded() && (
        <div class="space-y-4 pt-2 border-t border-terminal-750">
          {/* Sentiment Filter */}
          <div>
            <label class="block text-sm font-medium text-gray-400 mb-2">Sentiment</label>
            <select
              value={props.filters.sentiment || 'all'}
              onChange={(e) => handleSentimentChange(e.currentTarget.value)}
              class="w-full bg-terminal-850 border border-terminal-750 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-accent-500"
            >
              <For each={SENTIMENT_OPTIONS}>
                {(option) => <option value={option.value}>{option.label}</option>}
              </For>
            </select>
          </div>

          {/* Connection Type Filter */}
          <div>
            <label class="block text-sm font-medium text-gray-400 mb-2">Arc Connection Type</label>
            <select
              value={props.filters.connection_type || 'all'}
              onChange={(e) => handleConnectionTypeChange(e.currentTarget.value)}
              class="w-full bg-terminal-850 border border-terminal-750 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-accent-500"
            >
              <For each={CONNECTION_TYPE_OPTIONS}>
                {(option) => <option value={option.value}>{option.label}</option>}
              </For>
            </select>
          </div>

          {/* Min Articles Slider */}
          <div>
            <label class="block text-sm font-medium text-gray-400 mb-2">
              Min Articles: {props.filters.min_articles || 0}
            </label>
            <input
              type="range"
              min="0"
              max="50"
              step="5"
              value={props.filters.min_articles || 0}
              onInput={(e) => props.onFilterChange({ min_articles: parseInt(e.currentTarget.value) })}
              class="w-full accent-accent-500"
            />
          </div>

          {/* Arc Strength Slider */}
          <div>
            <label class="block text-sm font-medium text-gray-400 mb-2">
              Min Arc Strength: {((props.filters.min_strength || 0.3) * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={props.filters.min_strength || 0.3}
              onInput={(e) => props.onFilterChange({ min_strength: parseFloat(e.currentTarget.value) })}
              class="w-full accent-accent-500"
            />
          </div>
        </div>
      )}

      {/* Active Filters & Reset */}
      <div class="flex items-center justify-between pt-2 border-t border-terminal-750">
        <span class="text-sm text-gray-400">
          {activeFilterCount()} filter{activeFilterCount() !== 1 ? 's' : ''} active
        </span>
        <button
          onClick={handleReset}
          class="text-sm text-accent-400 hover:text-accent-300 font-medium transition-colors"
        >
          Reset All
        </button>
      </div>
    </div>
  );
}
