/**
 * Chart Controls Component v2.0
 * 
 * Bloomberg/TradingView-grade chart controls:
 * - Symbol search with autocomplete
 * - Timeframe quick-select with keyboard hints
 * - Chart type selector (Candlestick, Line, Area, Heikin-Ashi)
 * - Scale options (Linear, Log, Percent)
 * - Comparison mode toggle
 * - Drawing tools toggle
 * - Sidebar toggle
 * - Fullscreen
 */

import { createSignal, For, Show } from 'solid-js';
import { Search, TrendingUp, Maximize2, CandlestickChart, LineChart, PenTool, PanelRight, BarChart2 } from 'lucide-solid';
import { TIMEFRAMES } from '~/types/chart.types';
import type { ChartTimeframe } from '~/types/chart.types';

export interface ChartControlsProps {
  symbol: string;
  timeframe: string;
  chartType?: 'candlestick' | 'line' | 'area' | 'heikin_ashi';
  showDrawings?: boolean;
  showSidebar?: boolean;
  scaleType?: 'linear' | 'log' | 'percent';
  comparisonSymbols?: string[];
  onSymbolChange: (symbol: string) => void;
  onTimeframeChange: (timeframe: string) => void;
  onChartTypeChange?: (type: 'candlestick' | 'line' | 'area' | 'heikin_ashi') => void;
  onToggleDrawings?: () => void;
  onToggleSidebar?: () => void;
  onFullscreen?: () => void;
  onScaleChange?: (scale: 'linear' | 'log' | 'percent') => void;
  onAddComparison?: (symbol: string) => void;
  onRemoveComparison?: (symbol: string) => void;
}

// Popular symbols for quick access
const POPULAR_SYMBOLS = [
  'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
  'META', 'NVDA', 'AMD', 'SPY', 'QQQ',
];

// Extended symbols for search
const ALL_SYMBOLS = [
  ...POPULAR_SYMBOLS,
  'NFLX', 'DIS', 'BABA', 'PYPL', 'INTC', 'COIN', 'UBER', 'LYFT',
  'JPM', 'BAC', 'GS', 'V', 'MA', 'SQ', 'PLTR', 'NIO', 'RIVN', 'LCID',
  'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD',
];

export default function ChartControls(props: ChartControlsProps) {
  const [symbolInput, setSymbolInput] = createSignal(props.symbol);
  const [showSymbolSearch, setShowSymbolSearch] = createSignal(false);

  /**
   * Handle symbol change from input
   */
  const handleSymbolSubmit = () => {
    const symbol = symbolInput().trim().toUpperCase().replace(' (New)', '');
    if (symbol && symbol !== props.symbol) {
      props.onSymbolChange(symbol);
      setShowSymbolSearch(false);
    }
  };

  /**
   * Quick symbol selection
   */
  const selectSymbol = (symbol: string) => {
    const cleanSymbol = symbol.replace(' (New)', '');
    setSymbolInput(cleanSymbol);
    props.onSymbolChange(cleanSymbol);
    setShowSymbolSearch(false);
  };

  return (
    <div class="bg-terminal-900 border-b border-terminal-750 p-2 sm:p-3">
      <div class="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2 sm:gap-4">
        {/* Left Section: Symbol Search */}
        <div class="flex items-center gap-2 sm:gap-3 flex-wrap">
          {/* Symbol Display/Search */}
          <div class="relative">
            <button
              onClick={() => setShowSymbolSearch(!showSymbolSearch())}
              class="flex items-center gap-2 px-3 sm:px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 rounded transition-colors"
            >
              <TrendingUp size={16} class="text-accent-500" />
              <span class="font-bold text-white text-base sm:text-lg">{props.symbol}</span>
              <Search size={14} class="text-gray-500" />
            </button>

            {/* Symbol Search Dropdown */}
            <Show when={showSymbolSearch()}>
              <div class="absolute top-full left-0 mt-2 w-[calc(100vw-2rem)] sm:w-80 max-w-[320px] bg-terminal-900 border border-terminal-750 rounded-lg shadow-xl z-50">
                {/* Search Input */}
                <div class="p-3 border-b border-terminal-750">
                  <div class="relative">
                    <Search
                      size={16}
                      class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
                    />
                    <input
                      type="text"
                      value={symbolInput()}
                      onInput={(e) => setSymbolInput(e.target.value.toUpperCase())}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          handleSymbolSubmit();
                        }
                      }}
                      placeholder="Enter symbol..."
                      class="w-full bg-terminal-850 border border-terminal-750 text-white pl-10 pr-4 py-2 rounded focus:outline-none focus:border-accent-500"
                      autofocus
                    />
                  </div>
                </div>

                {/* Popular Symbols */}
                <div class="p-3">
                  <div class="text-xs text-gray-500 mb-2">Popular Symbols</div>
                  <div class="grid grid-cols-4 sm:grid-cols-5 gap-1.5 sm:gap-2">
                    <For each={POPULAR_SYMBOLS}>
                      {(symbol) => (
                        <button
                          onClick={() => selectSymbol(symbol)}
                          class="px-2 sm:px-3 py-1.5 bg-terminal-850 hover:bg-accent-600 border border-terminal-750 hover:border-accent-500 text-white text-xs rounded transition-colors"
                          classList={{
                            'bg-accent-600 border-accent-500': symbol === props.symbol,
                          }}
                        >
                          {symbol}
                        </button>
                      )}
                    </For>
                  </div>
                </div>

                {/* Watchlists (Phase 2) */}
                <div class="p-3 border-t border-terminal-750">
                  <div class="text-xs text-gray-500 mb-2">Your Watchlists</div>
                  <div class="text-xs text-gray-600 text-center py-2">
                    Connect to view your watchlists
                  </div>
                </div>
              </div>
            </Show>

            {/* Click outside to close */}
            <Show when={showSymbolSearch()}>
              <div
                class="fixed inset-0 z-40"
                onClick={() => setShowSymbolSearch(false)}
              />
            </Show>
          </div>

          {/* Divider - hidden on mobile */}
          <div class="hidden sm:block h-8 w-px bg-terminal-750" />

          {/* Timeframe Selector - scrollable on mobile */}
          <div class="flex items-center gap-1 overflow-x-auto pb-1 sm:pb-0 scrollbar-thin">
            <For each={TIMEFRAMES}>
              {(tf: ChartTimeframe) => (
                <button
                  onClick={() => props.onTimeframeChange(tf.value)}
                  class="px-2 sm:px-3 py-1 sm:py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap flex-shrink-0"
                  classList={{
                    'bg-accent-600 text-white': tf.value === props.timeframe,
                    'bg-terminal-850 text-gray-400 hover:bg-terminal-800 hover:text-white':
                      tf.value !== props.timeframe,
                  }}
                >
                  {tf.label}
                </button>
              )}
            </For>
          </div>
        </div>

        {/* Right Section: Actions - wrap on mobile */}
        <div class="flex items-center gap-1 sm:gap-2 flex-wrap justify-end">
          {/* Chart Type Selector */}
          <div class="flex items-center gap-1 bg-terminal-850 rounded p-1 border border-terminal-750">
            <button
              onClick={() => props.onChartTypeChange?.('candlestick')}
              class="p-1.5 rounded transition-colors"
              classList={{
                'bg-accent-600 text-white': (props.chartType === 'candlestick' || !props.chartType),
                'text-gray-400 hover:text-white hover:bg-terminal-800': (props.chartType !== 'candlestick' && !!props.chartType),
              }}
              title="Candlestick Chart"
            >
              <CandlestickChart size={16} />
            </button>
            <button
              onClick={() => props.onChartTypeChange?.('line')}
              class="p-1.5 rounded transition-colors"
              classList={{
                'bg-accent-600 text-white': (props.chartType === 'line'),
                'text-gray-400 hover:text-white hover:bg-terminal-800': (props.chartType !== 'line'),
              }}
              title="Line Chart"
            >
              <LineChart size={16} />
            </button>
            <button
              onClick={() => props.onChartTypeChange?.('heikin_ashi')}
              class="p-1.5 rounded transition-colors"
              classList={{
                'bg-accent-600 text-white': (props.chartType === 'heikin_ashi'),
                'text-gray-400 hover:text-white hover:bg-terminal-800': (props.chartType !== 'heikin_ashi'),
              }}
              title="Heikin Ashi"
            >
              <BarChart2 size={16} />
            </button>
          </div>

          {/* Drawing Tools Toggle */}
          <button
            onClick={props.onToggleDrawings}
            class="p-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 rounded transition-colors"
            classList={{
              'text-accent-500 border-accent-500': props.showDrawings,
              'text-gray-400': !props.showDrawings,
            }}
            title="Toggle Drawing Tools"
          >
            <PenTool size={16} />
          </button>

          {/* Sidebar Toggle */}
          <button
            onClick={props.onToggleSidebar}
            class="p-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 rounded transition-colors"
            classList={{
              'text-accent-500 border-accent-500': props.showSidebar,
              'text-gray-400': !props.showSidebar,
            }}
            title="Toggle Indicators & Settings"
          >
            <PanelRight size={16} />
          </button>

          {/* Fullscreen */}
          <Show when={props.onFullscreen}>
            <button
              onClick={props.onFullscreen}
              class="p-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 rounded transition-colors"
              title="Fullscreen"
            >
              <Maximize2 size={16} class="text-gray-400" />
            </button>
          </Show>
        </div>
      </div>
    </div>
  );
}
