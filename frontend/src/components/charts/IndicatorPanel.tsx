/**
 * Indicator Panel Component
 * 
 * Advanced indicator selection and configuration panel.
 * Indicators calculated server-side using Polars (12x faster).
 */

import { createSignal, For, Show } from 'solid-js';
import { TrendingUp, BarChart2, Activity, ChevronDown, ChevronUp } from 'lucide-solid';

export interface IndicatorConfig {
  id: string;
  name: string;
  enabled: boolean;
  category: 'trend' | 'momentum' | 'volatility' | 'volume';
  color?: string;
  params?: Record<string, number>;
}

export interface IndicatorPanelProps {
  activeIndicators: IndicatorConfig[];
  onToggle: (indicatorId: string) => void;
  onConfigChange?: (indicatorId: string, params: Record<string, number>) => void;
}

// Available indicators (backend calculates these)
const AVAILABLE_INDICATORS: Omit<IndicatorConfig, 'enabled'>[] = [
  // Trend indicators
  {
    id: 'sma_20',
    name: 'SMA 20',
    category: 'trend',
    color: '#3b82f6',
    params: { period: 20 },
  },
  {
    id: 'sma_50',
    name: 'SMA 50',
    category: 'trend',
    color: '#f59e0b',
    params: { period: 50 },
  },
  {
    id: 'sma_200',
    name: 'SMA 200',
    category: 'trend',
    color: '#8b5cf6',
    params: { period: 200 },
  },
  {
    id: 'ema_12',
    name: 'EMA 12',
    category: 'trend',
    color: '#10b981',
    params: { period: 12 },
  },
  {
    id: 'ema_26',
    name: 'EMA 26',
    category: 'trend',
    color: '#06b6d4',
    params: { period: 26 },
  },
  
  // Volatility indicators
  {
    id: 'bb_bands',
    name: 'Bollinger Bands',
    category: 'volatility',
    color: '#ec4899',
    params: { period: 20, stdDev: 2 },
  },
  
  // Momentum indicators
  {
    id: 'macd',
    name: 'MACD',
    category: 'momentum',
    color: '#f97316',
    params: { fast: 12, slow: 26, signal: 9 },
  },
  {
    id: 'rsi_14',
    name: 'RSI (14)',
    category: 'momentum',
    color: '#a855f7',
    params: { period: 14 },
  },
  {
    id: 'stoch',
    name: 'Stochastic',
    category: 'momentum',
    color: '#22d3ee',
    params: { k: 14, d: 3, smooth: 3 },
  },
  
  // Volatility indicators
  {
    id: 'bb_bands',
    name: 'Bollinger Bands',
    category: 'volatility',
    color: '#ec4899',
    params: { period: 20, stdDev: 2 },
  },
  {
    id: 'atr_14',
    name: 'ATR (14)',
    category: 'volatility',
    color: '#f43f5e',
    params: { period: 14 },
  },
  
  // Volume indicators
  {
    id: 'volume_sma_20',
    name: 'Volume SMA',
    category: 'volume',
    color: '#64748b',
    params: { period: 20 },
  },
  {
    id: 'obv',
    name: 'On-Balance Volume',
    category: 'volume',
    color: '#8b5cf6',
    params: {},
  },
  {
    id: 'volume_profile',
    name: 'Volume Profile',
    category: 'volume',
    color: '#60a5fa',
    params: { bins: 50 },
  },

  // Trend indicators (continued)
  {
    id: 'ichimoku',
    name: 'Ichimoku Cloud',
    category: 'trend',
    color: '#10b981',
    params: { conversion: 9, base: 26, span: 52 },
  },
  {
    id: 'pivot_points',
    name: 'Pivot Points',
    category: 'trend',
    color: '#fbbf24',
    params: {},
  },
  
  // Patterns
  {
    id: 'patterns',
    name: 'Candlestick Patterns',
    category: 'trend', // Or 'pattern'
    color: '#ffffff',
    params: {},
  },
];

export default function IndicatorPanel(props: IndicatorPanelProps) {
  const [expanded, setExpanded] = createSignal(true);  // Expanded by default
  const [selectedCategory, setSelectedCategory] = createSignal<string>('all');

  const categories = [
    { id: 'all', name: 'All', icon: Activity },
    { id: 'trend', name: 'Trend', icon: TrendingUp },
    { id: 'momentum', name: 'Momentum', icon: BarChart2 },
    { id: 'volatility', name: 'Volatility', icon: Activity },
    { id: 'volume', name: 'Volume', icon: BarChart2 },
  ];

  const filteredIndicators = () => {
    const category = selectedCategory();
    if (category === 'all') return AVAILABLE_INDICATORS;
    return AVAILABLE_INDICATORS.filter((ind) => ind.category === category);
  };

  const isEnabled = (indicatorId: string) => {
    return props.activeIndicators.some((ind) => ind.id === indicatorId && ind.enabled);
  };

  const getActiveCount = () => {
    return props.activeIndicators.filter((ind) => ind.enabled).length;
  };

  return (
    <div class="bg-terminal-900 border border-terminal-750 rounded-lg">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded())}
        class="w-full flex items-center justify-between p-3 hover:bg-terminal-850 transition-colors"
      >
        <div class="flex items-center gap-2">
          <Activity size={16} class="text-accent-500" />
          <span class="font-semibold text-white">Technical Indicators</span>
          <Show when={getActiveCount() > 0}>
            <span class="px-2 py-0.5 bg-accent-500/20 text-accent-500 text-xs rounded">
              {getActiveCount()} active
            </span>
          </Show>
        </div>
        <div class="text-gray-500">
          {expanded() ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* Expanded Panel */}
      <Show when={expanded()}>
        <div class="border-t border-terminal-750 p-3">
          {/* Category Tabs */}
          <div class="flex items-center gap-1 mb-3">
            <For each={categories}>
              {(cat) => {
                const Icon = cat.icon;
                return (
                  <button
                    onClick={() => setSelectedCategory(cat.id)}
                    class="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded transition-colors"
                    classList={{
                      'bg-accent-600 text-white': selectedCategory() === cat.id,
                      'bg-terminal-850 text-gray-400 hover:bg-terminal-800 hover:text-white':
                        selectedCategory() !== cat.id,
                    }}
                  >
                    <Icon size={14} />
                    <span>{cat.name}</span>
                  </button>
                );
              }}
            </For>
          </div>

          {/* Indicator List */}
          <div class="space-y-2 max-h-64 overflow-y-auto">
            <For each={filteredIndicators()}>
              {(indicator) => (
                <div class="flex items-center justify-between p-2 bg-terminal-850 rounded hover:bg-terminal-800 transition-colors">
                  <div class="flex items-center gap-2 flex-1">
                    <input
                      type="checkbox"
                      checked={isEnabled(indicator.id)}
                      onChange={() => props.onToggle(indicator.id)}
                      class="rounded border-terminal-700 text-accent-600 focus:ring-accent-500 focus:ring-offset-0"
                    />
                    <div
                      class="w-3 h-3 rounded"
                      style={{ 'background-color': indicator.color }}
                    />
                    <span class="text-sm text-white">{indicator.name}</span>
                  </div>
                  <div class="text-xs text-gray-500 capitalize">{indicator.category}</div>
                </div>
              )}
            </For>
          </div>

          {/* Info Footer */}
          <div class="mt-3 pt-3 border-t border-terminal-750">
            <div class="text-xs text-gray-500">
              <span class="font-semibold text-accent-500">Server-side calculation:</span>
              <span class="ml-1">Polars processing (12x faster than Pandas)</span>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
