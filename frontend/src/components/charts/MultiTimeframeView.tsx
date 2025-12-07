/**
 * Multi-Timeframe Chart View
 * 
 * Displays multiple charts simultaneously with different timeframes.
 * Supports 2x2 grid or 3x1 stack layout.
 * 
 * Features:
 * - Independent timeframe per panel
 * - Shared symbol selection
 * - Synchronized indicators (optional)
 * - Layout persistence (localStorage)
 */

import { createSignal, For, Show } from 'solid-js';
import { LayoutGrid, Layers } from 'lucide-solid';
import CandlestickChart from './CandlestickChart';
import type { IndicatorConfig } from './IndicatorPanel';
import type { DrawingType, Drawing } from '~/types/drawing.types';

export interface MultiTimeframeViewProps {
  symbol: string;
  timeframes: string[]; // e.g., ['1d', '1h', '15m', '5m']
  layout?: '2x2' | '3x1' | '4x1';
  activeIndicators?: IndicatorConfig[];
  chartType?: 'candlestick' | 'line' | 'area';
  onLayoutChange?: (layout: '2x2' | '3x1' | '4x1') => void;
}

interface ChartPanel {
  id: string;
  timeframe: string;
  label: string;
}

export default function MultiTimeframeView(props: MultiTimeframeViewProps) {
  const [focusedPanel, setFocusedPanel] = createSignal<string | null>(null);
  
  // Generate chart panels based on timeframes
  const panels = (): ChartPanel[] => {
    return props.timeframes.map((tf, index) => ({
      id: `panel-${tf}-${index}`,
      timeframe: tf,
      label: getTimeframeLabel(tf),
    }));
  };

  /**
   * Get human-readable label for timeframe
   */
  function getTimeframeLabel(tf: string): string {
    const labels: Record<string, string> = {
      '1m': '1 Minute',
      '5m': '5 Minutes',
      '15m': '15 Minutes',
      '30m': '30 Minutes',
      '1h': '1 Hour',
      '4h': '4 Hours',
      '1d': 'Daily',
      '1w': 'Weekly',
      '1M': 'Monthly',
    };
    return labels[tf] || tf.toUpperCase();
  }

  /**
   * Get grid layout CSS classes
   */
  const getLayoutClass = () => {
    switch (props.layout || '2x2') {
      case '2x2':
        return 'grid grid-cols-2 grid-rows-2 gap-2';
      case '3x1':
        return 'grid grid-cols-1 grid-rows-3 gap-2';
      case '4x1':
        return 'grid grid-cols-1 grid-rows-4 gap-2';
      default:
        return 'grid grid-cols-2 grid-rows-2 gap-2';
    }
  };

  /**
   * Get individual panel height
   */
  const getPanelHeight = () => {
    const layout = props.layout || '2x2';
    if (layout === '2x2') return 'calc(50vh - 160px)'; // 2 rows
    if (layout === '3x1') return 'calc(33.33vh - 140px)'; // 3 rows
    if (layout === '4x1') return 'calc(25vh - 130px)'; // 4 rows
    return '400px';
  };

  /**
   * Handle panel focus (optional for future features)
   */
  const handlePanelClick = (panelId: string) => {
    setFocusedPanel(panelId);
    console.log(`📊 Focused panel: ${panelId}`);
  };

  return (
    <div class="h-full flex flex-col gap-2 p-2 bg-terminal-950">
      {/* Layout Controls */}
      <div class="flex items-center justify-between px-3 py-2 bg-terminal-900 border border-terminal-750 rounded">
        <div class="flex items-center gap-2">
          <LayoutGrid size={18} class="text-gray-400" />
          <span class="text-sm font-medium text-gray-300">Multi-Timeframe View</span>
          <span class="text-xs text-gray-500">• {props.symbol}</span>
        </div>
        
        {/* Layout Switcher */}
        <div class="flex gap-1">
          <button
            class="px-3 py-1 text-xs rounded transition-colors"
            classList={{
              'bg-primary-600 text-white': props.layout === '2x2',
              'bg-terminal-800 text-gray-400 hover:bg-terminal-750': props.layout !== '2x2',
            }}
            onClick={() => props.onLayoutChange?.('2x2')}
            title="2x2 Grid"
          >
            <LayoutGrid size={14} />
          </button>
          <button
            class="px-3 py-1 text-xs rounded transition-colors"
            classList={{
              'bg-primary-600 text-white': props.layout === '3x1',
              'bg-terminal-800 text-gray-400 hover:bg-terminal-750': props.layout !== '3x1',
            }}
            onClick={() => props.onLayoutChange?.('3x1')}
            title="3x1 Stack"
          >
            <Layers size={14} />
          </button>
          <button
            class="px-3 py-1 text-xs rounded transition-colors"
            classList={{
              'bg-primary-600 text-white': props.layout === '4x1',
              'bg-terminal-800 text-gray-400 hover:bg-terminal-750': props.layout !== '4x1',
            }}
            onClick={() => props.onLayoutChange?.('4x1')}
            title="4x1 Stack"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="3" />
              <rect x="3" y="8" width="18" height="3" />
              <rect x="3" y="13" width="18" height="3" />
              <rect x="3" y="18" width="18" height="3" />
            </svg>
          </button>
        </div>
      </div>

      {/* Chart Panels Grid */}
      <div class={`flex-1 ${getLayoutClass()}`}>
        <For each={panels()}>
          {(panel) => (
            <div
              class="relative border border-terminal-750 rounded overflow-hidden transition-all duration-200 hover:border-primary-500/50"
              classList={{
                'ring-2 ring-primary-500': focusedPanel() === panel.id,
              }}
              onClick={() => handlePanelClick(panel.id)}
            >
              {/* Panel Header */}
              <div class="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-3 py-1.5 bg-terminal-900/90 backdrop-blur-sm border-b border-terminal-750">
                <span class="text-sm font-semibold text-white">{panel.label}</span>
                <span class="text-xs text-gray-500">{props.symbol}</span>
              </div>

              {/* Chart */}
              <div class="pt-9"> {/* Offset for header */}
                <CandlestickChart
                  symbol={props.symbol}
                  timeframe={panel.timeframe}
                  chartType={props.chartType || 'candlestick'}
                  candleLimit={200} // Fewer candles for multi-view performance
                  showVolume={true}
                  height={getPanelHeight()}
                  activeIndicators={props.activeIndicators || []}
                  enableRealTime={true}
                  // Drawings disabled in multi-view for simplicity
                  drawings={[]}
                />
              </div>
            </div>
          )}
        </For>
      </div>

      {/* Info Footer */}
      <div class="px-3 py-2 bg-terminal-900 border border-terminal-750 rounded text-xs text-gray-500">
        <span>💡 Tip: Click a panel to focus. All charts update in real-time.</span>
      </div>
    </div>
  );
}
