/**
 * HeatmapGrid Component
 * 
 * Color-coded grid visualization for sector/stock performance.
 * Finviz-style heatmap for quick market overview.
 * 
 * Design System: Professional financial visualization
 */

import { createMemo, For, Show, createSignal } from 'solid-js';

export interface HeatmapCell {
  id: string;
  label: string;
  value: number;
  change?: number;
  changePercent?: number;
  size?: number; // For weighted display
  children?: HeatmapCell[];
}

interface HeatmapGridProps {
  data: HeatmapCell[];
  valueKey?: 'change' | 'changePercent' | 'value';
  colorScale?: 'diverging' | 'sequential';
  minColor?: string;
  maxColor?: string;
  showLabels?: boolean;
  showValues?: boolean;
  onCellClick?: (cell: HeatmapCell) => void;
  onCellHover?: (cell: HeatmapCell | null) => void;
  className?: string;
}

export function HeatmapGrid(props: HeatmapGridProps) {
  const [hoveredCell, setHoveredCell] = createSignal<string | null>(null);
  
  const valueKey = () => props.valueKey || 'changePercent';
  
  // Calculate min/max for color scaling
  const valueRange = createMemo(() => {
    const values = props.data.map(d => d[valueKey()] ?? 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { min, max, range: max - min || 1 };
  });
  
  // Get color based on value
  const getColor = (value: number) => {
    const { min, max } = valueRange();
    
    if (props.colorScale === 'sequential') {
      // 0% to 100% gradient
      const ratio = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
      return interpolateColor('#1f2937', '#3b82f6', ratio);
    }
    
    // Diverging scale (red-white-green for financial data)
    if (value > 3) return '#15803d'; // Dark green
    if (value > 2) return '#16a34a';
    if (value > 1) return '#22c55e';
    if (value > 0.5) return '#4ade80';
    if (value > 0) return '#86efac';
    if (value > -0.5) return '#fca5a5';
    if (value > -1) return '#f87171';
    if (value > -2) return '#ef4444';
    if (value > -3) return '#dc2626';
    return '#b91c1c'; // Dark red
  };
  
  // Simple color interpolation
  const interpolateColor = (color1: string, color2: string, ratio: number) => {
    const hex = (c: string) => parseInt(c.slice(1), 16);
    const r = (c: number) => (c >> 16) & 255;
    const g = (c: number) => (c >> 8) & 255;
    const b = (c: number) => c & 255;
    
    const c1 = hex(color1);
    const c2 = hex(color2);
    
    const red = Math.round(r(c1) + (r(c2) - r(c1)) * ratio);
    const green = Math.round(g(c1) + (g(c2) - g(c1)) * ratio);
    const blue = Math.round(b(c1) + (b(c2) - b(c1)) * ratio);
    
    return `rgb(${red}, ${green}, ${blue})`;
  };
  
  const handleCellHover = (cell: HeatmapCell | null) => {
    setHoveredCell(cell?.id || null);
    props.onCellHover?.(cell);
  };

  return (
    <div class={`${props.className || ''}`}>
      <div class="grid gap-1" style={{ 'grid-template-columns': `repeat(auto-fill, minmax(80px, 1fr))` }}>
        <For each={props.data}>
          {(cell) => {
            const value = () => cell[valueKey()] ?? 0;
            const isHovered = () => hoveredCell() === cell.id;
            
            return (
              <div
                class={`relative p-2 rounded cursor-pointer transition-all duration-200 
                  ${isHovered() ? 'ring-2 ring-white/30 z-10 scale-105' : ''}`}
                style={{ 'background-color': getColor(value()) }}
                onMouseEnter={() => handleCellHover(cell)}
                onMouseLeave={() => handleCellHover(null)}
                onClick={() => props.onCellClick?.(cell)}
              >
                <Show when={props.showLabels !== false}>
                  <div 
                    class="text-xs font-bold truncate drop-shadow-md"
                    style={{ 
                      color: Math.abs(value()) > 1 ? 'white' : '#d1d5db',
                      'text-shadow': '0 1px 2px rgba(0,0,0,0.5)'
                    }}
                  >
                    {cell.label}
                  </div>
                </Show>
                <Show when={props.showValues !== false}>
                  <div 
                    class="text-[11px] font-mono tabular-nums mt-0.5 drop-shadow"
                    style={{ 
                      color: Math.abs(value()) > 1 ? 'rgba(255,255,255,0.9)' : '#9ca3af',
                      'text-shadow': '0 1px 2px rgba(0,0,0,0.5)'
                    }}
                  >
                    {value() >= 0 ? '+' : ''}{value().toFixed(2)}%
                  </div>
                </Show>
              </div>
            );
          }}
        </For>
      </div>
      
      {/* Color scale legend */}
      <div class="flex items-center justify-center gap-2 mt-3 text-[10px] text-gray-500">
        <span>-3%</span>
        <div class="flex h-2 w-32 rounded overflow-hidden">
          <div class="flex-1 bg-danger-600" />
          <div class="flex-1 bg-danger-400" />
          <div class="flex-1 bg-gray-600" />
          <div class="flex-1 bg-success-400" />
          <div class="flex-1 bg-success-600" />
        </div>
        <span>+3%</span>
      </div>
    </div>
  );
}

/**
 * SectorHeatmap - Grouped by sector
 */
interface SectorHeatmapProps {
  sectors: Array<{
    name: string;
    change: number;
    stocks: HeatmapCell[];
  }>;
  onStockClick?: (stock: HeatmapCell, sector: string) => void;
  className?: string;
}

export function SectorHeatmap(props: SectorHeatmapProps) {
  return (
    <div class={`space-y-4 ${props.className || ''}`}>
      <For each={props.sectors}>
        {(sector) => (
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-semibold text-white">{sector.name}</span>
              <span class={`text-xs font-mono tabular-nums ${sector.change >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                {sector.change >= 0 ? '+' : ''}{sector.change.toFixed(2)}%
              </span>
            </div>
            <HeatmapGrid
              data={sector.stocks}
              onCellClick={(cell) => props.onStockClick?.(cell, sector.name)}
            />
          </div>
        )}
      </For>
    </div>
  );
}

/**
 * CorrelationMatrix - Heatmap for correlation data
 */
interface CorrelationMatrixProps {
  symbols: string[];
  correlations: number[][]; // 2D matrix
  onCellClick?: (row: string, col: string, value: number) => void;
  className?: string;
}

export function CorrelationMatrix(props: CorrelationMatrixProps) {
  const getColor = (value: number) => {
    // -1 to 1 scale
    if (value > 0.8) return '#15803d';
    if (value > 0.5) return '#22c55e';
    if (value > 0.2) return '#86efac';
    if (value > -0.2) return '#374151';
    if (value > -0.5) return '#fca5a5';
    if (value > -0.8) return '#ef4444';
    return '#b91c1c';
  };

  return (
    <div class={`overflow-x-auto ${props.className || ''}`}>
      <table class="text-xs">
        <thead>
          <tr>
            <th class="p-1 text-gray-500"></th>
            <For each={props.symbols}>
              {(symbol) => (
                <th class="p-1 text-gray-400 font-mono">{symbol}</th>
              )}
            </For>
          </tr>
        </thead>
        <tbody>
          <For each={props.symbols}>
            {(rowSymbol, rowIndex) => (
              <tr>
                <td class="p-1 text-gray-400 font-mono">{rowSymbol}</td>
                <For each={props.correlations[rowIndex()] || []}>
                  {(value, colIndex) => (
                    <td
                      class="p-1 text-center cursor-pointer hover:ring-1 hover:ring-white/30 transition-all"
                      style={{ 'background-color': getColor(value) }}
                      onClick={() => props.onCellClick?.(rowSymbol, props.symbols[colIndex()], value)}
                    >
                      <span 
                        class="font-mono tabular-nums text-[10px]"
                        style={{ color: Math.abs(value) > 0.5 ? 'white' : '#9ca3af' }}
                      >
                        {value.toFixed(2)}
                      </span>
                    </td>
                  )}
                </For>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}

export default HeatmapGrid;
