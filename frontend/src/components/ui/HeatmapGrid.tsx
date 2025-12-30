/**
 * HeatmapGrid Component
 * 
 * Color-coded treemap visualization for sector/stock performance.
 * Finviz-style heatmap with SIZE WEIGHTED cells (bigger market cap = bigger cell).
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
  size?: number; // For weighted display (e.g., market cap)
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
  weighted?: boolean; // Enable size-weighted treemap layout
  onCellClick?: (cell: HeatmapCell) => void;
  onCellHover?: (cell: HeatmapCell | null) => void;
  className?: string;
}

// Treemap layout algorithm (squarified)
interface TreemapRect {
  x: number;
  y: number;
  width: number;
  height: number;
  cell: HeatmapCell;
}

function calculateTreemap(
  data: HeatmapCell[],
  width: number,
  height: number
): TreemapRect[] {
  if (data.length === 0) return [];
  
  // Sort by size descending for better layout
  const sorted = [...data].sort((a, b) => (b.size || 1) - (a.size || 1));
  
  const totalSize = sorted.reduce((sum, d) => sum + (d.size || 1), 0);
  const rects: TreemapRect[] = [];
  
  // Simple row-based treemap algorithm
  let currentX = 0;
  let currentY = 0;
  let rowHeight = 0;
  let rowWidth = width;
  let remainingHeight = height;
  
  let rowItems: { cell: HeatmapCell; area: number }[] = [];
  let rowTotalArea = 0;
  
  for (const cell of sorted) {
    const cellArea = ((cell.size || 1) / totalSize) * width * height;
    
    // Check if we should start a new row
    const testRowArea = rowTotalArea + cellArea;
    const testRowHeight = testRowArea / rowWidth;
    
    // If adding this cell makes the row too tall (> 40% of remaining), start new row
    if (rowItems.length > 0 && testRowHeight > remainingHeight * 0.4) {
      // Finalize current row
      const actualRowHeight = rowTotalArea / rowWidth;
      let cellX = currentX;
      
      for (const item of rowItems) {
        const cellWidth = item.area / actualRowHeight;
        rects.push({
          x: cellX,
          y: currentY,
          width: cellWidth,
          height: actualRowHeight,
          cell: item.cell,
        });
        cellX += cellWidth;
      }
      
      // Start new row
      currentY += actualRowHeight;
      remainingHeight -= actualRowHeight;
      rowItems = [];
      rowTotalArea = 0;
    }
    
    rowItems.push({ cell, area: cellArea });
    rowTotalArea += cellArea;
  }
  
  // Finalize last row
  if (rowItems.length > 0) {
    const actualRowHeight = Math.min(rowTotalArea / rowWidth, remainingHeight);
    let cellX = currentX;
    
    for (const item of rowItems) {
      const cellWidth = item.area / actualRowHeight;
      rects.push({
        x: cellX,
        y: currentY,
        width: cellWidth,
        height: actualRowHeight,
        cell: item.cell,
      });
      cellX += cellWidth;
    }
  }
  
  return rects;
}

export function HeatmapGrid(props: HeatmapGridProps) {
  const [hoveredCell, setHoveredCell] = createSignal<string | null>(null);
  const [containerSize, setContainerSize] = createSignal({ width: 800, height: 400 });
  
  const valueKey = () => props.valueKey || 'changePercent';
  const isWeighted = () => props.weighted !== false; // Default to weighted
  
  // Calculate min/max for color scaling
  const valueRange = createMemo(() => {
    const values = props.data.map(d => d[valueKey()] ?? 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { min, max, range: max - min || 1 };
  });
  
  // Calculate treemap layout
  const treemapRects = createMemo(() => {
    if (!isWeighted()) return [];
    const { width, height } = containerSize();
    return calculateTreemap(props.data, width, height);
  });
  
  // Get color based on value
  const getColor = (value: number) => {
    const { min, max } = valueRange();
    
    if (props.colorScale === 'sequential') {
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
  
  // Format market cap for display
  const formatSize = (size: number) => {
    if (size >= 1e12) return `$${(size / 1e12).toFixed(1)}T`;
    if (size >= 1e9) return `$${(size / 1e9).toFixed(0)}B`;
    if (size >= 1e6) return `$${(size / 1e6).toFixed(0)}M`;
    return `$${size.toFixed(0)}`;
  };

  // Handle container resize
  let containerRef: HTMLDivElement | undefined;
  
  const updateSize = () => {
    if (containerRef) {
      const rect = containerRef.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setContainerSize({ width: rect.width, height: Math.max(rect.height, 300) });
      }
    }
  };

  return (
    <div class={`${props.className || ''}`}>
      {/* Weighted Treemap View */}
      <Show when={isWeighted() && props.data.length > 0}>
        <div 
          ref={containerRef}
          class="relative w-full bg-terminal-900 rounded-lg overflow-hidden"
          style={{ height: '400px' }}
          onLoad={updateSize}
        >
          {/* Initialize size on mount */}
          {(() => { setTimeout(updateSize, 0); return null; })()}
          
          <For each={treemapRects()}>
            {(rect) => {
              const value = () => rect.cell[valueKey()] ?? 0;
              const isHovered = () => hoveredCell() === rect.cell.id;
              const isLarge = () => rect.width > 60 && rect.height > 40;
              const isMedium = () => rect.width > 40 && rect.height > 30;
              
              return (
                <div
                  class={`absolute p-1 cursor-pointer transition-all duration-200 border border-terminal-900/50
                    ${isHovered() ? 'ring-2 ring-white/50 z-20 brightness-110' : 'z-10'}`}
                  style={{ 
                    left: `${rect.x}px`,
                    top: `${rect.y}px`,
                    width: `${rect.width}px`,
                    height: `${rect.height}px`,
                    'background-color': getColor(value()),
                  }}
                  onMouseEnter={() => handleCellHover(rect.cell)}
                  onMouseLeave={() => handleCellHover(null)}
                  onClick={() => props.onCellClick?.(rect.cell)}
                >
                  <div class="h-full flex flex-col justify-center items-center overflow-hidden">
                    {/* Symbol - always show if space */}
                    <Show when={isMedium() && props.showLabels !== false}>
                      <div 
                        class={`font-bold truncate drop-shadow-md ${isLarge() ? 'text-sm' : 'text-xs'}`}
                        style={{ 
                          color: Math.abs(value()) > 1 ? 'white' : '#d1d5db',
                          'text-shadow': '0 1px 2px rgba(0,0,0,0.7)'
                        }}
                      >
                        {rect.cell.label}
                      </div>
                    </Show>
                    
                    {/* Change % */}
                    <Show when={isMedium() && props.showValues !== false}>
                      <div 
                        class={`font-mono tabular-nums drop-shadow ${isLarge() ? 'text-xs' : 'text-[10px]'}`}
                        style={{ 
                          color: Math.abs(value()) > 1 ? 'rgba(255,255,255,0.9)' : '#9ca3af',
                          'text-shadow': '0 1px 2px rgba(0,0,0,0.7)'
                        }}
                      >
                        {value() >= 0 ? '+' : ''}{value().toFixed(2)}%
                      </div>
                    </Show>
                    
                    {/* Market cap for large cells */}
                    <Show when={isLarge() && rect.cell.size}>
                      <div 
                        class="text-[9px] opacity-70 font-mono mt-0.5"
                        style={{ color: 'rgba(255,255,255,0.7)' }}
                      >
                        {formatSize(rect.cell.size || 0)}
                      </div>
                    </Show>
                  </div>
                </div>
              );
            }}
          </For>
          
          {/* Tooltip for small cells */}
          <Show when={hoveredCell()}>
            {(() => {
              const rect = treemapRects().find(r => r.cell.id === hoveredCell());
              if (!rect || (rect.width > 40 && rect.height > 30)) return null;
              
              const value = rect.cell[valueKey()] ?? 0;
              return (
                <div 
                  class="fixed z-50 bg-terminal-800 border border-terminal-600 rounded px-2 py-1 shadow-lg pointer-events-none"
                  style={{
                    left: `${rect.x + rect.width / 2}px`,
                    top: `${rect.y - 40}px`,
                    transform: 'translateX(-50%)',
                  }}
                >
                  <div class="text-xs font-bold text-white">{rect.cell.label}</div>
                  <div class={`text-xs font-mono ${value >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                    {value >= 0 ? '+' : ''}{value.toFixed(2)}%
                  </div>
                  <Show when={rect.cell.size}>
                    <div class="text-[10px] text-gray-400">{formatSize(rect.cell.size || 0)}</div>
                  </Show>
                </div>
              );
            })()}
          </Show>
        </div>
      </Show>
      
      {/* Equal-size Grid View (fallback) */}
      <Show when={!isWeighted()}>
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
      </Show>
      
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
        <span class="ml-4 text-gray-600">|</span>
        <span class="ml-2 text-gray-400">Size = Market Cap</span>
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
