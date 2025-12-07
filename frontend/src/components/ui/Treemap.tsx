/**
 * Treemap Component
 * 
 * Interactive treemap visualization for portfolio holdings.
 * Visual weight corresponds to position size.
 * 
 * Design System: Bloomberg-inspired portfolio visualization
 */

import { createMemo, For, Show, createSignal } from 'solid-js';

export interface TreemapItem {
  id: string;
  label: string;
  value: number;
  change?: number;
  changePercent?: number;
  color?: string;
  children?: TreemapItem[];
}

interface TreemapProps {
  items: TreemapItem[];
  width?: number;
  height?: number;
  minCellHeight?: number;
  showLabels?: boolean;
  showValues?: boolean;
  onItemClick?: (item: TreemapItem) => void;
  onItemHover?: (item: TreemapItem | null) => void;
  className?: string;
}

interface TreemapCell {
  item: TreemapItem;
  x: number;
  y: number;
  width: number;
  height: number;
}

export function Treemap(props: TreemapProps) {
  const [hoveredItem, setHoveredItem] = createSignal<string | null>(null);
  
  const width = () => props.width || 400;
  const height = () => props.height || 300;
  const minCellHeight = () => props.minCellHeight || 40;
  
  // Safely access items
  const items = () => props.items || [];
  
  // Calculate total value - safely handle undefined/empty items
  const total = createMemo(() => items().reduce((sum, item) => sum + item.value, 0));
  
  // Squarified treemap algorithm (simplified)
  const cells = createMemo((): TreemapCell[] => {
    const sortedItems = [...items()].sort((a, b) => b.value - a.value);
    const totalVal = total();
    
    if (totalVal === 0 || sortedItems.length === 0) return [];
    
    const result: TreemapCell[] = [];
    let currentY = 0;
    
    // Simple row-based layout
    let currentRow: TreemapItem[] = [];
    let currentRowValue = 0;
    
    for (const item of sortedItems) {
      currentRow.push(item);
      currentRowValue += item.value;
      
      // Calculate row height based on value ratio
      const rowHeight = (currentRowValue / totalVal) * height();
      
      // If row height is reasonable or we've added all items
      if (rowHeight >= minCellHeight() || item === sortedItems[sortedItems.length - 1]) {
        // Layout items in this row
        let currentX = 0;
        
        for (const rowItem of currentRow) {
          const cellWidth = (rowItem.value / currentRowValue) * width();
          
          result.push({
            item: rowItem,
            x: currentX,
            y: currentY,
            width: cellWidth,
            height: rowHeight,
          });
          
          currentX += cellWidth;
        }
        
        currentY += rowHeight;
        currentRow = [];
        currentRowValue = 0;
      }
    }
    
    return result;
  });
  
  const getColor = (item: TreemapItem) => {
    if (item.color) return item.color;
    
    // Color based on change
    const change = item.changePercent ?? item.change ?? 0;
    if (change > 3) return '#22c55e';
    if (change > 1) return '#4ade80';
    if (change > 0) return '#86efac';
    if (change > -1) return '#fca5a5';
    if (change > -3) return '#f87171';
    return '#ef4444';
  };
  
  const handleItemHover = (item: TreemapItem | null) => {
    setHoveredItem(item?.id || null);
    props.onItemHover?.(item);
  };

  return (
    <div 
      class={`relative ${props.className || ''}`}
      style={{ width: `${width()}px`, height: `${height()}px` }}
    >
      <svg width={width()} height={height()} viewBox={`0 0 ${width()} ${height()}`}>
        <For each={cells()}>
          {(cell) => {
            const isHovered = () => hoveredItem() === cell.item.id;
            const color = () => getColor(cell.item);
            const showContent = () => cell.width > 50 && cell.height > 30;
            
            return (
              <g
                class="cursor-pointer transition-all duration-150"
                onMouseEnter={() => handleItemHover(cell.item)}
                onMouseLeave={() => handleItemHover(null)}
                onClick={() => props.onItemClick?.(cell.item)}
              >
                {/* Background rect */}
                <rect
                  x={cell.x + 1}
                  y={cell.y + 1}
                  width={Math.max(0, cell.width - 2)}
                  height={Math.max(0, cell.height - 2)}
                  fill={color()}
                  rx={4}
                  class="transition-all duration-150"
                  style={{
                    opacity: hoveredItem() && !isHovered() ? 0.6 : 0.85,
                    filter: isHovered() ? 'brightness(1.2)' : 'none',
                    transform: isHovered() ? 'scale(1.02)' : 'scale(1)',
                    'transform-origin': `${cell.x + cell.width / 2}px ${cell.y + cell.height / 2}px`,
                  }}
                />
                
                {/* Label and value */}
                <Show when={showContent() && props.showLabels !== false}>
                  <foreignObject
                    x={cell.x + 4}
                    y={cell.y + 4}
                    width={cell.width - 8}
                    height={cell.height - 8}
                    class="pointer-events-none"
                  >
                    <div class="flex flex-col h-full justify-between">
                      <span 
                        class="text-xs font-bold text-white truncate drop-shadow-md"
                        style={{ 'text-shadow': '0 1px 2px rgba(0,0,0,0.5)' }}
                      >
                        {cell.item.label}
                      </span>
                      <Show when={props.showValues !== false && cell.height > 50}>
                        <div class="flex flex-col">
                          <span 
                            class="text-[10px] font-mono text-white/90 tabular-nums drop-shadow"
                            style={{ 'text-shadow': '0 1px 2px rgba(0,0,0,0.5)' }}
                          >
                            ${cell.item.value.toLocaleString()}
                          </span>
                          <Show when={cell.item.changePercent !== undefined}>
                            <span 
                              class="text-[10px] font-mono text-white/80 tabular-nums drop-shadow"
                              style={{ 'text-shadow': '0 1px 2px rgba(0,0,0,0.5)' }}
                            >
                              {cell.item.changePercent! >= 0 ? '+' : ''}{cell.item.changePercent?.toFixed(2)}%
                            </span>
                          </Show>
                        </div>
                      </Show>
                    </div>
                  </foreignObject>
                </Show>
              </g>
            );
          }}
        </For>
      </svg>
      
      {/* Tooltip for hovered item */}
      <Show when={hoveredItem()}>
        {(() => {
          const cell = cells().find(c => c.item.id === hoveredItem());
          if (!cell) return null;
          
          const tooltipX = Math.min(cell.x + cell.width / 2, width() - 100);
          const tooltipY = cell.y > 60 ? cell.y - 10 : cell.y + cell.height + 10;
          
          return (
            <div
              class="absolute z-10 bg-terminal-900 border border-terminal-700 rounded-lg px-3 py-2 shadow-xl pointer-events-none"
              style={{
                left: `${tooltipX}px`,
                top: `${tooltipY}px`,
                transform: 'translate(-50%, -100%)',
              }}
            >
              <div class="text-sm font-bold text-white">{cell.item.label}</div>
              <div class="text-xs text-gray-400 font-mono tabular-nums">
                ${cell.item.value.toLocaleString()}
              </div>
              <Show when={cell.item.changePercent !== undefined}>
                <div class={`text-xs font-mono tabular-nums ${cell.item.changePercent! >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                  {cell.item.changePercent! >= 0 ? '+' : ''}{cell.item.changePercent?.toFixed(2)}%
                </div>
              </Show>
            </div>
          );
        })()}
      </Show>
    </div>
  );
}

export default Treemap;
