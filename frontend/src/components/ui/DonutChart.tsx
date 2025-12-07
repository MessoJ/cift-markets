/**
 * DonutChart Component
 * 
 * Interactive donut/pie chart for portfolio allocation visualization.
 * Uses SVG for crisp rendering and smooth animations.
 * 
 * Design System: Professional financial visualization
 */

import { createMemo, For, Show, createSignal } from 'solid-js';

export interface DonutSegment {
  id: string;
  label: string;
  value: number;
  color: string;
  percentage?: number;
}

interface DonutChartProps {
  segments: DonutSegment[];
  size?: number;
  thickness?: number;
  showLabels?: boolean;
  showLegend?: boolean;
  showCenter?: boolean;
  centerLabel?: string;
  centerValue?: string;
  animated?: boolean;
  onSegmentClick?: (segment: DonutSegment) => void;
  onSegmentHover?: (segment: DonutSegment | null) => void;
  className?: string;
}

export function DonutChart(props: DonutChartProps) {
  const [hoveredSegment, setHoveredSegment] = createSignal<string | null>(null);
  
  const size = () => props.size || 200;
  const thickness = () => props.thickness || 30;
  const radius = () => (size() - thickness()) / 2;
  const centerX = () => size() / 2;
  const centerY = () => size() / 2;
  
  // Calculate total and percentages - safely handle undefined/empty segments
  const segments = () => props.segments || [];
  const total = createMemo(() => segments().reduce((sum, seg) => sum + seg.value, 0));
  
  const segmentsWithAngles = createMemo(() => {
    const segs = segments();
    const totalValue = total();
    if (totalValue === 0 || segs.length === 0) return [];
    
    let currentAngle = -90; // Start from top
    
    return segs.map(segment => {
      const percentage = (segment.value / totalValue) * 100;
      const angle = (percentage / 100) * 360;
      const startAngle = currentAngle;
      const endAngle = currentAngle + angle;
      currentAngle = endAngle;
      
      return {
        ...segment,
        percentage,
        startAngle,
        endAngle,
        midAngle: startAngle + angle / 2,
      };
    });
  });
  
  // Create SVG arc path
  const createArcPath = (startAngle: number, endAngle: number, r: number) => {
    const startRad = (startAngle * Math.PI) / 180;
    const endRad = (endAngle * Math.PI) / 180;
    
    const x1 = centerX() + r * Math.cos(startRad);
    const y1 = centerY() + r * Math.sin(startRad);
    const x2 = centerX() + r * Math.cos(endRad);
    const y2 = centerY() + r * Math.sin(endRad);
    
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    
    return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
  };
  
  const handleSegmentHover = (segment: DonutSegment | null) => {
    setHoveredSegment(segment?.id || null);
    props.onSegmentHover?.(segment);
  };

  // Default chart colors
  const defaultColors = [
    '#3b82f6', // Blue
    '#22c55e', // Green
    '#f59e0b', // Orange
    '#a855f7', // Purple
    '#06b6d4', // Cyan
    '#ec4899', // Pink
    '#f97316', // Orange-red
    '#14b8a6', // Teal
  ];

  return (
    <div class={`flex flex-col items-center gap-4 ${props.className || ''}`}>
      {/* Chart */}
      <div class="relative" style={{ width: `${size()}px`, height: `${size()}px` }}>
        <svg
          width={size()}
          height={size()}
          viewBox={`0 0 ${size()} ${size()}`}
          class="transform -rotate-0"
        >
          {/* Background circle */}
          <circle
            cx={centerX()}
            cy={centerY()}
            r={radius()}
            fill="none"
            stroke="#1f2937"
            stroke-width={thickness()}
          />
          
          {/* Segments */}
          <For each={segmentsWithAngles()}>
            {(segment, index) => {
              const isHovered = () => hoveredSegment() === segment.id;
              const segmentRadius = () => isHovered() ? radius() + 4 : radius();
              const segmentThickness = () => isHovered() ? thickness() + 4 : thickness();
              
              return (
                <g
                  class={`cursor-pointer transition-all duration-200 ${props.animated ? 'animate-in fade-in' : ''}`}
                  style={{ 'animation-delay': `${index() * 50}ms` }}
                  onMouseEnter={() => handleSegmentHover(segment)}
                  onMouseLeave={() => handleSegmentHover(null)}
                  onClick={() => props.onSegmentClick?.(segment)}
                >
                  <path
                    d={createArcPath(segment.startAngle, segment.endAngle - 0.5, segmentRadius())}
                    fill="none"
                    stroke={segment.color || defaultColors[index() % defaultColors.length]}
                    stroke-width={segmentThickness()}
                    stroke-linecap="round"
                    class="transition-all duration-200"
                    style={{ 
                      opacity: hoveredSegment() && !isHovered() ? 0.5 : 1,
                      filter: isHovered() ? 'brightness(1.2)' : 'none'
                    }}
                  />
                </g>
              );
            }}
          </For>
        </svg>
        
        {/* Center content */}
        <Show when={props.showCenter !== false}>
          <div class="absolute inset-0 flex flex-col items-center justify-center">
            <Show when={hoveredSegment()}>
              {(() => {
                const segment = segmentsWithAngles().find(s => s.id === hoveredSegment());
                return segment ? (
                  <>
                    <span class="text-xs text-gray-400 font-medium">{segment.label}</span>
                    <span class="text-xl font-bold text-white tabular-nums">
                      {segment.percentage.toFixed(1)}%
                    </span>
                  </>
                ) : null;
              })()}
            </Show>
            <Show when={!hoveredSegment()}>
              <span class="text-xs text-gray-500">{props.centerLabel || 'Total'}</span>
              <span class="text-lg font-bold text-white tabular-nums">
                {props.centerValue || total().toLocaleString()}
              </span>
            </Show>
          </div>
        </Show>
      </div>
      
      {/* Legend */}
      <Show when={props.showLegend}>
        <div class="flex flex-wrap justify-center gap-3 max-w-full">
          <For each={segmentsWithAngles()}>
            {(segment, index) => (
              <div
                class={`flex items-center gap-1.5 px-2 py-1 rounded cursor-pointer transition-all duration-200
                  ${hoveredSegment() === segment.id ? 'bg-terminal-800' : 'hover:bg-terminal-850'}`}
                onMouseEnter={() => handleSegmentHover(segment)}
                onMouseLeave={() => handleSegmentHover(null)}
                onClick={() => props.onSegmentClick?.(segment)}
              >
                <div
                  class="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ 'background-color': segment.color || defaultColors[index() % defaultColors.length] }}
                />
                <span class="text-xs text-gray-300 whitespace-nowrap">{segment.label}</span>
                <span class="text-xs font-mono text-gray-500 tabular-nums">
                  {segment.percentage.toFixed(1)}%
                </span>
              </div>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}

/**
 * MiniDonutChart - Compact version for inline use
 */
interface MiniDonutChartProps {
  segments: DonutSegment[];
  size?: number;
  className?: string;
}

export function MiniDonutChart(props: MiniDonutChartProps) {
  return (
    <DonutChart
      segments={props.segments}
      size={props.size || 48}
      thickness={8}
      showLabels={false}
      showLegend={false}
      showCenter={false}
      className={props.className}
    />
  );
}

export default DonutChart;
