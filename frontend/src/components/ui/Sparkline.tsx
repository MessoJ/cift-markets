/**
 * Sparkline Component
 * 
 * Minimal inline chart for showing price trends in tables and lists.
 * Uses SVG for crisp rendering at any size.
 * 
 * Design System: Professional financial visualization
 */

import { createMemo, Show } from 'solid-js';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  strokeWidth?: number;
  showArea?: boolean;
  showDots?: boolean;
  animated?: boolean;
  className?: string;
}

export function Sparkline(props: SparklineProps) {
  const width = () => props.width || 80;
  const height = () => props.height || 24;
  const color = () => props.color || (isPositive() ? '#22c55e' : '#ef4444');
  const strokeWidth = () => props.strokeWidth || 1.5;
  
  // Calculate if trend is positive
  const isPositive = createMemo(() => {
    const data = props.data;
    if (!data || data.length < 2) return true;
    return data[data.length - 1] >= data[0];
  });
  
  // Generate SVG path
  const pathData = createMemo(() => {
    const data = props.data;
    if (!data || data.length === 0) return '';
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const padding = 2;
    const w = width() - padding * 2;
    const h = height() - padding * 2;
    
    const points = data.map((value, index) => {
      const x = padding + (index / (data.length - 1)) * w;
      const y = padding + h - ((value - min) / range) * h;
      return { x, y };
    });
    
    // Line path
    const linePath = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
      .join(' ');
    
    return linePath;
  });
  
  // Area path (for gradient fill)
  const areaPath = createMemo(() => {
    const data = props.data;
    if (!data || data.length === 0 || !props.showArea) return '';
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const padding = 2;
    const w = width() - padding * 2;
    const h = height() - padding * 2;
    
    const points = data.map((value, index) => {
      const x = padding + (index / (data.length - 1)) * w;
      const y = padding + h - ((value - min) / range) * h;
      return { x, y };
    });
    
    const linePath = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
      .join(' ');
    
    // Close the area
    const lastX = points[points.length - 1].x;
    const firstX = points[0].x;
    const bottom = height() - padding;
    
    return `${linePath} L ${lastX} ${bottom} L ${firstX} ${bottom} Z`;
  });
  
  // Last point for dot
  const lastPoint = createMemo(() => {
    const data = props.data;
    if (!data || data.length === 0) return null;
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const padding = 2;
    const w = width() - padding * 2;
    const h = height() - padding * 2;
    
    const index = data.length - 1;
    const value = data[index];
    const x = padding + (index / (data.length - 1)) * w;
    const y = padding + h - ((value - min) / range) * h;
    
    return { x, y };
  });

  const gradientId = createMemo(() => `sparkline-gradient-${Math.random().toString(36).slice(2)}`);

  return (
    <Show when={props.data && props.data.length > 1} fallback={
      <div 
        class={`flex items-center justify-center text-gray-600 ${props.className || ''}`}
        style={{ width: `${width()}px`, height: `${height()}px` }}
      >
        —
      </div>
    }>
      <svg
        width={width()}
        height={height()}
        class={`${props.animated ? 'transition-all duration-300' : ''} ${props.className || ''}`}
        viewBox={`0 0 ${width()} ${height()}`}
      >
        {/* Gradient definition for area fill */}
        <Show when={props.showArea}>
          <defs>
            <linearGradient id={gradientId()} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color={color()} stop-opacity="0.3" />
              <stop offset="100%" stop-color={color()} stop-opacity="0.05" />
            </linearGradient>
          </defs>
          
          {/* Area fill */}
          <path
            d={areaPath()}
            fill={`url(#${gradientId()})`}
          />
        </Show>
        
        {/* Main line */}
        <path
          d={pathData()}
          fill="none"
          stroke={color()}
          stroke-width={strokeWidth()}
          stroke-linecap="round"
          stroke-linejoin="round"
          class={props.animated ? 'animate-draw' : ''}
        />
        
        {/* End dot */}
        <Show when={props.showDots && lastPoint()}>
          <circle
            cx={lastPoint()!.x}
            cy={lastPoint()!.y}
            r={3}
            fill={color()}
            class={props.animated ? 'animate-pulse' : ''}
          />
        </Show>
      </svg>
    </Show>
  );
}

/**
 * SparklineWithValue - Sparkline with current value and change
 */
interface SparklineWithValueProps extends SparklineProps {
  value: number;
  change?: number;
  changePercent?: number;
  formatValue?: (value: number) => string;
}

export function SparklineWithValue(props: SparklineWithValueProps) {
  const formatValue = () => props.formatValue || ((v: number) => v.toFixed(2));
  const isPositive = () => (props.change ?? 0) >= 0;
  
  return (
    <div class="flex items-center gap-2">
      <Sparkline
        data={props.data}
        width={props.width}
        height={props.height}
        color={props.color}
        strokeWidth={props.strokeWidth}
        showArea={props.showArea}
        showDots={props.showDots}
        animated={props.animated}
      />
      <div class="flex flex-col items-end min-w-0">
        <span class="text-xs font-mono font-semibold text-white tabular-nums truncate">
          {formatValue()(props.value)}
        </span>
        <Show when={props.changePercent !== undefined}>
          <span class={`text-[10px] font-mono tabular-nums ${isPositive() ? 'text-success-400' : 'text-danger-400'}`}>
            {isPositive() ? '+' : ''}{props.changePercent?.toFixed(2)}%
          </span>
        </Show>
      </div>
    </div>
  );
}

export default Sparkline;
