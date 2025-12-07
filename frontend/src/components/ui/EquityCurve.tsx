/**
 * EquityCurve Component
 * 
 * Interactive equity/performance curve chart.
 * Shows portfolio value over time with benchmark comparison.
 * 
 * Design System: Professional financial visualization
 */

import { createEffect } from 'solid-js';
import { useECharts } from '~/hooks/useECharts';
import type { EChartsOption } from 'echarts';

export interface EquityDataPoint {
  timestamp: number | string;
  value: number;
  benchmark?: number;
}

interface EquityCurveProps {
  data: EquityDataPoint[];
  benchmarkData?: EquityDataPoint[];
  benchmarkLabel?: string;
  height?: number;
  showBenchmark?: boolean;
  showDrawdown?: boolean;
  showGrid?: boolean;
  animated?: boolean;
  className?: string;
  onRangeChange?: (start: number, end: number) => void;
}

export function EquityCurve(props: EquityCurveProps) {
  let chartContainer: HTMLDivElement | undefined;
  
  const height = () => props.height || 300;
  
  // Calculate performance metrics
  const metrics = () => {
    const data = props.data;
    if (!data || data.length < 2) {
      return { totalReturn: 0, maxDrawdown: 0, currentValue: 0, startValue: 0 };
    }
    
    const startValue = data[0].value;
    const currentValue = data[data.length - 1].value;
    const totalReturn = ((currentValue - startValue) / startValue) * 100;
    
    // Calculate max drawdown
    let peak = startValue;
    let maxDrawdown = 0;
    
    for (const point of data) {
      if (point.value > peak) peak = point.value;
      const drawdown = ((peak - point.value) / peak) * 100;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }
    
    return { totalReturn, maxDrawdown, currentValue, startValue };
  };
  
  // Generate chart options
  const generateOptions = (): EChartsOption => {
    const data = props.data;
    if (!data || data.length === 0) {
      return {
        title: {
          text: 'No data available',
          left: 'center',
          top: 'center',
          textStyle: { color: '#6b7280' },
        },
      } as EChartsOption;
    }
    
    const portfolioData = data.map(d => [
      typeof d.timestamp === 'string' ? new Date(d.timestamp).getTime() : d.timestamp,
      d.value,
    ]);
    
    const benchmarkData = props.showBenchmark && props.benchmarkData
      ? props.benchmarkData.map(d => [
          typeof d.timestamp === 'string' ? new Date(d.timestamp).getTime() : d.timestamp,
          d.value,
        ])
      : null;
    
    return {
      backgroundColor: 'transparent',
      animation: props.animated !== false,
      animationDuration: 500,
      
      tooltip: {
        trigger: 'axis' as const,
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        borderColor: '#374151',
        textStyle: { color: '#f9fafb', fontSize: 12 },
        formatter: (params: any) => {
          if (!params || params.length === 0) return '';
          
          const date = new Date(params[0].value[0]);
          let html = `<div class="font-medium mb-1">${date.toLocaleDateString()}</div>`;
          
          for (const p of params) {
            const color = p.seriesName === 'Portfolio' ? '#3b82f6' : '#9ca3af';
            html += `
              <div class="flex items-center gap-2">
                <span style="color: ${color}">●</span>
                <span class="text-gray-400">${p.seriesName}:</span>
                <span class="font-mono font-semibold">$${p.value[1].toLocaleString()}</span>
              </div>
            `;
          }
          
          return html;
        },
      },
      
      legend: {
        show: props.showBenchmark && benchmarkData !== null,
        bottom: 0,
        textStyle: { color: '#9ca3af', fontSize: 11 },
        data: ['Portfolio', props.benchmarkLabel || 'Benchmark'],
      },
      
      grid: {
        left: '3%',
        right: '3%',
        top: '8%',
        bottom: props.showBenchmark ? '15%' : '10%',
        containLabel: true,
      },
      
      xAxis: {
        type: 'time' as const,
        axisLine: { lineStyle: { color: '#374151' } },
        axisLabel: { color: '#9ca3af', fontSize: 10 },
        splitLine: { show: props.showGrid, lineStyle: { color: '#1f2937', type: 'dashed' as const } },
      },
      
      yAxis: {
        type: 'value' as const,
        scale: true,
        axisLine: { lineStyle: { color: '#374151' } },
        axisLabel: {
          color: '#9ca3af',
          fontSize: 10,
          formatter: (value: number) => `$${(value / 1000).toFixed(0)}k`,
        },
        splitLine: { show: props.showGrid, lineStyle: { color: '#1f2937', type: 'dashed' as const } },
      },
      
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
      ],
      
      series: [
        {
          name: 'Portfolio',
          type: 'line' as const,
          data: portfolioData,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            color: '#3b82f6',
            width: 2,
          },
          areaStyle: {
            color: {
              type: 'linear' as const,
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.05)' },
              ],
            },
          },
        },
        ...(benchmarkData ? [{
          name: props.benchmarkLabel || 'Benchmark',
          type: 'line' as const,
          data: benchmarkData,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            color: '#9ca3af',
            width: 1.5,
            type: 'dashed' as const,
          },
        }] : []),
      ],
    } as EChartsOption;
  };
  
  const chart = useECharts(() => ({
    container: chartContainer,
    options: generateOptions(),
    loading: false,
    autoResize: true,
    theme: 'dark',
  }));
  
  // Update chart when data changes
  createEffect(() => {
    if (chart.getInstance() && props.data) {
      chart.updateChart(generateOptions(), false);
    }
  });

  return (
    <div class={`flex flex-col ${props.className || ''}`}>
      {/* Metrics Header */}
      <div class="flex items-center gap-6 mb-3 px-1">
        <div>
          <span class="text-[10px] text-gray-500 uppercase tracking-wide">Total Return</span>
          <div class={`text-lg font-bold font-mono tabular-nums ${metrics().totalReturn >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
            {metrics().totalReturn >= 0 ? '+' : ''}{metrics().totalReturn.toFixed(2)}%
          </div>
        </div>
        <div>
          <span class="text-[10px] text-gray-500 uppercase tracking-wide">Max Drawdown</span>
          <div class="text-lg font-bold font-mono tabular-nums text-danger-400">
            -{metrics().maxDrawdown.toFixed(2)}%
          </div>
        </div>
        <div>
          <span class="text-[10px] text-gray-500 uppercase tracking-wide">Current Value</span>
          <div class="text-lg font-bold font-mono tabular-nums text-white">
            ${metrics().currentValue.toLocaleString()}
          </div>
        </div>
      </div>
      
      {/* Chart */}
      <div 
        ref={chartContainer}
        style={{ height: `${height()}px` }}
        class="w-full"
      />
    </div>
  );
}

/**
 * MiniEquityCurve - Compact version for cards
 */
interface MiniEquityCurveProps {
  data: Array<{ date?: string; timestamp?: number | string; value: number }>;
  width?: number;
  height?: number;
  className?: string;
}

export function MiniEquityCurve(props: MiniEquityCurveProps) {
  const width = () => props.width || 120;
  const height = () => props.height || 40;
  
  const pathData = () => {
    const data = props.data;
    if (!data || data.length < 2) return '';
    
    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    
    const points = data.map((d, i) => {
      const x = (i / (data.length - 1)) * width();
      const y = height() - ((d.value - min) / range) * height();
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    });
    
    return points.join(' ');
  };
  
  const isPositive = () => {
    const data = props.data;
    if (!data || data.length < 2) return true;
    return data[data.length - 1].value >= data[0].value;
  };

  return (
    <svg 
      width={width()} 
      height={height()} 
      class={props.className}
      viewBox={`0 0 ${width()} ${height()}`}
    >
      <path
        d={pathData()}
        fill="none"
        stroke={isPositive() ? '#22c55e' : '#ef4444'}
        stroke-width={1.5}
        stroke-linecap="round"
        stroke-linejoin="round"
      />
    </svg>
  );
}

export default EquityCurve;
