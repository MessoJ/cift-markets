/**
 * Multi-Panel Indicator Charts
 * 
 * Renders technical indicators (MACD, RSI) in separate panels below the main chart.
 * Synchronized zoom/pan with main chart.
 * 
 * Uses ECharts grid system for efficient multi-chart rendering.
 */

import { createEffect, onMount, onCleanup, Show } from 'solid-js';
import * as echarts from 'echarts';
import type { IndicatorData } from '~/hooks/useIndicators';
import { DARK_THEME } from '~/types/chart.types';

export interface IndicatorPanelsProps {
  indicators: IndicatorData[];
  activeIndicatorIds: string[];
  height?: string;
  onZoomChange?: (start: number, end: number) => void;
}

interface PanelConfig {
  id: string;
  title: string;
  height: number; // Percentage of total height
  gridIndex: number;
  yAxisIndex: number;
}

export default function IndicatorPanels(props: IndicatorPanelsProps) {
  let chartContainer: HTMLDivElement | undefined;
  let chartInstance: echarts.ECharts | undefined;

  /**
   * Determine which panels to show based on active indicators
   */
  const getActivePanels = (): PanelConfig[] => {
    const panels: PanelConfig[] = [];
    let gridIndex = 0;

    // Check for MACD
    if (props.activeIndicatorIds.includes('macd')) {
      panels.push({
        id: 'macd',
        title: 'MACD',
        height: 25,
        gridIndex: gridIndex++,
        yAxisIndex: gridIndex,
      });
    }

    // Check for RSI
    if (props.activeIndicatorIds.some(id => id.startsWith('rsi_'))) {
      panels.push({
        id: 'rsi',
        title: 'RSI',
        height: 20,
        gridIndex: gridIndex++,
        yAxisIndex: gridIndex,
      });
    }

    return panels;
  };

  /**
   * Extract indicator data by column name
   */
  const extractIndicatorData = (columnName: string): [number, number][] => {
    return props.indicators
      .filter(d => d[columnName] !== null && d[columnName] !== undefined)
      .map(d => [d.timestamp, d[columnName] as number]);
  };

  /**
   * Generate MACD series configuration
   */
  const generateMACDSeries = (gridIndex: number, yAxisIndex: number) => {
    const macdLine = extractIndicatorData('macd');
    const signalLine = extractIndicatorData('macd_signal');
    const histogram = extractIndicatorData('macd_histogram');

    return [
      {
        name: 'MACD',
        type: 'line',
        data: macdLine,
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
        lineStyle: { color: '#3b82f6', width: 2 },
        showSymbol: false,
        smooth: true,
      },
      {
        name: 'Signal',
        type: 'line',
        data: signalLine,
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
        lineStyle: { color: '#f97316', width: 2 },
        showSymbol: false,
        smooth: true,
      },
      {
        name: 'Histogram',
        type: 'bar',
        data: histogram.map(([ts, val]) => [
          ts,
          val,
          val >= 0 ? '#10b981' : '#ef4444', // Green for positive, red for negative
        ]),
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
        itemStyle: {
          color: (params: any) => params.data[2],
        },
        barWidth: '60%',
      },
    ];
  };

  /**
   * Generate RSI series configuration
   */
  const generateRSISeries = (gridIndex: number, yAxisIndex: number) => {
    const rsi14 = extractIndicatorData('rsi_14');
    const rsi7 = extractIndicatorData('rsi_7');

    return [
      {
        name: 'RSI(14)',
        type: 'line',
        data: rsi14,
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
        lineStyle: { color: '#a855f7', width: 2 },
        showSymbol: false,
        smooth: true,
      },
      // Show RSI(7) if available
      ...(rsi7.length > 0 ? [{
        name: 'RSI(7)',
        type: 'line',
        data: rsi7,
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
        lineStyle: { color: '#ec4899', width: 1, type: 'dashed' as const },
        showSymbol: false,
        smooth: true,
      }] : []),
      // Overbought line (70)
      {
        name: 'Overbought',
        type: 'line',
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: {
            color: '#ef4444',
            type: 'dashed',
            width: 1,
          },
          data: [{ yAxis: 70 }],
          label: {
            show: true,
            position: 'end',
            formatter: '70',
            color: '#ef4444',
          },
        },
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
      },
      // Oversold line (30)
      {
        name: 'Oversold',
        type: 'line',
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: {
            color: '#10b981',
            type: 'dashed',
            width: 1,
          },
          data: [{ yAxis: 30 }],
          label: {
            show: true,
            position: 'end',
            formatter: '30',
            color: '#10b981',
          },
        },
        xAxisIndex: gridIndex,
        yAxisIndex: yAxisIndex,
      },
    ];
  };

  /**
   * Initialize/Update ECharts instance
   */
  createEffect(() => {
    if (!chartContainer || props.indicators.length === 0) return;

    const activePanels = getActivePanels();
    if (activePanels.length === 0) {
      // No panels to show, clear chart
      if (chartInstance) {
        chartInstance.clear();
      }
      return;
    }

    // Initialize chart if needed
    if (!chartInstance) {
      chartInstance = echarts.init(chartContainer, DARK_THEME);
    }

    // Calculate grid positions
    const panelSpacing = 2; // % spacing between panels
    let currentTop = 0;

    const grids: any[] = [];
    const xAxes: any[] = [];
    const yAxes: any[] = [];
    const series: any[] = [];

    activePanels.forEach((panel, index) => {
      // Grid configuration
      grids.push({
        left: '5%',
        right: '5%',
        top: `${currentTop}%`,
        height: `${panel.height}%`,
        containLabel: true,
      });

      // X-Axis configuration
      xAxes.push({
        type: 'time',
        gridIndex: panel.gridIndex,
        axisLine: { lineStyle: { color: '#374151' } },
        axisLabel: {
          show: index === activePanels.length - 1, // Only show on last panel
          color: '#9ca3af',
        },
        splitLine: { show: false },
      });

      // Y-Axis configuration
      yAxes.push({
        type: 'value',
        gridIndex: panel.gridIndex,
        scale: true,
        axisLine: { lineStyle: { color: '#374151' } },
        axisLabel: { color: '#9ca3af' },
        splitLine: {
          lineStyle: { color: '#1f2937', type: 'dashed' },
        },
        name: panel.title,
        nameTextStyle: { color: '#9ca3af', fontSize: 12 },
        nameLocation: 'middle',
        nameGap: 50,
      });

      // Generate series for this panel
      if (panel.id === 'macd') {
        series.push(...generateMACDSeries(panel.gridIndex, panel.yAxisIndex));
      } else if (panel.id === 'rsi') {
        series.push(...generateRSISeries(panel.gridIndex, panel.yAxisIndex));
      }

      currentTop += panel.height + panelSpacing;
    });

    // Configure ECharts option
    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      animation: false,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series: series,
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          link: [{ xAxisIndex: 'all' }],
        },
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        borderColor: '#374151',
        textStyle: { color: '#f9fafb' },
        formatter: (params: any) => {
          if (!params || params.length === 0) return '';
          
          const timestamp = new Date(params[0].value[0]).toLocaleString();
          let tooltip = `<div style="font-size: 12px;"><b>${timestamp}</b><br/>`;
          
          params.forEach((param: any) => {
            const value = typeof param.value[1] === 'number' 
              ? param.value[1].toFixed(2) 
              : param.value[1];
            tooltip += `${param.marker} ${param.seriesName}: <b>${value}</b><br/>`;
          });
          
          tooltip += '</div>';
          return tooltip;
        },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: Array.from({ length: activePanels.length }, (_, i) => i),
          start: 0,
          end: 100,
        },
      ],
      legend: {
        show: false,
      },
    };

    chartInstance.setOption(option, true);

    console.log(`📊 Rendered ${activePanels.length} indicator panels:`, activePanels.map(p => p.title).join(', '));
  });

  // Handle resize
  createEffect(() => {
    if (!chartInstance) return;

    const handleResize = () => {
      chartInstance?.resize();
    };

    window.addEventListener('resize', handleResize);
    onCleanup(() => {
      window.removeEventListener('resize', handleResize);
    });
  });

  // Cleanup
  onCleanup(() => {
    if (chartInstance) {
      chartInstance.dispose();
      chartInstance = undefined;
    }
  });

  const activePanels = getActivePanels();
  const shouldShow = activePanels.length > 0 && props.indicators.length > 0;

  return (
    <Show when={shouldShow}>
      <div 
        ref={chartContainer}
        style={{
          width: '100%',
          height: props.height || '300px',
          'background-color': '#111827',
          'border-top': '1px solid #1f2937',
        }}
      />
    </Show>
  );
}
