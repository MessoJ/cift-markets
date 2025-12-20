/**
 * Indicator Utilities
 * 
 * Transform indicator data for ECharts visualization.
 */

import type { IndicatorData } from '~/hooks/useIndicators';
import type { IndicatorConfig } from '~/components/charts/IndicatorPanel';

/**
 * Extract series data for a specific indicator (category-axis compatible)
 * Returns just the values, aligned to category indices
 */
export function extractIndicatorSeries(
  data: IndicatorData[],
  indicatorKey: string
): (number | null)[] {
  return data.map((point) => {
    const value = point[indicatorKey];
    return value !== undefined && value !== null ? Number(value) : null;
  });
}

/**
 * Generate ECharts series configuration for an indicator
 */
export function generateIndicatorSeries(
  data: IndicatorData[],
  config: IndicatorConfig
) {
  const seriesData = extractIndicatorSeries(data, config.id);

  return {
    name: config.name,
    type: 'line',
    data: seriesData,
    smooth: true,
    showSymbol: false,
    lineStyle: {
      color: config.color || '#3b82f6',
      width: 2,
      opacity: 1,
    },
    z: 10,  // Render above candles
    emphasis: {
      focus: 'series',
    },
  };
}

/**
 * Generate Bollinger Bands series (upper, middle, lower)
 */
export function generateBollingerBandsSeries(data: IndicatorData[], color: string = '#ec4899') {
  const upperData = extractIndicatorSeries(data, 'bb_upper');
  const middleData = extractIndicatorSeries(data, 'bb_middle');
  const lowerData = extractIndicatorSeries(data, 'bb_lower');

  return [
    {
      name: 'BB Upper',
      type: 'line',
      data: upperData,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        color: color,
        width: 1,
        type: 'dashed',
      },
      emphasis: {
        disabled: true,
      },
    },
    {
      name: 'BB Middle',
      type: 'line',
      data: middleData,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        color: color,
        width: 1.5,
      },
      emphasis: {
        focus: 'series',
      },
    },
    {
      name: 'BB Lower',
      type: 'line',
      data: lowerData,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        color: color,
        width: 1,
        type: 'dashed',
      },
      emphasis: {
        disabled: true,
      },
      // Area fill between bands
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            {
              offset: 0,
              color: `${color}20`,
            },
            {
              offset: 1,
              color: `${color}05`,
            },
          ],
        },
      },
    },
  ];
}

/**
 * Generate MACD panel series (separate from price chart)
 */
export function generateMACDSeries(data: IndicatorData[]) {
  const macdData = extractIndicatorSeries(data, 'macd');
  const signalData = extractIndicatorSeries(data, 'macd_signal');
  const histogramData = extractIndicatorSeries(data, 'macd_histogram');

  return {
    macd: {
      name: 'MACD',
      type: 'line',
      data: macdData,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        color: '#3b82f6',
        width: 2,
      },
    },
    signal: {
      name: 'Signal',
      type: 'line',
      data: signalData,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        color: '#f59e0b',
        width: 2,
      },
    },
    histogram: {
      name: 'Histogram',
      type: 'bar',
      data: histogramData,
      itemStyle: {
        color: (params: any) => {
          const value = params.value[1];
          return value >= 0 ? '#10b981' : '#ef4444';
        },
      },
    },
  };
}

/**
 * Check if indicator requires separate panel (not overlaid on price)
 */
export function requiresSeparatePanel(indicatorId: string): boolean {
  const separatePanelIndicators = ['macd', 'rsi_14', 'stoch'];
  return separatePanelIndicators.includes(indicatorId);
}
