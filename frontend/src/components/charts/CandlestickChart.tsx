/**
 * Advanced Candlestick Chart Component
 * 
 * Production-grade financial charting with:
 * - Real-time data from QuestDB via FastAPI
 * - ECharts for GPU-accelerated rendering
 * - Volume analysis
 * - Prepared for ML model overlays (Hawkes process)
 * - Zero mock data - all from database
 */

import { createSignal, createEffect, on, Show, onCleanup } from 'solid-js';
import { useECharts } from '~/hooks/useECharts';
import { useMarketDataWebSocket } from '~/hooks/useMarketDataWebSocket';
import { useIndicators } from '~/hooks/useIndicators';
import { apiClient } from '~/lib/api/client';
import type { EChartsOption } from 'echarts';
import type { OHLCVBar, TickUpdate, CandleUpdate } from '~/types/chart.types';
import type { IndicatorConfig } from '~/components/charts/IndicatorPanel';
import type { DrawingType, Drawing, DrawingPoint } from '~/types/drawing.types';
import {
  transformToEChartsData,
  transformToVolumeData,
  transformToHeikinAshi,
  formatPrice,
  formatVolume,
  getLatestBarInfo,
  validateAndFilterBars,
} from '~/lib/utils/chart.utils';
import {
  generateIndicatorSeries,
  generateBollingerBandsSeries,
  requiresSeparatePanel,
} from '~/lib/utils/indicator.utils';
import { DARK_THEME } from '~/types/chart.types';

export interface CandlestickChartProps {
  symbol: string;
  timeframe: string;
  candleLimit?: number;
  showVolume?: boolean;
  chartType?: 'candlestick' | 'line' | 'area' | 'heikin_ashi';
  enableRealTime?: boolean;
  activeIndicators?: IndicatorConfig[];
  activeTool?: DrawingType | null;
  drawings?: Drawing[];
  selectedDrawingId?: string | null;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  onIndicatorsChange?: (indicators: IndicatorConfig[]) => void;
  onDrawingComplete?: (drawing: Partial<Drawing>) => void;
  onDrawingSelect?: (drawingId: string | null) => void;
  height?: string;
}

export default function CandlestickChart(props: CandlestickChartProps) {
  // ============================================================================
  // STATE
  // ============================================================================
  
  let chartContainer: HTMLDivElement | undefined;
  const [bars, setBars] = createSignal<OHLCVBar[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [latestInfo, setLatestInfo] = createSignal<ReturnType<typeof getLatestBarInfo>>(null);
  const [livePrice, setLivePrice] = createSignal<number | null>(null);

  // Drawing tools state
  const [drawingPoints, setDrawingPoints] = createSignal<DrawingPoint[]>([]);
  const [tempDrawing, setTempDrawing] = createSignal<Partial<Drawing> | null>(null);

  // ============================================================================
  // WEBSOCKET REAL-TIME UPDATES
  // ============================================================================

  const enableRealTime = props.enableRealTime !== false; // Default true
  
  const ws = useMarketDataWebSocket({
    autoConnect: enableRealTime,
  });

  /**
   * Handle real-time price updates from WebSocket
   */
  const handlePriceUpdate = (tick: TickUpdate) => {
    if (tick.symbol !== props.symbol) return;
    
    // Update live price display
    setLivePrice(tick.price);
    
    // Update latest info for overlay
    const current = latestInfo();
    if (current) {
      const updatedInfo = {
        ...current,
        price: tick.price,
        change: tick.price - current.open,
        changePercent: ((tick.price - current.open) / current.open) * 100,
        direction: tick.price >= current.open ? 'up' as const : 'down' as const,
      };
      setLatestInfo(updatedInfo);
    }

    console.debug(`💹 Live tick: ${tick.symbol} @ $${formatPrice(tick.price)}`);
  };

  /**
   * Handle real-time candle updates from WebSocket
   */
  const handleCandleUpdate = (candle: CandleUpdate) => {
    if (candle.symbol !== props.symbol || candle.timeframe !== props.timeframe) return;
    
    setBars(prev => {
      if (prev.length === 0) return prev;
      
      const lastBar = prev[prev.length - 1];
      const candleTimestamp = new Date(candle.timestamp).getTime();
      
      // Update last bar if it's the same timestamp
      if (lastBar.timestamp === candleTimestamp) {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...lastBar,
          high: Math.max(lastBar.high, candle.high),
          low: Math.min(lastBar.low, candle.low),
          close: candle.close,
          volume: candle.volume,
        };
        
        console.debug(`📊 Candle update: ${candle.symbol} ${candle.timeframe} - H:${candle.high} L:${candle.low} C:${candle.close}`);
        return updated;
      }
      
      // Add new bar if it's a new timestamp (candle closed)
      if (candle.is_closed && candleTimestamp > (typeof lastBar.timestamp === 'string' ? new Date(lastBar.timestamp).getTime() : lastBar.timestamp)) {
        const newBar: OHLCVBar = {
          timestamp: candleTimestamp,
          symbol: candle.symbol,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          volume: candle.volume,
        };
        
        console.info(`🕯️ New candle: ${candle.symbol} ${candle.timeframe} @ ${new Date(candleTimestamp).toISOString()}`);
        return [...prev, newBar];
      }
      
      return prev;
    });
  };

  // Register WebSocket callbacks
  let unsubscribeTick: (() => void) | undefined;
  let unsubscribeCandle: (() => void) | undefined;
  
  createEffect(() => {
    if (enableRealTime && ws.isConnected()) {
      // Clean up previous subscriptions
      if (unsubscribeTick) {
        unsubscribeTick();
      }
      if (unsubscribeCandle) {
        unsubscribeCandle();
      }
      
      // Subscribe to price and candle updates
      unsubscribeTick = ws.onTick(handlePriceUpdate);
      unsubscribeCandle = ws.onCandle(handleCandleUpdate);
      
      // Subscribe to symbol
      ws.subscribe([props.symbol]);
      console.info(`🔔 Subscribed to real-time updates for ${props.symbol} (${props.timeframe})`);
    }
  });

  // Cleanup WebSocket subscription on unmount
  onCleanup(() => {
    if (unsubscribeTick) {
      unsubscribeTick();
    }
    if (unsubscribeCandle) {
      unsubscribeCandle();
    }
    if (enableRealTime && ws.isConnected()) {
      ws.unsubscribe([props.symbol]);
    }
  });

  // ============================================================================
  // TECHNICAL INDICATORS
  // ============================================================================

  const activeIndicators = () => props.activeIndicators || [];
  const enabledIndicatorIds = () => 
    activeIndicators()
      .filter((ind) => ind.enabled)
      .map((ind) => ind.id);

  const indicatorData = useIndicators({
    symbol: props.symbol,
    timeframe: props.timeframe,
    limit: props.candleLimit || 500,
    indicators: enabledIndicatorIds(),
    enabled: enabledIndicatorIds().length > 0,
  });

  // ============================================================================
  // DATA FETCHING FROM DATABASE
  // ============================================================================

  /**
   * Fetch OHLCV data from backend (QuestDB via FastAPI)
   * Uses QuestDB's SAMPLE BY for optimal performance
   */
  const fetchChartData = async () => {
    setLoading(true);
    setError(null);

    try {
      console.info(`📊 Fetching ${props.candleLimit || 500} bars for ${props.symbol} (${props.timeframe})`);
      
      const data = await apiClient.getBars(
        props.symbol,
        props.timeframe,
        props.candleLimit || 500
      );

      if (!data || data.length === 0) {
        throw new Error(`No data available for ${props.symbol}`);
      }

      // Validate data integrity
      const validBars = validateAndFilterBars(data);
      
      if (validBars.length === 0) {
        throw new Error('All bars failed validation');
      }

      // Sort by timestamp ascending (oldest first)
      validBars.sort((a, b) => {
        const timeA = typeof a.timestamp === 'string' ? new Date(a.timestamp).getTime() : a.timestamp;
        const timeB = typeof b.timestamp === 'string' ? new Date(b.timestamp).getTime() : b.timestamp;
        return timeA - timeB;
      });

      setBars(validBars);
      setLatestInfo(getLatestBarInfo(validBars));
      
      console.info(`✅ Loaded ${validBars.length} bars from database`);
    } catch (err: any) {
      console.error('❌ Chart data fetch failed:', err);
      console.error('Error details:', {
        message: err.message,
        stack: err.stack,
        response: err.response,
      });
      setError(err.message || 'Failed to load chart data');
    } finally {
      setLoading(false);
    }
  };

  // ============================================================================
  // ECHART CONFIGURATION
  // ============================================================================



  /**
   * Generate ECharts options from OHLCV data
   * Optimized for performance with large datasets
   */
  const generateChartOptions = (): EChartsOption => {
    const currentBars = bars();
    
    if (currentBars.length === 0) {
      return {
        title: {
          text: 'No data available',
          left: 'center',
          top: 'center',
          textStyle: { color: DARK_THEME.textColor },
        },
      };
    }

    // Transform data for ECharts
    let displayBars = currentBars;
    if (props.chartType === 'heikin_ashi') {
      displayBars = transformToHeikinAshi(currentBars);
    }

    // Generate category labels (TradingView-style: no gaps for non-trading hours)
    const categoryLabels = displayBars.map((bar) => {
      const date = new Date(bar.timestamp);
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      });
    });

    // Transform to category-based data (index-based, not time-based)
    const candleData = displayBars.map((bar) => [bar.open, bar.close, bar.low, bar.high]);
    const volumeData = currentBars.map((bar) => {
      const direction = bar.close >= bar.open ? 1 : -1;
      return [bar.volume, direction];
    });

    // Calculate grid layout based on volume visibility
    const showVol = props.showVolume !== false;
    const mainGridHeight = showVol ? '60%' : '85%';
    const volumeGridTop = showVol ? '68%' : undefined;

    const option: EChartsOption = {
      backgroundColor: DARK_THEME.background,
      animation: true,
      animationDuration: 300,
      animationEasing: 'cubicOut',

      // Tooltip configuration
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          lineStyle: {
            color: DARK_THEME.gridColor,
            type: 'dashed',
          },
        },
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        borderColor: DARK_THEME.gridColor,
        textStyle: {
          color: DARK_THEME.textColor,
          fontSize: 12,
        },
        formatter: (params: any) => {
          if (!params || params.length === 0) return '';
          
          const candleParam = params[0];
          if (!candleParam || !candleParam.data) return '';
          
          // Category axis format: [open, close, low, high]
          const [open, close, low, high] = candleParam.data;
          const dateLabel = candleParam.axisValue || candleParam.name; // Use category label
          const volumeParam = params.find((p: any) => p.seriesName === 'Volume');
          
          const change = close - open;
          const changePercent = ((change / open) * 100).toFixed(2);
          const changeColor = change >= 0 ? DARK_THEME.bullish : DARK_THEME.bearish;
          
          return `
            <div style="padding: 8px;">
              <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">
                ${dateLabel}
              </div>
              <div style="display: grid; grid-template-columns: auto auto; gap: 4px 12px; font-size: 12px;">
                <span style="color: #999;">Open:</span>
                <span style="font-weight: 600;">${formatPrice(open)}</span>
                
                <span style="color: #999;">High:</span>
                <span style="font-weight: 600; color: ${DARK_THEME.bullish};">${formatPrice(high)}</span>
                
                <span style="color: #999;">Low:</span>
                <span style="font-weight: 600; color: ${DARK_THEME.bearish};">${formatPrice(low)}</span>
                
                <span style="color: #999;">Close:</span>
                <span style="font-weight: 600;">${formatPrice(close)}</span>
                
                <span style="color: #999;">Change:</span>
                <span style="font-weight: 600; color: ${changeColor};">
                  ${change >= 0 ? '+' : ''}${formatPrice(change)} (${changePercent}%)
                </span>
                
                ${volumeParam ? `
                  <span style="color: #999;">Volume:</span>
                  <span style="font-weight: 600;">${formatVolume(volumeParam.data[0])}</span>
                ` : ''}
              </div>
            </div>
          `;
        },
      },

      // Grid layout
      grid: [
        {
          left: '3%',
          right: '3%',
          top: '8%',
          height: mainGridHeight,
          containLabel: true,
        },
        ...(showVol
          ? [
              {
                left: '3%',
                right: '3%',
                top: volumeGridTop,
                height: '18%',
                containLabel: true,
              },
            ]
          : []),
      ],

      // X-Axis configuration (category axis - no gaps like TradingView)
      xAxis: [
        {
          type: 'category',
          data: categoryLabels,
          gridIndex: 0,
          axisLine: {
            lineStyle: { color: DARK_THEME.gridColor },
          },
          axisLabel: {
            color: DARK_THEME.textColor,
            fontSize: 11,
            rotate: 0,
            interval: 'auto',
          },
          axisTick: {
            alignWithLabel: true,
          },
          splitLine: {
            show: false,
          },
          boundaryGap: true,
        },
        ...(showVol
          ? [
              {
                type: 'category' as const,
                data: categoryLabels,
                gridIndex: 1,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { show: false },
                splitLine: { show: false },
              },
            ]
          : []),
      ],

      // Y-Axis configuration
      yAxis: [
        {
          type: 'value',
          gridIndex: 0,
          scale: true,
          splitLine: {
            show: true,
            lineStyle: {
              color: DARK_THEME.gridColor,
              type: 'dashed',
            },
          },
          axisLine: {
            lineStyle: { color: DARK_THEME.gridColor },
          },
          axisLabel: {
            color: DARK_THEME.textColor,
            fontSize: 11,
            formatter: (value: number) => '$' + value.toFixed(2),  // Professional format: $170.52
          },
        },
        ...(showVol
          ? [
              {
                type: 'value' as const,
                gridIndex: 1,
                scale: true,
                splitLine: { show: false },
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: {
                  show: true,
                  color: DARK_THEME.textColor,
                  fontSize: 10,
                  formatter: (value: number) => formatVolume(value),
                },
              },
            ]
          : []),
      ],

      // Data zoom for interactive navigation
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: showVol ? [0, 1] : [0],
          start: 0,
          end: 100,
          minValueSpan: 10, // Minimum 10 candles visible
        },
        {
          type: 'slider',
          xAxisIndex: showVol ? [0, 1] : [0],
          start: 0,
          end: 100,
          bottom: '2%',
          height: 20,
          borderColor: DARK_THEME.gridColor,
          fillerColor: 'rgba(249, 115, 22, 0.1)',
          handleStyle: {
            color: '#f97316',
            borderColor: '#ea580c',
          },
          textStyle: {
            color: DARK_THEME.textColor,
          },
          dataBackground: {
            lineStyle: { color: DARK_THEME.gridColor },
            areaStyle: { color: 'rgba(249, 115, 22, 0.05)' },
          },
        },
      ],

      // Series data
      series: [
        // Price series - Candlestick or Line based on chartType
        ...(props.chartType === 'line' || props.chartType === 'area' ? [
          {
            name: 'Price',
            type: 'line',
            data: candleData.map((d: any) => d[1]),  // close price (index 1 in new format)
            smooth: false,
            showSymbol: false,
            lineStyle: {
              color: DARK_THEME.bullish,
              width: 2,
            },
            areaStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: 'rgba(38, 166, 154, 0.3)' },
                  { offset: 1, color: 'rgba(38, 166, 154, 0.05)' },
                ],
              },
            },
          },
        ] : [
          {
            name: props.chartType === 'heikin_ashi' ? 'Heikin Ashi' : 'Price',
            type: 'candlestick',
            data: candleData,
            itemStyle: {
              color: DARK_THEME.bullish, // Bullish candle (close > open)
              color0: DARK_THEME.bearish, // Bearish candle (close < open)
              borderColor: DARK_THEME.bullish,
              borderColor0: DARK_THEME.bearish,
              borderWidth: 1,
            },
            // TradingView-style: bars with minimal gaps
            barMinWidth: 2,
            barMaxWidth: 20,
            emphasis: {
              itemStyle: {
                borderWidth: 2,
              },
            },
          },
        ]),
        // Volume series (if enabled)
        ...(showVol
          ? [
              {
                name: 'Volume',
                type: 'bar',
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: volumeData.map((d: any) => d[0]),  // Just volume value
                barMinWidth: 2,
                barMaxWidth: 20,
                itemStyle: {
                  color: (params: any) => {
                    // Color based on direction: up (green) or down (red)
                    const direction = volumeData[params.dataIndex]?.[1] || 1;
                    return direction > 0 ? DARK_THEME.volumeUp : DARK_THEME.volumeDown;
                  },
                },
                emphasis: {
                  itemStyle: {
                    opacity: 0.8,
                  },
                },
              },
            ]
          : []),
        // Technical Indicators
        ...(() => {
          const indicators = indicatorData.data();
          const enabledIndicators = activeIndicators().filter((ind) => ind.enabled);
          
          console.log('📊 Indicator Debug:', {
            indicatorsDataLength: indicators?.length || 0,
            enabledIndicatorIds: enabledIndicators.map(i => i.id),
            indicatorsDataSample: indicators?.[0],
            indicatorDataError: indicatorData.error(),
            indicatorDataLoading: indicatorData.loading(),
          });
          
          if (!indicators || indicators.length === 0) {
            console.warn('⚠️ No indicator data available', {
              hasIndicators: !!indicators,
              length: indicators?.length,
              error: indicatorData.error(),
            });
            return [];
          }

          const series: any[] = [];

          for (const indicator of enabledIndicators) {
            // Skip indicators that need separate panels
            if (requiresSeparatePanel(indicator.id)) {
              console.log(`⏭️ Skipping ${indicator.id} (separate panel required)`);
              continue;
            }

            // Handle Bollinger Bands specially (3 lines)
            if (indicator.id === 'bb_bands') {
              const bbSeries = generateBollingerBandsSeries(indicators, indicator.color || '#ec4899');
              console.log(`✅ Added Bollinger Bands (3 series)`);
              series.push(...bbSeries);
            } else {
              // Regular indicator (SMA, EMA, etc.)
              const indicatorSeries = generateIndicatorSeries(indicators, indicator);
              console.log(`✅ Added ${indicator.name} indicator`, {
                id: indicator.id,
                dataPoints: indicatorSeries.data.length,
                validPoints: indicatorSeries.data.filter((d: any) => d[1] !== null).length,
                sampleData: indicatorSeries.data.slice(0, 3),
              });
              series.push(indicatorSeries);
            }
          }

          console.log(`📈 Total indicator series added: ${series.length}`, series.map((s: any) => s.name));
          return series;
        })(),
        // Drawing tools (trendlines, shapes, annotations)
        ...generateDrawingSeries(),
      ],
    };

    return option;
  };

  // ============================================================================
  // DRAWING TOOLS LOGIC
  // ============================================================================

  /**
   * Handle chart click for drawing tools AND selection
   */
  const handleChartClick = (params: any) => {
    // Check if user clicked on an existing drawing
    if (params && params.seriesId) {
      const clickedDrawingId = params.seriesId;
      const isDrawing = props.drawings?.some(d => d.id === clickedDrawingId);
      
      if (isDrawing) {
        console.log('🎯 Drawing clicked:', clickedDrawingId);
        props.onDrawingSelect?.(clickedDrawingId);
        return; // Don't create new drawing
      }
    }
    
    // If no active tool, deselect any selected drawing
    if (!props.activeTool) {
      if (props.selectedDrawingId) {
        console.log('❌ Deselect drawing');
        props.onDrawingSelect?.(null);
      }
      return;
    }
    
    // Only process clicks on the main series (not volume)
    if (!params || !params.value) return;
    
    const point: DrawingPoint = {
      timestamp: params.value[0],
      price: params.value[1] || params.value[4], // Use close price for candlestick
    };
    
    console.log('📍 Chart click:', { tool: props.activeTool, point });
    
    // Handle two-point drawings (trendline, fibonacci, rectangle, arrow)
    if (['trendline', 'fibonacci', 'rectangle', 'arrow'].includes(props.activeTool)) {
      const points = drawingPoints();
      
      if (points.length === 0) {
        // First click - start drawing
        setDrawingPoints([point]);
        console.log('✏️ Drawing started, waiting for second point...');
      } else {
        // Second click - complete drawing
        const newDrawing: Partial<Drawing> = {
          type: props.activeTool as any,
          points: [points[0], point] as any,
          symbol: props.symbol,
          timeframe: props.timeframe,
          style: {
            color: '#3b82f6',
            lineWidth: 2,
            lineType: 'solid',
          },
          visible: true,
          locked: false,
        };
        
        console.log('✅ Drawing complete:', newDrawing);
        props.onDrawingComplete?.(newDrawing);
        setDrawingPoints([]);
      }
    }
    
    // Handle single-point drawings (horizontal line, text)
    if (['horizontal_line', 'text'].includes(props.activeTool)) {
      const newDrawing: Partial<Drawing> = {
        type: props.activeTool as any,
        point: point as any,
        price: point.price,
        symbol: props.symbol,
        timeframe: props.timeframe,
        style: {
          color: '#10b981',
          lineWidth: 1,
          lineType: 'dashed',
        },
        visible: true,
        locked: false,
      };
      
      console.log('✅ Single-point drawing complete:', newDrawing);
      props.onDrawingComplete?.(newDrawing);
    }
  };

  /**
   * Generate ECharts series for drawings
   */
  const generateDrawingSeries = (): any[] => {
    const drawings = props.drawings || [];
    if (drawings.length === 0) return [];
    
    console.log('🎨 Rendering', drawings.length, 'drawings');
    
    return drawings.map(drawing => {
      const isSelected = drawing.id === props.selectedDrawingId;
      
      if (drawing.type === 'trendline' && 'points' in drawing) {
        return {
          type: 'line',
          name: `Trendline-${drawing.id}`,
          id: drawing.id, // Add ID for click detection
          data: [
            [drawing.points[0].timestamp, drawing.points[0].price],
            [drawing.points[1].timestamp, drawing.points[1].price],
          ],
          lineStyle: {
            color: isSelected ? '#fbbf24' : (drawing.style?.color || '#3b82f6'), // Yellow when selected
            width: isSelected ? 4 : (drawing.style?.lineWidth || 2), // Thicker when selected
            type: drawing.style?.lineType || 'solid',
          },
          symbol: 'circle',
          symbolSize: isSelected ? 10 : 6, // Larger handles when selected
          itemStyle: {
            color: isSelected ? '#fbbf24' : (drawing.style?.color || '#3b82f6'),
            borderWidth: isSelected ? 2 : 1,
            borderColor: '#1f2937',
          },
          z: isSelected ? 20 : 15, // Selected drawings on top
          silent: false, // Allow interaction
          emphasis: {
            disabled: false,
            lineStyle: {
              width: 4,
            },
          },
        };
      }
      
      if (drawing.type === 'horizontal_line' && 'price' in drawing) {
        return {
          type: 'line',
          name: `HLine-${drawing.id}`,
          id: drawing.id,
          markLine: {
            silent: false,
            symbol: 'none',
            data: [
              {
                yAxis: drawing.price,
                lineStyle: {
                  color: isSelected ? '#fbbf24' : (drawing.style?.color || '#10b981'),
                  width: isSelected ? 3 : (drawing.style?.lineWidth || 1),
                  type: drawing.style?.lineType || 'dashed',
                },
                label: {
                  show: isSelected,
                  formatter: () => `${drawing.price.toFixed(2)}`,
                  color: '#fbbf24',
                },
              },
            ],
          },
          z: isSelected ? 20 : 15,
        };
      }
      
      return null;
    }).filter(Boolean);
  };

  // Initialize ECharts
  const chart = useECharts(() => ({
    container: chartContainer,
    options: generateChartOptions(),
    loading: loading(),
    autoResize: true,
    theme: 'dark',
  }));

  // Wire up click event for drawing tools AND selection
  createEffect(() => {
    const instance = chart.getInstance();
    if (instance) {
      if (props.activeTool) {
        console.log('🔧 Wiring up click handler for tool:', props.activeTool);
      }
      instance.on('click', handleChartClick);
      
      onCleanup(() => {
        instance?.off('click', handleChartClick);
      });
    }
  });

  // ============================================================================
  // EFFECTS
  // ============================================================================

  // Fetch data when symbol or timeframe changes
  createEffect(
    on([() => props.symbol, () => props.timeframe], () => {
      fetchChartData();
    })
  );

  // Update chart when bars change
  createEffect(
    on(bars, () => {
      if (bars().length > 0 && chart.getInstance()) {
        chart.updateChart(generateChartOptions(), false);
      }
    })
  );

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div class="relative w-full" style={{ height: props.height || '600px' }}>
      {/* Chart Container */}
      <div ref={chartContainer} class="w-full h-full" />

      {/* Error State */}
      <Show when={error()}>
        <div class="absolute inset-0 flex items-center justify-center bg-terminal-950/90 backdrop-blur-sm">
          <div class="text-center p-6 bg-terminal-900 border border-red-500/50 rounded-lg max-w-md">
            <div class="text-red-500 text-lg font-semibold mb-2">
              ⚠️ Chart Error
            </div>
            <div class="text-gray-400 text-sm mb-4">{error()}</div>
            <button
              onClick={() => fetchChartData()}
              class="px-4 py-2 bg-accent-600 hover:bg-accent-700 text-white rounded transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </Show>

      {/* Latest Price Info Overlay */}
      <Show when={latestInfo()}>
        <div class="absolute top-4 left-4 bg-terminal-900/90 backdrop-blur-sm border border-terminal-750 rounded-lg p-3 text-xs">
          <div class="flex items-center gap-3">
            <div>
              <div class="text-gray-500 mb-1">Price</div>
              <div class="text-white font-bold text-lg">
                ${formatPrice(latestInfo()!.price)}
              </div>
            </div>
            <div>
              <div class="text-gray-500 mb-1">Change</div>
              <div
                class="font-semibold"
                classList={{
                  'text-green-500': latestInfo()!.direction === 'up',
                  'text-red-500': latestInfo()!.direction === 'down',
                  'text-gray-400': latestInfo()!.direction === 'neutral',
                }}
              >
                {latestInfo()!.change >= 0 ? '+' : ''}${formatPrice(latestInfo()!.change)}
                ({(latestInfo()!.changePercent ?? 0).toFixed(2)}%)
              </div>
            </div>
            <div>
              <div class="text-gray-500 mb-1">Volume</div>
              <div class="text-white font-semibold">
                {formatVolume(latestInfo()!.volume)}
              </div>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
