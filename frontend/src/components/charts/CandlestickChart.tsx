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
import type { PredictedBar } from '~/types/prediction.types';
import { DEFAULT_PREDICTION_STYLE } from '~/types/prediction.types';

export interface CandlestickChartProps {
  symbol: string;
  timeframe: string;
  candleLimit?: number;
  showVolume?: boolean;
  showVolumeProfile?: boolean;
  chartType?: 'candlestick' | 'line' | 'area' | 'heikin_ashi';
  enableRealTime?: boolean;
  activeIndicators?: IndicatorConfig[];
  activeTool?: DrawingType | null;
  drawings?: Drawing[];
  selectedDrawingId?: string | null;
  // Prediction props
  predictedBars?: PredictedBar[];
  predictionStartIndex?: number;
  showPrediction?: boolean;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  onIndicatorsChange?: (indicators: IndicatorConfig[]) => void;
  onDrawingComplete?: (drawing: Partial<Drawing>) => void;
  onDrawingSelect?: (drawingId: string | null) => void;
  onBarsLoaded?: (bars: OHLCVBar[]) => void;
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

  // Volume Profile State
  const [volumeProfile, setVolumeProfile] = createSignal<{price_levels: number[], volumes: number[], max_volume: number} | null>(null);
  
  const showVolumeProfile = () => props.activeIndicators?.some(i => i.id === 'volume_profile' && i.enabled) ?? props.showVolumeProfile ?? false;

  // Fetch volume profile when enabled
  createEffect(() => {
    if (showVolumeProfile() && props.symbol) {
      const fetchData = async () => {
        try {
          const end = new Date();
          const start = new Date();
          start.setDate(start.getDate() - 200); // Default 200 days
          
          const data = await apiClient.get(`/market-data/profile/${props.symbol}`, {
            params: {
              start_date: start.toISOString(),
              end_date: end.toISOString(),
              bins: 100 // High resolution
            }
          });
          setVolumeProfile(data);
        } catch (e) {
          console.error("Failed to load volume profile", e);
        }
      };
      fetchData();
    } else {
      setVolumeProfile(null);
    }
  });

  // Drawing tools state
  const [drawingPoints, setDrawingPoints] = createSignal<DrawingPoint[]>([]);
  const [tempDrawing, setTempDrawing] = createSignal<Partial<Drawing> | null>(null);

  // Replay Mode State
  const [isReplayMode, setIsReplayMode] = createSignal(false);
  const [replayIndex, setReplayIndex] = createSignal(0);
  const [isReplayPlaying, setIsReplayPlaying] = createSignal(false);
  const [replaySpeed, setReplaySpeed] = createSignal(500); // ms per candle

  // Replay Loop
  createEffect(() => {
    if (isReplayPlaying() && isReplayMode()) {
      const timer = setInterval(() => {
        setReplayIndex(prev => {
          if (prev >= bars().length - 1) {
            setIsReplayPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, replaySpeed());
      onCleanup(() => clearInterval(timer));
    }
  });

  const visibleBars = () => {
    if (isReplayMode()) {
      return bars().slice(0, replayIndex() + 1);
    }
    return bars();
  };

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
      
      // Check if patterns or advanced indicators need indicators from backend
      const needsIndicators = activeIndicators().some(
        i => i.enabled && ['patterns', 'volume_profile', 'ichimoku', 'pivot_points', 'stoch', 'obv', 'atr_14'].includes(i.id)
      );
      
      const data = await apiClient.getBars(
        props.symbol,
        props.timeframe,
        props.candleLimit || 500,
        needsIndicators
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
      
      // Notify parent of loaded bars (for prediction feature)
      props.onBarsLoaded?.(validBars);
      
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
    const currentBars = visibleBars();
    
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
    let categoryLabels = displayBars.map((bar) => {
      const date = new Date(bar.timestamp);
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      });
    });

    // Add predicted bar labels to category axis
    if (props.showPrediction && props.predictedBars && props.predictedBars.length > 0) {
      const predictionLabels = props.predictedBars.map((bar) => {
        const date = new Date(bar.timestamp);
        return date.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        });
      });
      categoryLabels = [...categoryLabels, ...predictionLabels];
    }

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
          
          // Handle Line Chart (data is number)
          if (typeof candleParam.data === 'number') {
            const close = candleParam.data;
            const dateLabel = candleParam.axisValue || candleParam.name;
            return `
              <div style="padding: 8px;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">
                  ${dateLabel}
                </div>
                <div style="font-size: 12px;">
                  <span style="color: #999;">Price:</span>
                  <span style="font-weight: 600;">${formatPrice(close)}</span>
                </div>
              </div>
            `;
          }

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
        // Volume Profile Series - rendered as markArea bars on the right side
        ...(showVolumeProfile() && volumeProfile()
          ? [
              {
                name: 'Volume Profile',
                type: 'line',
                data: [],
                markArea: {
                  silent: true,
                  data: (() => {
                    const profile = volumeProfile();
                    if (!profile || profile.volumes.length === 0) return [];
                    
                    const maxVol = profile.max_volume || Math.max(...profile.volumes);
                    const areas: any[] = [];
                    const chartWidth = 0.15; // 15% of chart width for volume profile
                    
                    profile.price_levels.forEach((price, i) => {
                      const vol = profile.volumes[i];
                      if (vol <= 0) return;
                      
                      const nextPrice = profile.price_levels[i + 1] || 
                        (price + (profile.price_levels[1] - profile.price_levels[0]));
                      
                      // Volume bar width normalized (0-1 scale) mapped to x-position
                      const volRatio = vol / maxVol;
                      
                      // Point of Control (highest volume) gets special color
                      const isPOC = vol === maxVol;
                      
                      areas.push([
                        {
                          yAxis: price,
                          x: '85%', // Start from right
                          itemStyle: {
                            color: isPOC ? 'rgba(251, 191, 36, 0.4)' : 'rgba(59, 130, 246, 0.25)',
                            borderColor: isPOC ? '#fbbf24' : '#3b82f6',
                            borderWidth: isPOC ? 2 : 0.5,
                          }
                        },
                        {
                          yAxis: nextPrice,
                          x: `${85 + volRatio * 14}%`, // Max 99%
                        }
                      ]);
                    });
                    return areas;
                  })()
                }
              }
            ]
          : []),
        // Pattern Recognition Series
        ...(activeIndicators().some(i => i.id === 'patterns' && i.enabled)
          ? [
              {
                name: 'Patterns',
                type: 'scatter',
                symbol: 'pin', // Use pin or circle
                symbolSize: 15,
                data: bars().map((bar, index) => {
                  // Prioritize patterns
                  if (bar.pattern_bullish_engulfing) return [index, bar.low * 0.999, 'Bull Engulf', 1];
                  if (bar.pattern_bearish_engulfing) return [index, bar.high * 1.001, 'Bear Engulf', -1];
                  if (bar.pattern_hammer) return [index, bar.low * 0.999, 'Hammer', 1];
                  if (bar.pattern_shooting_star) return [index, bar.high * 1.001, 'Shoot Star', -1];
                  if (bar.pattern_doji) return [index, bar.high * 1.001, 'Doji', 0];
                  return null;
                }).filter(Boolean),
                itemStyle: {
                  color: (params: any) => {
                    const type = params.data[3];
                    return type === 1 ? '#10b981' : (type === -1 ? '#ef4444' : '#fbbf24');
                  }
                },
                label: {
                  show: true,
                  formatter: (params: any) => params.data[2],
                  position: 'top',
                  color: '#fff',
                  fontSize: 9,
                  distance: 5
                },
                z: 100
              }
            ]
          : []),
        // Drawing tools (trendlines, shapes, annotations)
        ...generateDrawingSeries(),
        // Prediction series (predicted candles + start line)
        ...generatePredictionSeries(),
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
    
    // Handle two-point drawings (trendline, fibonacci, rectangle, arrow, ruler)
    if (['trendline', 'fibonacci', 'rectangle', 'arrow', 'ruler'].includes(props.activeTool)) {
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

      // Fibonacci Retracement
      if (drawing.type === 'fibonacci' && 'points' in drawing) {
        const p1 = drawing.points[0];
        const p2 = drawing.points[1];
        const diff = p2.price - p1.price;
        const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
        
        const markLineData = levels.map(level => {
          const price = p1.price + diff * level;
          return {
            yAxis: price,
            lineStyle: {
              color: isSelected ? '#fbbf24' : (drawing.style?.color || '#f59e0b'),
              width: 1,
              type: 'solid',
              opacity: 0.7,
            },
            label: {
              show: true,
              position: 'end',
              formatter: `${level} (${price.toFixed(2)})`,
              color: isSelected ? '#fbbf24' : (drawing.style?.color || '#f59e0b'),
              fontSize: 10,
            }
          };
        });

        // Add diagonal trendline connecting start and end
        return {
          type: 'line',
          name: `Fib-${drawing.id}`,
          id: drawing.id,
          data: [
            [p1.timestamp, p1.price],
            [p2.timestamp, p2.price]
          ],
          lineStyle: {
            color: isSelected ? '#fbbf24' : (drawing.style?.color || '#f59e0b'),
            width: 1,
            type: 'dashed',
            opacity: 0.5,
          },
          markLine: {
            silent: true,
            symbol: 'none',
            data: markLineData,
            animation: false,
          },
          z: isSelected ? 20 : 15,
        };
      }

      // Text Annotation
      if (drawing.type === 'text' && 'point' in drawing) {
        // For text, we use a scatter point with a label
        // In a real implementation, we'd need a text input dialog
        // For now, we'll use a placeholder or the drawing ID
        const text = (drawing as any).text || 'Annotation';
        
        return {
          type: 'scatter',
          name: `Text-${drawing.id}`,
          id: drawing.id,
          data: [[drawing.point.timestamp, drawing.point.price]],
          symbolSize: 1, // Invisible point
          label: {
            show: true,
            formatter: text,
            color: isSelected ? '#fbbf24' : (drawing.style?.color || '#e2e8f0'),
            fontSize: 12,
            fontWeight: 'bold',
            backgroundColor: 'rgba(0,0,0,0.5)',
            padding: [4, 8],
            borderRadius: 4,
            position: 'top',
          },
          z: isSelected ? 20 : 15,
        };
      }

      // Price Ruler Tool
      if (drawing.type === 'ruler' && 'points' in drawing) {
        const p1 = drawing.points[0];
        const p2 = drawing.points[1];
        
        // Calculate measurements
        const priceChange = p2.price - p1.price;
        const priceChangePercent = ((priceChange / p1.price) * 100).toFixed(2);
        const barsCount = Math.abs(p2.timestamp - p1.timestamp);
        const timeDiff = Math.abs(p2.timestamp - p1.timestamp);
        const hours = Math.floor(timeDiff / (1000 * 60 * 60));
        const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
        const timeStr = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
        
        const midPrice = (p1.price + p2.price) / 2;
        const midTime = (p1.timestamp + p2.timestamp) / 2;
        
        return {
          type: 'line',
          name: `Ruler-${drawing.id}`,
          id: drawing.id,
          data: [
            [p1.timestamp, p1.price],
            [p2.timestamp, p2.price]
          ],
          lineStyle: {
            color: isSelected ? '#fbbf24' : (drawing.style?.color || '#22d3ee'),
            width: isSelected ? 3 : (drawing.style?.lineWidth || 2),
            type: 'dashed',
          },
          symbol: ['circle', 'circle'],
          symbolSize: 8,
          // Add measurement label in the middle
          label: {
            show: true,
            position: 'middle',
            formatter: () => {
              const sign = priceChange >= 0 ? '+' : '';
              return `${sign}$${priceChange.toFixed(2)} (${sign}${priceChangePercent}%)\n${timeStr}`;
            },
            color: '#fff',
            fontSize: 11,
            fontWeight: 'bold',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: [6, 10],
            borderRadius: 4,
            borderColor: isSelected ? '#fbbf24' : (drawing.style?.color || '#22d3ee'),
            borderWidth: 1,
          },
          // Also add start/end labels
          markPoint: {
            symbol: 'circle',
            symbolSize: 6,
            data: [
              {
                coord: [p1.timestamp, p1.price],
                itemStyle: { color: isSelected ? '#fbbf24' : '#22d3ee' },
                label: {
                  show: true,
                  formatter: `$${p1.price.toFixed(2)}`,
                  position: 'left',
                  color: '#fff',
                  fontSize: 9,
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  padding: [2, 4],
                  borderRadius: 2,
                }
              },
              {
                coord: [p2.timestamp, p2.price],
                itemStyle: { color: isSelected ? '#fbbf24' : '#22d3ee' },
                label: {
                  show: true,
                  formatter: `$${p2.price.toFixed(2)}`,
                  position: 'right',
                  color: '#fff',
                  fontSize: 9,
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  padding: [2, 4],
                  borderRadius: 2,
                }
              }
            ]
          },
          z: isSelected ? 20 : 15,
        };
      }
      
      return null;
    }).filter(Boolean);
  };

  /**
   * Generate ECharts series for predictions
   * Shows predicted bars with faded appearance and a dotted line at prediction start
   */
  const generatePredictionSeries = (): any[] => {
    if (!props.showPrediction || !props.predictedBars || props.predictedBars.length === 0) {
      return [];
    }

    const currentBars = bars();
    if (currentBars.length === 0) return [];

    const series: any[] = [];
    const style = DEFAULT_PREDICTION_STYLE;

    // Predicted candlestick data - positioned after actual bars
    // We need to add null values for all actual bar indices, then add prediction values
    const predictedCandleData: any[] = [];
    
    // Fill with null for actual bars (so predictions start after real data)
    for (let i = 0; i < currentBars.length; i++) {
      predictedCandleData.push('-'); // ECharts uses '-' for missing data
    }
    
    // Add predicted bars
    props.predictedBars.forEach((bar) => {
      predictedCandleData.push({
        value: [bar.open, bar.close, bar.low, bar.high],
        itemStyle: {
          color: bar.close >= bar.open 
            ? style.bullishPredictColor 
            : style.bearishPredictColor,
          borderColor: bar.close >= bar.open 
            ? style.bullishPredictColor 
            : style.bearishPredictColor,
          opacity: style.predictedBarOpacity,
        },
      });
    });

    // Add predicted candles series
    series.push({
      name: 'Prediction',
      type: 'candlestick',
      data: predictedCandleData,
      itemStyle: {
        color: style.bullishPredictColor,
        color0: style.bearishPredictColor,
        borderColor: style.bullishPredictColor,
        borderColor0: style.bearishPredictColor,
        opacity: style.predictedBarOpacity,
      },
      barMinWidth: 2,
      barMaxWidth: 20,
      z: 5, // Below main candles
    });

    // Add vertical dotted line at prediction start
    const predictionStartIndex = props.predictionStartIndex ?? (currentBars.length - 1);
    
    series.push({
      name: 'Prediction Start',
      type: 'line',
      markLine: {
        silent: true,
        symbol: ['none', 'none'],
        animation: false,
        data: [
          {
            xAxis: predictionStartIndex,
            lineStyle: {
              color: style.predictionLineColor,
              width: style.predictionLineWidth,
              type: 'dashed',
              dashOffset: 5,
            },
            label: {
              show: true,
              position: 'start',
              formatter: '{icon|◉} AI Prediction',
              rich: {
                icon: {
                  color: '#f97316',
                  fontSize: 14,
                  fontWeight: 'bold',
                },
              },
              color: style.predictionLineColor,
              fontSize: 11,
              fontWeight: 'bold',
              backgroundColor: 'rgba(0, 0, 0, 0.85)',
              padding: [4, 10],
              borderRadius: 4,
              borderColor: style.predictionLineColor,
              borderWidth: 1,
            },
          },
        ],
      },
      z: 25, // Above everything
    });

    // Add confidence indicators for high-confidence predictions
    const highConfidenceBars = props.predictedBars
      .map((bar, i) => ({ bar, index: currentBars.length + i }))
      .filter(({ bar }) => bar.confidence >= style.confidenceShowThreshold);

    if (highConfidenceBars.length > 0) {
      series.push({
        name: 'Confidence',
        type: 'scatter',
        data: highConfidenceBars.map(({ bar, index }) => ({
          value: [index, bar.high + (bar.high * 0.002)], // Slightly above high
          symbol: 'diamond',
          symbolSize: 8 * bar.confidence, // Size based on confidence
          itemStyle: {
            color: bar.close >= bar.open ? '#22c55e' : '#ef4444',
            opacity: bar.confidence,
          },
        })),
        z: 10,
      });
    }

    console.log('🔮 Rendering', props.predictedBars.length, 'predicted bars');
    return series;
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

  // Refetch data when indicators requiring backend calculation are toggled
  createEffect(
    on(() => activeIndicators().filter(i => i.enabled && ['patterns', 'volume_profile', 'ichimoku', 'pivot_points', 'stoch', 'obv', 'atr_14'].includes(i.id)).length, 
    () => {
      // Refetch when patterns/advanced indicators count changes
      if (bars().length > 0) {
        fetchChartData();
      }
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

      {/* Replay Controls */}
      <Show when={isReplayMode()}>
        <div class="absolute top-4 right-4 bg-terminal-900/90 backdrop-blur-sm border border-terminal-750 rounded-lg p-3 flex flex-col gap-2 z-50 shadow-lg">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-bold text-white">Replay Mode</span>
            <button 
              class="text-gray-400 hover:text-white"
              onClick={() => {
                setIsReplayMode(false);
                setIsReplayPlaying(false);
              }}
            >
              ✕
            </button>
          </div>
          <div class="flex items-center gap-2">
            <button
              class="p-1.5 rounded bg-terminal-800 hover:bg-terminal-700 text-white"
              onClick={() => setReplayIndex(Math.max(0, replayIndex() - 1))}
              title="Step Back"
            >
              ⏮
            </button>
            <button
              class="p-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white w-8 flex justify-center"
              onClick={() => setIsReplayPlaying(!isReplayPlaying())}
              title={isReplayPlaying() ? 'Pause' : 'Play'}
            >
              {isReplayPlaying() ? '⏸' : '▶'}
            </button>
            <button
              class="p-1.5 rounded bg-terminal-800 hover:bg-terminal-700 text-white"
              onClick={() => setReplayIndex(Math.min(bars().length - 1, replayIndex() + 1))}
              title="Step Forward"
            >
              ⏭
            </button>
          </div>
          <div class="flex items-center gap-2 text-xs text-gray-400">
            <span>Speed:</span>
            <select 
              class="bg-terminal-800 border border-terminal-700 rounded px-1 py-0.5 text-white outline-none"
              value={replaySpeed()}
              onChange={(e) => setReplaySpeed(Number(e.currentTarget.value))}
            >
              <option value="1000">1x</option>
              <option value="500">2x</option>
              <option value="200">5x</option>
              <option value="100">10x</option>
            </select>
          </div>
          <div class="text-xs text-gray-500 text-center mt-1">
            {bars()[replayIndex()] ? new Date(bars()[replayIndex()].timestamp).toLocaleString() : ''}
          </div>
        </div>
      </Show>

      {/* Screenshot Button */}
      <button
        class="absolute top-4 right-28 bg-terminal-900/80 hover:bg-accent-500/20 border border-accent-500/50 rounded-lg px-3 py-1.5 text-xs font-medium text-white z-40 flex items-center gap-2 shadow-lg backdrop-blur-sm transition-colors group"
        onClick={() => {
          const instance = chart.getInstance();
          if (!instance) return;
          const url = instance.getDataURL({
            type: 'png',
            pixelRatio: 2,
            backgroundColor: DARK_THEME.background
          });
          if (url) {
            const link = document.createElement('a');
            link.download = `${props.symbol}_chart.png`;
            link.href = url;
            link.click();
          }
        }}
        title="Take Screenshot"
      >
        <span class="text-accent-500 group-hover:text-accent-400">📷</span>
      </button>

      {/* Replay Toggle Button (when not in replay mode) */}
      <Show when={!isReplayMode() && bars().length > 0}>
        <button
          class="absolute top-4 right-4 bg-terminal-900/80 hover:bg-terminal-800 border border-terminal-750 rounded-lg px-3 py-1.5 text-xs font-medium text-white z-40 flex items-center gap-2 shadow-lg backdrop-blur-sm transition-colors"
          onClick={() => {
            setIsReplayMode(true);
            setReplayIndex(Math.max(0, bars().length - 100)); // Start 100 bars back
          }}
        >
          <span class="text-blue-400">↺</span> Replay
        </button>
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
