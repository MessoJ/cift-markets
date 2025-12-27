/**
 * CIFT Markets - Prediction Accuracy Panel
 * 
 * Detailed comparison panel showing predicted vs actual results.
 * Visualizes accuracy metrics and bar-by-bar comparison.
 */

import { Show, For, createMemo } from 'solid-js';
import { 
  TrendingUp, TrendingDown, Target, BarChart3, 
  CheckCircle, XCircle, ArrowUpRight, ArrowDownRight 
} from 'lucide-solid';
import { predictionStore } from '~/stores/prediction.store';
import type { PredictedBar, ActualBar, PredictionAccuracy } from '~/types/prediction.types';

export interface PredictionAccuracyPanelProps {
  symbol: string;
  timeframe: string;
  onClose?: () => void;
}

export default function PredictionAccuracyPanel(props: PredictionAccuracyPanelProps) {
  const session = () => predictionStore.getActiveSession(props.symbol, props.timeframe);
  
  // Prepare bar comparison data
  const comparisons = createMemo(() => {
    const s = session();
    if (!s) return [];
    
    return s.predictedBars.map((pred, i) => {
      const actual = s.actualBars[i];
      if (!actual) return { predicted: pred, actual: null, match: null };
      
      const predDirection = pred.close >= pred.open ? 'up' : 'down';
      const actDirection = actual.close >= actual.open ? 'up' : 'down';
      const directionMatch = predDirection === actDirection;
      
      const priceError = Math.abs(pred.close - actual.close);
      const priceErrorPct = (priceError / actual.close) * 100;
      
      const withinRange = actual.high <= pred.high && actual.low >= pred.low;
      
      return {
        predicted: pred,
        actual,
        match: {
          direction: directionMatch,
          priceError,
          priceErrorPct,
          withinRange,
        },
      };
    });
  });

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatPercent = (pct: number) => `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
  
  const getScoreGradient = (score: number): string => {
    if (score >= 70) return 'from-green-500 to-emerald-500';
    if (score >= 50) return 'from-yellow-500 to-amber-500';
    return 'from-red-500 to-rose-500';
  };

  return (
    <div class="bg-terminal-900 border border-terminal-750 rounded-lg shadow-xl overflow-hidden flex flex-col max-h-[70vh]">
      {/* Header */}
      <div class="flex items-center justify-between p-4 border-b border-terminal-750 bg-terminal-850 flex-shrink-0">
        <div class="flex items-center gap-3">
          <div class="p-2 bg-accent-500/20 rounded-lg">
            <Target size={20} class="text-accent-400" />
          </div>
          <div>
            <h3 class="text-white font-semibold">Prediction Accuracy</h3>
            <p class="text-gray-500 text-sm">
              {props.symbol} · {props.timeframe} · {session()?.predictedBars.length || 0} bars
            </p>
          </div>
        </div>
        <Show when={props.onClose}>
          <button
            onClick={props.onClose}
            class="p-2 hover:bg-terminal-800 rounded-lg transition-colors"
          >
            <XCircle size={20} class="text-gray-500" />
          </button>
        </Show>
      </div>

      {/* Overall Score */}
      <Show when={session()?.accuracy}>
        {(() => {
          const acc = session()!.accuracy!;
          return (
            <div class="p-4 border-b border-terminal-750">
              <div class="flex items-center gap-4">
                {/* Main Score */}
                <div class={`flex-shrink-0 w-24 h-24 rounded-full bg-gradient-to-br ${getScoreGradient(acc.overallScore)} p-1`}>
                  <div class="w-full h-full rounded-full bg-terminal-900 flex items-center justify-center">
                    <div class="text-center">
                      <div class="text-2xl font-bold text-white">{acc.overallScore.toFixed(0)}</div>
                      <div class="text-xs text-gray-500">Score</div>
                    </div>
                  </div>
                </div>

                {/* Detailed Metrics */}
                <div class="flex-1 grid grid-cols-2 gap-3">
                  <div class="bg-terminal-850 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <TrendingUp size={14} class="text-green-400" />
                      <span class="text-gray-500 text-xs">Direction</span>
                    </div>
                    <div class="text-white font-semibold">{acc.directional.toFixed(1)}%</div>
                  </div>
                  
                  <div class="bg-terminal-850 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <Target size={14} class="text-blue-400" />
                      <span class="text-gray-500 text-xs">Range</span>
                    </div>
                    <div class="text-white font-semibold">{acc.highLowAccuracy.toFixed(1)}%</div>
                  </div>
                  
                  <div class="bg-terminal-850 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <BarChart3 size={14} class="text-accent-400" />
                      <span class="text-gray-500 text-xs">Volume</span>
                    </div>
                    <div class="text-white font-semibold">{acc.volumeAccuracy.toFixed(1)}%</div>
                  </div>
                  
                  <div class="bg-terminal-850 rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-1">
                      <span class="text-gray-500 text-xs">Price RMSE</span>
                    </div>
                    <div class="text-white font-semibold">${acc.priceRMSE.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}
      </Show>

      {/* Bar-by-Bar Comparison */}
      <div class="p-4 flex-1 overflow-y-auto min-h-0">
        <h4 class="text-gray-400 text-sm font-medium mb-3">Bar-by-Bar Comparison</h4>
        <div class="space-y-2">
          <For each={comparisons()}>
            {(comp, index) => (
              <div class="flex items-center gap-3 p-2 bg-terminal-850 rounded-lg">
                <div class="text-gray-600 text-xs w-6">#{index() + 1}</div>
                
                {/* Predicted */}
                <div class="flex-1">
                  <div class="text-gray-500 text-xs mb-0.5">Predicted</div>
                  <div class="flex items-center gap-2">
                    <Show
                      when={comp.predicted.close >= comp.predicted.open}
                      fallback={<ArrowDownRight size={12} class="text-red-400" />}
                    >
                      <ArrowUpRight size={12} class="text-green-400" />
                    </Show>
                    <span class="text-white text-sm font-mono">
                      {formatPrice(comp.predicted.close)}
                    </span>
                    <span class="text-gray-600 text-xs">
                      ({(comp.predicted.confidence * 100).toFixed(0)}% conf)
                    </span>
                  </div>
                </div>

                {/* Actual */}
                <div class="flex-1">
                  <div class="text-gray-500 text-xs mb-0.5">Actual</div>
                  <Show
                    when={comp.actual}
                    fallback={<span class="text-gray-600 text-sm">Pending...</span>}
                  >
                    <div class="flex items-center gap-2">
                      <Show
                        when={comp.actual!.close >= comp.actual!.open}
                        fallback={<ArrowDownRight size={12} class="text-red-400" />}
                      >
                        <ArrowUpRight size={12} class="text-green-400" />
                      </Show>
                      <span class="text-white text-sm font-mono">
                        {formatPrice(comp.actual!.close)}
                      </span>
                    </div>
                  </Show>
                </div>

                {/* Match Status */}
                <Show when={comp.match}>
                  <div class="flex items-center gap-2">
                    <Show
                      when={comp.match!.direction}
                      fallback={<XCircle size={16} class="text-red-400" />}
                    >
                      <CheckCircle size={16} class="text-green-400" />
                    </Show>
                    <span class="text-gray-500 text-xs">
                      {formatPercent(-comp.match!.priceErrorPct)}
                    </span>
                  </div>
                </Show>
              </div>
            )}
          </For>
        </div>
      </div>

      {/* Footer */}
      <div class="p-3 border-t border-terminal-750 bg-terminal-850/50 flex-shrink-0">
        <p class="text-gray-600 text-xs text-center">
          Model: {session()?.modelId || 'N/A'} v{session()?.modelVersion || '0.0'}
        </p>
      </div>
    </div>
  );
}
