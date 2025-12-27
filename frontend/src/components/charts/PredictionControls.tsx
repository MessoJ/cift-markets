/**
 * CIFT Markets - Prediction Controls Component
 * 
 * UI controls for the prediction feature:
 * - Predict button to generate forecasts
 * - Compare button to check accuracy
 * - Clear button to remove predictions
 * - Status indicator showing prediction state
 */

import { createSignal, Show, createEffect, on } from 'solid-js';
import { BarChart3, Trash2, AlertCircle } from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { predictionStore } from '~/stores/prediction.store';
import type { PredictionAccuracy } from '~/types/prediction.types';

export interface PredictionControlsProps {
  symbol: string;
  timeframe: string;
  onPredict: () => void;
  onClear: () => void;
  onCompare: () => void;
  disabled?: boolean;
}

export default function PredictionControls(props: PredictionControlsProps) {
  const [showAccuracy, setShowAccuracy] = createSignal(false);

  // Get current prediction session
  const session = () => predictionStore.getActiveSession(props.symbol, props.timeframe);
  const hasPrediction = () => predictionStore.hasPrediction(props.symbol, props.timeframe);
  const isLoading = () => predictionStore.isLoading();
  const error = () => predictionStore.error();

  // Get accuracy score color
  const getScoreColor = (score: number): string => {
    if (score >= 70) return 'text-green-400';
    if (score >= 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  // Format accuracy display
  const formatAccuracy = (accuracy: PredictionAccuracy | undefined) => {
    if (!accuracy) return null;
    return {
      overall: accuracy.overallScore.toFixed(1),
      directional: accuracy.directional.toFixed(1),
      range: accuracy.highLowAccuracy.toFixed(1),
      bars: accuracy.barsCompared,
    };
  };

  return (
    <div class="flex items-center gap-2">
      {/* Predict Button */}
      <Show when={!hasPrediction()}>
        <button
          onClick={props.onPredict}
          disabled={props.disabled || isLoading()}
          class="flex items-center gap-2 px-3 py-1.5 bg-accent-600 hover:bg-accent-500 disabled:bg-gray-600 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-accent-500/20 hover:shadow-accent-500/30 disabled:shadow-none"
          title="Generate price prediction using ML model"
        >
          <AIIcon size={16} animate={isLoading()} />
          <span class="hidden sm:inline">{isLoading() ? 'Analyzing...' : 'Predict'}</span>
        </button>
      </Show>

      {/* Active Prediction Controls */}
      <Show when={hasPrediction()}>
        {/* Status Badge */}
        <div class="flex items-center gap-2 px-3 py-1.5 bg-accent-500/20 border border-accent-500/30 rounded-lg">
          <div class="w-2 h-2 rounded-full bg-accent-400 animate-pulse" />
          <span class="text-accent-300 text-sm font-medium">
            {session()?.status === 'active' ? 'Predicting' : 'Completed'}
          </span>
          <span class="text-accent-400/60 text-xs">
            {session()?.predictedBars.length} bars
          </span>
        </div>

        {/* Compare Button */}
        <button
          onClick={() => {
            props.onCompare();
            setShowAccuracy(true);
          }}
          class="flex items-center gap-2 px-3 py-1.5 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-300 hover:text-white text-sm rounded-lg transition-colors"
          title="Compare prediction with actual results"
        >
          <BarChart3 size={16} />
          <span class="hidden sm:inline">Compare</span>
        </button>

        {/* Clear Button */}
        <button
          onClick={() => {
            props.onClear();
            setShowAccuracy(false);
          }}
          class="flex items-center gap-2 px-3 py-1.5 bg-terminal-850 hover:bg-red-900/50 border border-terminal-750 hover:border-red-500/50 text-gray-400 hover:text-red-400 text-sm rounded-lg transition-colors"
          title="Clear prediction"
        >
          <Trash2 size={16} />
        </button>
      </Show>

      {/* Accuracy Display */}
      <Show when={showAccuracy() && session()?.accuracy}>
        {(() => {
          const acc = formatAccuracy(session()?.accuracy);
          if (!acc) return null;
          return (
            <div class="flex items-center gap-3 px-3 py-1.5 bg-terminal-850 border border-terminal-750 rounded-lg">
              <div class="flex items-center gap-1">
                <span class="text-gray-500 text-xs">Score:</span>
                <span class={`font-bold text-sm ${getScoreColor(session()?.accuracy?.overallScore || 0)}`}>
                  {acc.overall}%
                </span>
              </div>
              <div class="w-px h-4 bg-terminal-700" />
              <div class="flex items-center gap-1">
                <span class="text-gray-500 text-xs">Dir:</span>
                <span class="text-gray-300 text-xs">{acc.directional}%</span>
              </div>
              <div class="flex items-center gap-1">
                <span class="text-gray-500 text-xs">Range:</span>
                <span class="text-gray-300 text-xs">{acc.range}%</span>
              </div>
              <div class="text-gray-600 text-xs">
                ({acc.bars} bars)
              </div>
            </div>
          );
        })()}
      </Show>

      {/* Error Display */}
      <Show when={error()}>
        <div class="flex items-center gap-2 px-3 py-1.5 bg-red-900/30 border border-red-500/30 rounded-lg">
          <AlertCircle size={14} class="text-red-400" />
          <span class="text-red-300 text-xs">{error()}</span>
        </div>
      </Show>
    </div>
  );
}
