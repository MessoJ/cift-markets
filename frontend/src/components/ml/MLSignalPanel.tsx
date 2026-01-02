/**
 * CIFT Markets - ML Signal Panel
 * 
 * Real-time ML predictions panel for Trading page.
 * Shows:
 * - Current signal (Buy/Sell/Hold)
 * - Confidence level with visual gauge
 * - Price targets and stop loss
 * - AI-powered reasoning
 * - Historical signal accuracy
 * 
 * Integrates with:
 * - Order Flow Transformer model
 * - Technical analysis
 * - Gemini AI for explanations
 */

import { createSignal, createEffect, Show, onCleanup } from 'solid-js';
import { 
  TrendingUp, TrendingDown, Activity, Target, AlertTriangle,
  RefreshCw, Zap, Brain, Shield, Clock, ChevronDown, ChevronUp,
  BarChart3, ArrowUpRight, ArrowDownRight, Minus, Info
} from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { apiClient } from '~/lib/api/client';
import { formatCurrency, formatPercent } from '~/lib/utils/format';

interface MLSignal {
  symbol: string;
  signal_type: string;
  confidence: number;
  entry_price: number;
  target_price: number | null;
  stop_loss: number | null;
  risk_reward_ratio: number;
  expected_return_pct: number;
  max_loss_pct: number;
  hold_duration: string;
  urgency: string;
  reasons: string[];
  ai_explanation: string;
  timestamp: string;
}

interface MLSignalPanelProps {
  symbol: string;
  currentPrice?: number;
  className?: string;
  collapsed?: boolean;
  onSignalChange?: (signal: MLSignal | null) => void;
}

export function MLSignalPanel(props: MLSignalPanelProps) {
  const [signal, setSignal] = createSignal<MLSignal | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [expanded, setExpanded] = createSignal(!props.collapsed);
  const [refreshing, setRefreshing] = createSignal(false);

  // Auto-refresh every 5 minutes
  let refreshInterval: number | undefined;

  createEffect(() => {
    const sym = props.symbol;
    if (sym) {
      fetchSignal(sym);
      
      // Set up auto-refresh
      refreshInterval = setInterval(() => {
        if (sym) fetchSignal(sym, true);
      }, 300000) as unknown as number; // 5 minutes
    }
    
    onCleanup(() => {
      if (refreshInterval) clearInterval(refreshInterval);
    });
  });

  const fetchSignal = async (symbol: string, silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);
    
    setError(null);
    
    try {
      const data = await apiClient.get(`/ml-signals/signal/${symbol}`);
      setSignal(data);
      props.onSignalChange?.(data);
    } catch (err: any) {
      console.error('ML Signal fetch failed:', err);
      if (!silent) setError(err.message || 'Failed to load signal');
      setSignal(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    if (props.symbol) {
      fetchSignal(props.symbol);
    }
  };

  const getSignalConfig = (signalType: string) => {
    switch (signalType) {
      case 'strong_buy':
        return {
          label: 'STRONG BUY',
          color: 'text-success-400',
          bg: 'bg-success-500/20',
          border: 'border-success-500/40',
          icon: TrendingUp,
          gradient: 'from-success-500 to-emerald-400',
        };
      case 'buy':
        return {
          label: 'BUY',
          color: 'text-success-400',
          bg: 'bg-success-500/10',
          border: 'border-success-500/30',
          icon: TrendingUp,
          gradient: 'from-success-500/80 to-emerald-400/80',
        };
      case 'sell':
        return {
          label: 'SELL',
          color: 'text-danger-400',
          bg: 'bg-danger-500/10',
          border: 'border-danger-500/30',
          icon: TrendingDown,
          gradient: 'from-danger-500/80 to-red-400/80',
        };
      case 'strong_sell':
        return {
          label: 'STRONG SELL',
          color: 'text-danger-400',
          bg: 'bg-danger-500/20',
          border: 'border-danger-500/40',
          icon: TrendingDown,
          gradient: 'from-danger-500 to-red-400',
        };
      default:
        return {
          label: 'HOLD',
          color: 'text-warning-400',
          bg: 'bg-warning-500/10',
          border: 'border-warning-500/30',
          icon: Minus,
          gradient: 'from-warning-500/80 to-amber-400/80',
        };
    }
  };

  const getUrgencyBadge = (urgency: string) => {
    switch (urgency) {
      case 'critical':
        return { text: 'CRITICAL', color: 'bg-danger-500 text-white animate-pulse' };
      case 'high':
        return { text: 'HIGH', color: 'bg-accent-500 text-white' };
      case 'normal':
        return { text: 'NORMAL', color: 'bg-terminal-700 text-gray-300' };
      default:
        return { text: 'LOW', color: 'bg-terminal-800 text-gray-400' };
    }
  };

  return (
    <div class={`bg-terminal-900 border border-terminal-700 rounded-lg overflow-hidden ${props.className || ''}`}>
      {/* Header */}
      <div 
        class="flex items-center justify-between px-4 py-3 bg-terminal-800/50 cursor-pointer hover:bg-terminal-800 transition-colors"
        onClick={() => setExpanded(!expanded())}
      >
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-full bg-gradient-to-br from-accent-500 to-primary-500 flex items-center justify-center">
            <Brain class="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 class="text-sm font-semibold text-white flex items-center gap-2">
              ML Signal
              <Show when={refreshing()}>
                <RefreshCw class="w-3 h-3 text-accent-400 animate-spin" />
              </Show>
            </h3>
            <p class="text-xs text-gray-500">AI-Powered Analysis</p>
          </div>
        </div>
        
        <div class="flex items-center gap-2">
          <Show when={signal()}>
            {(() => {
              const config = getSignalConfig(signal()!.signal_type);
              return (
                <span class={`px-2 py-0.5 rounded text-xs font-bold ${config.bg} ${config.color} ${config.border} border`}>
                  {config.label}
                </span>
              );
            })()}
          </Show>
          
          <button
            onClick={(e) => { e.stopPropagation(); handleRefresh(); }}
            class="p-1 hover:bg-terminal-700 rounded transition-colors"
            disabled={loading()}
          >
            <RefreshCw class={`w-4 h-4 text-gray-400 ${loading() ? 'animate-spin' : ''}`} />
          </button>
          
          {expanded() ? (
            <ChevronUp class="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown class="w-4 h-4 text-gray-400" />
          )}
        </div>
      </div>

      {/* Expanded Content */}
      <Show when={expanded()}>
        <div class="p-4 space-y-4">
          {/* Loading State */}
          <Show when={loading() && !signal()}>
            <div class="flex items-center justify-center py-8">
              <div class="flex items-center gap-3">
                <div class="w-6 h-6 rounded-full border-2 border-accent-500 border-t-transparent animate-spin" />
                <span class="text-sm text-gray-400">Analyzing {props.symbol}...</span>
              </div>
            </div>
          </Show>

          {/* Error State */}
          <Show when={error() && !loading()}>
            <div class="text-center py-6">
              <AlertTriangle class="w-8 h-8 text-danger-400 mx-auto mb-2" />
              <p class="text-sm text-gray-400">{error()}</p>
              <button
                onClick={handleRefresh}
                class="mt-3 px-4 py-2 text-xs bg-terminal-700 hover:bg-terminal-600 text-white rounded transition-colors"
              >
                Retry
              </button>
            </div>
          </Show>

          {/* Signal Content */}
          <Show when={signal() && !loading()}>
            {(() => {
              const sig = signal()!;
              const config = getSignalConfig(sig.signal_type);
              const SignalIcon = config.icon;
              const urgency = getUrgencyBadge(sig.urgency);
              
              return (
                <>
                  {/* Main Signal Display */}
                  <div class={`p-4 rounded-lg bg-gradient-to-r ${config.bg} border ${config.border}`}>
                    <div class="flex items-center justify-between mb-4">
                      <div class="flex items-center gap-3">
                        <div class={`w-12 h-12 rounded-full bg-gradient-to-br ${config.gradient} flex items-center justify-center`}>
                          <SignalIcon class="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <div class={`text-2xl font-bold ${config.color}`}>
                            {config.label}
                          </div>
                          <div class="text-xs text-gray-400">
                            @ {formatCurrency(sig.entry_price)}
                          </div>
                        </div>
                      </div>
                      
                      <div class="text-right">
                        <span class={`px-2 py-0.5 rounded text-[10px] font-bold ${urgency.color}`}>
                          {urgency.text}
                        </span>
                      </div>
                    </div>

                    {/* Confidence Gauge */}
                    <div class="mb-4">
                      <div class="flex items-center justify-between mb-1">
                        <span class="text-xs text-gray-400">Confidence</span>
                        <span class={`text-lg font-bold font-mono ${config.color}`}>
                          {(sig.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div class="h-2 bg-terminal-800 rounded-full overflow-hidden">
                        <div 
                          class={`h-full rounded-full bg-gradient-to-r ${config.gradient} transition-all duration-500`}
                          style={{ width: `${sig.confidence * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Targets Grid */}
                    <div class="grid grid-cols-3 gap-3">
                      <Show when={sig.target_price}>
                        <div class="bg-terminal-900/50 rounded p-2">
                          <div class="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                            <Target class="w-3 h-3" /> Target
                          </div>
                          <div class="text-sm font-mono text-success-400">
                            {formatCurrency(sig.target_price!)}
                          </div>
                          <div class="text-[10px] text-success-400">
                            +{sig.expected_return_pct.toFixed(1)}%
                          </div>
                        </div>
                      </Show>
                      
                      <Show when={sig.stop_loss}>
                        <div class="bg-terminal-900/50 rounded p-2">
                          <div class="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                            <Shield class="w-3 h-3" /> Stop
                          </div>
                          <div class="text-sm font-mono text-danger-400">
                            {formatCurrency(sig.stop_loss!)}
                          </div>
                          <div class="text-[10px] text-danger-400">
                            -{sig.max_loss_pct.toFixed(1)}%
                          </div>
                        </div>
                      </Show>
                      
                      <div class="bg-terminal-900/50 rounded p-2">
                        <div class="text-[10px] text-gray-500 uppercase flex items-center gap-1">
                          <BarChart3 class="w-3 h-3" /> R:R
                        </div>
                        <div class={`text-sm font-mono ${sig.risk_reward_ratio >= 2 ? 'text-success-400' : sig.risk_reward_ratio >= 1 ? 'text-warning-400' : 'text-danger-400'}`}>
                          {sig.risk_reward_ratio.toFixed(1)}:1
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Reasons */}
                  <Show when={sig.reasons.length > 0}>
                    <div class="space-y-2">
                      <h4 class="text-xs text-gray-500 uppercase font-semibold">Key Factors</h4>
                      <ul class="space-y-1">
                        {sig.reasons.map((reason, i) => (
                          <li class="flex items-start gap-2 text-xs text-gray-300">
                            <Zap class={`w-3 h-3 mt-0.5 flex-shrink-0 ${
                              sig.signal_type.includes('buy') ? 'text-success-400' : 
                              sig.signal_type.includes('sell') ? 'text-danger-400' : 'text-warning-400'
                            }`} />
                            {reason}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </Show>

                  {/* AI Explanation */}
                  <Show when={sig.ai_explanation}>
                    <div class="bg-gradient-to-r from-primary-500/5 to-accent-500/5 border border-primary-500/20 rounded-lg p-3">
                      <div class="flex items-center gap-2 mb-2">
                        <AIIcon size={14} />
                        <span class="text-xs text-primary-400 font-semibold">AI Analysis</span>
                      </div>
                      <p class="text-xs text-gray-300 leading-relaxed">
                        {sig.ai_explanation}
                      </p>
                    </div>
                  </Show>

                  {/* Hold Duration & Timestamp */}
                  <div class="flex items-center justify-between text-xs text-gray-500">
                    <div class="flex items-center gap-1">
                      <Clock class="w-3 h-3" />
                      <span>Hold: {sig.hold_duration}</span>
                    </div>
                    <span>
                      Updated: {new Date(sig.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </>
              );
            })()}
          </Show>
        </div>
      </Show>
    </div>
  );
}

export default MLSignalPanel;
