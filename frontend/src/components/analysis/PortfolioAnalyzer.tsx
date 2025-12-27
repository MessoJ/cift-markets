/**
 * PortfolioAnalyzer - Compact Analysis Badge for Portfolio Holdings
 * 
 * Design: A sleek pill badge that pulses when clicked, reveals a slide-down panel
 * with key metrics and a gradient progress bar for score visualization.
 * Different from Trading page - vertical expansion with gradient accent.
 * 
 * Brand Colors: Orange accent (#f97316), terminal blacks, success green, danger red
 */

import { createSignal, Show } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { ChevronDown, ChevronRight, TrendingUp, TrendingDown, Target, AlertTriangle, Activity } from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { apiClient } from '~/lib/api/client';
import { formatCurrency } from '~/lib/utils/format';

interface PortfolioAnalyzerProps {
  symbol: string;
  currentPrice?: number;
  avgCost?: number;
  className?: string;
}

interface AnalysisResult {
  overall_score: number;
  suggested_action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  risk_level: string;
  target_1?: number;
  stop_loss?: number;
  key_insight?: string;
}

export function PortfolioAnalyzer(props: PortfolioAnalyzerProps) {
  const navigate = useNavigate();
  const [isExpanded, setIsExpanded] = createSignal(false);
  const [isAnalyzing, setIsAnalyzing] = createSignal(false);
  const [analysis, setAnalysis] = createSignal<AnalysisResult | null>(null);
  const [error, setError] = createSignal(false);

  const handleAnalyze = async (e: Event) => {
    e.stopPropagation();
    e.preventDefault();
    
    if (isExpanded()) {
      setIsExpanded(false);
      return;
    }

    setIsAnalyzing(true);
    setError(false);
    setIsExpanded(true);

    try {
      const data = await apiClient.get(`/analysis/${props.symbol}`);
      
      // Extract key insight from bullish/bearish factors
      let insight = '';
      if (data.bullish_factors?.length > 0) {
        insight = data.bullish_factors[0];
      } else if (data.bearish_factors?.length > 0) {
        insight = data.bearish_factors[0];
      }
      
      setAnalysis({
        overall_score: data.overall_score || 50,
        suggested_action: data.suggested_action || 'HOLD',
        confidence: data.confidence || 0.5,
        risk_level: data.risk?.risk_level || 'medium',
        target_1: data.target_1,
        stop_loss: data.stop_loss,
        key_insight: insight,
      });
    } catch (err) {
      console.error('Portfolio analysis failed:', err);
      setError(true);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const goToFullAnalysis = (e: Event) => {
    e.stopPropagation();
    e.preventDefault();
    navigate(`/analysis/${props.symbol}`);
  };

  const getActionConfig = (action: string) => {
    switch (action) {
      case 'BUY': return { 
        bg: 'from-success-500/20 to-success-600/10',
        border: 'border-success-500/40',
        text: 'text-success-400',
        icon: TrendingUp
      };
      case 'SELL': return { 
        bg: 'from-danger-500/20 to-danger-600/10',
        border: 'border-danger-500/40',
        text: 'text-danger-400',
        icon: TrendingDown
      };
      default: return { 
        bg: 'from-warning-500/20 to-warning-600/10',
        border: 'border-warning-500/40',
        text: 'text-warning-400',
        icon: Activity
      };
    }
  };

  const getScoreGradient = (score: number) => {
    if (score >= 70) return 'from-success-500 to-success-400';
    if (score >= 50) return 'from-warning-500 to-accent-500';
    return 'from-danger-500 to-danger-400';
  };

  // Calculate potential based on target vs current
  const calculatePotential = () => {
    if (!analysis()?.target_1 || !props.currentPrice) return null;
    const potential = ((analysis()!.target_1! - props.currentPrice) / props.currentPrice) * 100;
    return potential;
  };

  return (
    <div class={`relative ${props.className || ''}`} onClick={(e) => e.stopPropagation()}>
      {/* Trigger Pill */}
      <button
        onClick={handleAnalyze}
        class={`
          inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full
          text-[10px] font-medium uppercase tracking-wide
          transition-all duration-300 ease-out
          ${isExpanded() 
            ? 'bg-accent-500/20 text-accent-400 ring-1 ring-accent-500/30' 
            : 'bg-terminal-800 text-gray-400 hover:bg-accent-500/10 hover:text-accent-400'
          }
        `}
      >
        <AIIcon size={12} animate={isAnalyzing()} />
        <span>AI</span>
        <ChevronDown 
          class={`w-3 h-3 transition-transform duration-200 ${isExpanded() ? 'rotate-180' : ''}`} 
        />
      </button>

      {/* Expanding Panel */}
      <Show when={isExpanded()}>
        <div 
          class={`
            absolute top-full left-0 mt-2 z-50
            min-w-[260px] max-w-[320px]
            bg-gradient-to-br from-terminal-900 to-terminal-950
            border border-terminal-700 rounded-lg shadow-2xl
            overflow-hidden
            animate-in fade-in slide-in-from-top-2 duration-200
          `}
        >
          {/* Loading State */}
          <Show when={isAnalyzing()}>
            <div class="p-4 flex items-center justify-center gap-2">
              <div class="w-4 h-4 rounded-full border-2 border-accent-500 border-t-transparent animate-spin" />
              <span class="text-xs text-gray-400">Analyzing {props.symbol}...</span>
            </div>
          </Show>

          {/* Error State */}
          <Show when={error() && !isAnalyzing()}>
            <div class="p-4 text-center">
              <AlertTriangle class="w-6 h-6 text-danger-400 mx-auto mb-2" />
              <p class="text-xs text-gray-400">Analysis unavailable</p>
            </div>
          </Show>

          {/* Results */}
          <Show when={analysis() && !isAnalyzing() && !error()}>
            {(() => {
              const config = getActionConfig(analysis()!.suggested_action);
              const ActionIcon = config.icon;
              const potential = calculatePotential();
              
              return (
                <>
                  {/* Header with Action */}
                  <div class={`px-4 py-3 bg-gradient-to-r ${config.bg} border-b ${config.border}`}>
                    <div class="flex items-center justify-between">
                      <div class="flex items-center gap-2">
                        <ActionIcon class={`w-4 h-4 ${config.text}`} />
                        <span class={`text-sm font-bold ${config.text}`}>
                          {analysis()!.suggested_action}
                        </span>
                      </div>
                      <div class="text-right">
                        <div class="text-xs text-gray-400">Confidence</div>
                        <div class="text-sm font-mono text-white">
                          {Math.round(analysis()!.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Score Bar */}
                  <div class="px-4 py-3 border-b border-terminal-800">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs text-gray-400">AI Score</span>
                      <span class={`text-lg font-bold font-mono ${
                        analysis()!.overall_score >= 70 ? 'text-success-400' :
                        analysis()!.overall_score >= 50 ? 'text-warning-400' : 'text-danger-400'
                      }`}>
                        {analysis()!.overall_score}
                      </span>
                    </div>
                    <div class="h-2 bg-terminal-800 rounded-full overflow-hidden">
                      <div 
                        class={`h-full rounded-full bg-gradient-to-r ${getScoreGradient(analysis()!.overall_score)} transition-all duration-500`}
                        style={{ width: `${analysis()!.overall_score}%` }}
                      />
                    </div>
                  </div>

                  {/* Targets */}
                  <div class="px-4 py-3 grid grid-cols-2 gap-3 border-b border-terminal-800">
                    <Show when={analysis()!.target_1}>
                      <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1 flex items-center gap-1">
                          <Target class="w-3 h-3" /> Target
                        </div>
                        <div class="text-sm font-mono text-success-400">
                          {formatCurrency(analysis()!.target_1!)}
                        </div>
                        <Show when={potential !== null}>
                          <div class={`text-[10px] ${potential! >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                            {potential! >= 0 ? '+' : ''}{potential!.toFixed(1)}%
                          </div>
                        </Show>
                      </div>
                    </Show>
                    <Show when={analysis()!.stop_loss}>
                      <div>
                        <div class="text-[10px] text-gray-500 uppercase mb-1 flex items-center gap-1">
                          <AlertTriangle class="w-3 h-3" /> Stop
                        </div>
                        <div class="text-sm font-mono text-danger-400">
                          {formatCurrency(analysis()!.stop_loss!)}
                        </div>
                      </div>
                    </Show>
                  </div>

                  {/* Key Insight */}
                  <Show when={analysis()!.key_insight}>
                    <div class="px-4 py-3 border-b border-terminal-800">
                      <div class="text-[10px] text-gray-500 uppercase mb-1">Key Insight</div>
                      <p class="text-xs text-gray-300 line-clamp-2">{analysis()!.key_insight}</p>
                    </div>
                  </Show>

                  {/* Full Analysis Link */}
                  <button
                    onClick={goToFullAnalysis}
                    class="w-full px-4 py-3 flex items-center justify-between text-xs text-gray-400 hover:text-accent-400 hover:bg-terminal-800/50 transition-colors group"
                  >
                    <span>View Full Analysis</span>
                    <ChevronRight class="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                </>
              );
            })()}
          </Show>
        </div>
      </Show>
    </div>
  );
}

export default PortfolioAnalyzer;
