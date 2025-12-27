/**
 * InlineAnalyzer - Expanding Inline Analysis Component for Trading Page
 * 
 * Design: A tiny icon that rotates while analyzing, then expands horizontally
 * to reveal analysis results. Arrow at the end for full analysis page.
 * 
 * Brand Colors: Orange accent (#f97316), terminal blacks, success green, danger red
 */

import { createSignal, Show } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { ChevronRight, Shield, Target, Zap } from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { apiClient } from '~/lib/api/client';

interface InlineAnalyzerProps {
  symbol: string;
  className?: string;
}

interface AnalysisResult {
  overall_score: number;
  suggested_action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  risk_level: string;
  technical_signal: string;
  price_target?: number;
  stop_loss?: number;
}

export function InlineAnalyzer(props: InlineAnalyzerProps) {
  const navigate = useNavigate();
  const [isExpanded, setIsExpanded] = createSignal(false);
  const [isAnalyzing, setIsAnalyzing] = createSignal(false);
  const [analysis, setAnalysis] = createSignal<AnalysisResult | null>(null);
  const [error, setError] = createSignal(false);

  const handleAnalyze = async (e: Event) => {
    e.stopPropagation();
    e.preventDefault();
    
    if (isExpanded() && analysis()) {
      // Already expanded and has data - just collapse
      setIsExpanded(false);
      return;
    }

    setIsAnalyzing(true);
    setError(false);
    setIsExpanded(true);

    try {
      const data = await apiClient.get(`/analysis/${props.symbol}`);
      setAnalysis({
        overall_score: data.overall_score || 50,
        suggested_action: data.suggested_action || 'HOLD',
        confidence: data.confidence || 0.5,
        risk_level: data.risk?.risk_level || 'medium',
        technical_signal: data.technical?.signal || 'NEUTRAL',
        price_target: data.target_1,
        stop_loss: data.stop_loss,
      });
    } catch (err) {
      console.error('Inline analysis failed:', err);
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

  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY': return 'text-success-400';
      case 'SELL': return 'text-danger-400';
      default: return 'text-warning-400';
    }
  };

  const getActionBg = (action: string) => {
    switch (action) {
      case 'BUY': return 'bg-success-500/20 border-success-500/30';
      case 'SELL': return 'bg-danger-500/20 border-danger-500/30';
      default: return 'bg-warning-500/20 border-warning-500/30';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-success-400';
    if (score >= 50) return 'text-warning-400';
    return 'text-danger-400';
  };

  return (
    <div class={`inline-flex items-center ${props.className || ''}`}>
      {/* Trigger Button */}
      <button
        onClick={handleAnalyze}
        class={`
          relative flex items-center justify-center
          w-6 h-6 rounded-md
          transition-all duration-300 ease-out
          ${isExpanded() 
            ? 'bg-accent-500/20 text-accent-400 rounded-r-none' 
            : 'bg-terminal-800 hover:bg-accent-500/20 text-gray-400 hover:text-accent-400'
          }
        `}
        title={`Analyze ${props.symbol}`}
      >
        <AIIcon size={14} animate={isAnalyzing()} />
      </button>

      {/* Expanding Content */}
      <div 
        class={`
          flex items-center overflow-hidden
          transition-all duration-300 ease-out
          ${isExpanded() 
            ? 'max-w-[400px] opacity-100' 
            : 'max-w-0 opacity-0'
          }
        `}
      >
        <div class="flex items-center gap-2 px-3 py-1 bg-terminal-900 border border-l-0 border-terminal-700 rounded-r-md whitespace-nowrap">
          {/* Loading State */}
          <Show when={isAnalyzing()}>
            <span class="text-xs text-gray-400 animate-pulse">Analyzing...</span>
          </Show>

          {/* Error State */}
          <Show when={error() && !isAnalyzing()}>
            <span class="text-xs text-danger-400">Analysis failed</span>
          </Show>

          {/* Results */}
          <Show when={analysis() && !isAnalyzing() && !error()}>
            {/* Action Badge */}
            <div class={`px-2 py-0.5 rounded text-xs font-bold border ${getActionBg(analysis()!.suggested_action)}`}>
              <span class={getActionColor(analysis()!.suggested_action)}>
                {analysis()!.suggested_action}
              </span>
            </div>

            {/* Score */}
            <div class="flex items-center gap-1">
              <Zap class="w-3 h-3 text-accent-400" />
              <span class={`text-xs font-mono font-bold ${getScoreColor(analysis()!.overall_score)}`}>
                {analysis()!.overall_score}
              </span>
            </div>

            {/* Risk */}
            <div class="flex items-center gap-1">
              <Shield class={`w-3 h-3 ${
                analysis()!.risk_level === 'low' ? 'text-success-400' :
                analysis()!.risk_level === 'high' ? 'text-danger-400' : 'text-warning-400'
              }`} />
              <span class="text-xs text-gray-400 uppercase">
                {analysis()!.risk_level}
              </span>
            </div>

            {/* Target (if buy) */}
            <Show when={analysis()!.suggested_action === 'BUY' && analysis()!.price_target}>
              <div class="flex items-center gap-1">
                <Target class="w-3 h-3 text-success-400" />
                <span class="text-xs font-mono text-success-400">
                  ${analysis()!.price_target?.toFixed(2)}
                </span>
              </div>
            </Show>

            {/* Full Analysis Arrow */}
            <button
              onClick={goToFullAnalysis}
              class="group flex items-center gap-1 ml-2 pl-2 border-l border-terminal-700 text-gray-400 hover:text-accent-400 transition-colors"
              title="View full analysis"
            >
              <span class="text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                Details
              </span>
              <ChevronRight class="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </button>
          </Show>
        </div>
      </div>
    </div>
  );
}

export default InlineAnalyzer;
