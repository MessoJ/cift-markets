/**
 * WatchlistAnalyzer - Inline Row Expansion for Watchlist Items
 * 
 * Design: A subtle dot indicator that expands the table row to show
 * a mini analysis card with score ring and quick metrics.
 * Different from Trading (horizontal) and Portfolio (dropdown) - row expansion.
 * 
 * Brand Colors: Orange accent (#f97316), terminal blacks, success green, danger red
 */

import { createSignal, Show, onCleanup, createEffect } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { ChevronRight, Gauge, Shield, Zap, X } from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { apiClient } from '~/lib/api/client';

interface WatchlistAnalyzerProps {
  symbol: string;
  onExpand?: (expanded: boolean) => void;
  className?: string;
}

interface AnalysisResult {
  overall_score: number;
  suggested_action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  risk_level: string;
  technical_signal: string;
  momentum_score?: number;
}

export function WatchlistAnalyzer(props: WatchlistAnalyzerProps) {
  const navigate = useNavigate();
  const [isExpanded, setIsExpanded] = createSignal(false);
  const [isAnalyzing, setIsAnalyzing] = createSignal(false);
  const [analysis, setAnalysis] = createSignal<AnalysisResult | null>(null);
  const [error, setError] = createSignal(false);
  const [panelPosition, setPanelPosition] = createSignal({ top: 0, left: 0 });
  let triggerRef: HTMLButtonElement | undefined;
  let panelRef: HTMLDivElement | undefined;

  // Click outside handler
  const handleClickOutside = (e: MouseEvent) => {
    if (panelRef && !panelRef.contains(e.target as Node) && 
        triggerRef && !triggerRef.contains(e.target as Node)) {
      setIsExpanded(false);
      props.onExpand?.(false);
    }
  };

  createEffect(() => {
    if (isExpanded()) {
      document.addEventListener('click', handleClickOutside);
    } else {
      document.removeEventListener('click', handleClickOutside);
    }
  });

  onCleanup(() => {
    document.removeEventListener('click', handleClickOutside);
  });

  const updatePosition = () => {
    if (triggerRef) {
      const rect = triggerRef.getBoundingClientRect();
      const panelWidth = 500;
      let left = rect.left + rect.width / 2 - panelWidth / 2;
      // Keep within viewport
      left = Math.max(16, Math.min(left, window.innerWidth - panelWidth - 16));
      setPanelPosition({
        top: rect.bottom + 8,
        left
      });
    }
  };

  const handleAnalyze = async (e: Event) => {
    e.stopPropagation();
    e.preventDefault();
    
    const newExpanded = !isExpanded();
    setIsExpanded(newExpanded);
    props.onExpand?.(newExpanded);

    if (newExpanded) {
      updatePosition();
    }

    if (!newExpanded || analysis()) return;

    setIsAnalyzing(true);
    setError(false);

    try {
      const data = await apiClient.get(`/analysis/${props.symbol}`);
      setAnalysis({
        overall_score: data.overall_score || 50,
        suggested_action: data.suggested_action || 'HOLD',
        confidence: data.confidence || 0.5,
        risk_level: data.risk?.risk_level || 'medium',
        technical_signal: data.technical?.signal || 'NEUTRAL',
        momentum_score: data.momentum?.score,
      });
    } catch (err) {
      console.error('Watchlist analysis failed:', err);
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
      case 'BUY': return { dot: 'bg-success-400', text: 'text-success-400', ring: 'ring-success-400' };
      case 'SELL': return { dot: 'bg-danger-400', text: 'text-danger-400', ring: 'ring-danger-400' };
      default: return { dot: 'bg-warning-400', text: 'text-warning-400', ring: 'ring-warning-400' };
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-success-400';
      case 'high': return 'text-danger-400';
      default: return 'text-warning-400';
    }
  };

  // Mini score ring component
  const ScoreRing = (props: { score: number; size?: number }) => {
    const size = props.size || 36;
    const strokeWidth = 3;
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const progress = (props.score / 100) * circumference;
    
    const color = props.score >= 70 ? '#22c55e' : props.score >= 50 ? '#f59e0b' : '#ef4444';
    
    return (
      <div class="relative" style={{ width: `${size}px`, height: `${size}px` }}>
        <svg class="transform -rotate-90" viewBox={`0 0 ${size} ${size}`}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            stroke-width={strokeWidth}
            fill="none"
            class="text-terminal-700"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            stroke-width={strokeWidth}
            fill="none"
            stroke-linecap="round"
            stroke-dasharray={String(circumference)}
            stroke-dashoffset={circumference - progress}
            class="transition-all duration-700 ease-out"
          />
        </svg>
        <div class="absolute inset-0 flex items-center justify-center">
          <span class="text-[10px] font-bold font-mono text-white">{props.score}</span>
        </div>
      </div>
    );
  };

  return (
    <div class={`${props.className || ''}`}>
      {/* Trigger - Subtle Crosshair Icon */}
      <button
        ref={triggerRef}
        onClick={handleAnalyze}
        class={`
          inline-flex items-center justify-center
          w-5 h-5 rounded
          transition-all duration-200
          ${isExpanded() 
            ? 'bg-accent-500/20 text-accent-400' 
            : 'bg-transparent text-gray-500 hover:text-accent-400 hover:bg-accent-500/10'
          }
        `}
        title={`Analyze ${props.symbol}`}
      >
        <AIIcon size={14} animate={isAnalyzing()} />
      </button>

      {/* Expanded Panel - Fixed positioning to avoid clipping */}
      <Show when={isExpanded()}>
        <div 
          ref={panelRef}
          class="fixed z-[9999]"
          style={{
            "top": `${panelPosition().top}px`,
            "left": `${panelPosition().left}px`,
            "width": "min(500px, calc(100vw - 32px))",
          }}
        >
          <div class="bg-terminal-900 border border-terminal-700 rounded-lg shadow-2xl overflow-hidden">
            {/* Close button */}
            <button
              onClick={(e) => { e.stopPropagation(); setIsExpanded(false); props.onExpand?.(false); }}
              class="absolute top-2 right-2 w-5 h-5 flex items-center justify-center rounded text-gray-500 hover:text-white hover:bg-terminal-700 transition-colors z-10"
            >
              <X class="w-3 h-3" />
            </button>
            
            {/* Loading */}
            <Show when={isAnalyzing()}>
              <div class="px-4 py-3 flex items-center gap-3">
                <div class="w-5 h-5 rounded-full border-2 border-accent-500 border-t-transparent animate-spin" />
                <span class="text-xs text-gray-400">Running AI analysis...</span>
              </div>
            </Show>

            {/* Error */}
            <Show when={error() && !isAnalyzing()}>
              <div class="px-4 py-3 text-xs text-danger-400">
                Failed to analyze. Try again.
              </div>
            </Show>

            {/* Results */}
            <Show when={analysis() && !isAnalyzing() && !error()}>
              {(() => {
                const colors = getActionColor(analysis()!.suggested_action);
                return (
                  <div class="flex items-stretch">
                    {/* Left: Score Ring + Action */}
                    <div class="flex items-center gap-3 px-4 py-3 border-r border-terminal-700">
                      <ScoreRing score={analysis()!.overall_score} />
                      <div>
                        <div class={`text-sm font-bold ${colors.text}`}>
                          {analysis()!.suggested_action}
                        </div>
                        <div class="text-[10px] text-gray-500">
                          {Math.round(analysis()!.confidence * 100)}% conf
                        </div>
                      </div>
                    </div>

                    {/* Middle: Metrics */}
                    <div class="flex-1 flex items-center gap-4 px-4 py-3">
                      {/* Technical Signal */}
                      <div class="flex items-center gap-1.5">
                        <Zap class="w-3.5 h-3.5 text-accent-400" />
                        <div>
                          <div class="text-[9px] text-gray-500 uppercase">Technical</div>
                          <div class={`text-xs font-medium ${
                            analysis()!.technical_signal?.includes('BULL') ? 'text-success-400' :
                            analysis()!.technical_signal?.includes('BEAR') ? 'text-danger-400' : 'text-warning-400'
                          }`}>
                            {analysis()!.technical_signal?.replace('_', ' ')}
                          </div>
                        </div>
                      </div>

                      {/* Risk */}
                      <div class="flex items-center gap-1.5">
                        <Shield class={`w-3.5 h-3.5 ${getRiskColor(analysis()!.risk_level)}`} />
                        <div>
                          <div class="text-[9px] text-gray-500 uppercase">Risk</div>
                          <div class={`text-xs font-medium uppercase ${getRiskColor(analysis()!.risk_level)}`}>
                            {analysis()!.risk_level}
                          </div>
                        </div>
                      </div>

                      {/* Momentum */}
                      <Show when={analysis()!.momentum_score !== undefined}>
                        <div class="flex items-center gap-1.5">
                          <Gauge class="w-3.5 h-3.5 text-accent-400" />
                          <div>
                            <div class="text-[9px] text-gray-500 uppercase">Momentum</div>
                            <div class={`text-xs font-mono font-medium ${
                              analysis()!.momentum_score! >= 60 ? 'text-success-400' :
                              analysis()!.momentum_score! >= 40 ? 'text-warning-400' : 'text-danger-400'
                            }`}>
                              {analysis()!.momentum_score}
                            </div>
                          </div>
                        </div>
                      </Show>
                    </div>

                    {/* Right: Full Analysis Link */}
                    <button
                      onClick={goToFullAnalysis}
                      class="flex items-center gap-1 px-4 py-3 bg-terminal-800/50 hover:bg-accent-500/10 text-gray-400 hover:text-accent-400 transition-colors group"
                    >
                      <span class="text-xs">Full</span>
                      <ChevronRight class="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                    </button>
                  </div>
                );
              })()}
            </Show>
          </div>
        </div>
      </Show>
    </div>
  );
}

export default WatchlistAnalyzer;
