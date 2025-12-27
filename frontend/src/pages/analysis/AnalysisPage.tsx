/**
 * CIFT Markets - Professional Stock Analysis Page (Redesigned)
 * 
 * Features:
 * - Portfolio-aware analysis with position context
 * - ML model predictions integrated into trade suggestions
 * - Expandable inline sections (no modals)
 * - Clean, professional UI maintaining brand consistency
 * - Real-time data integration
 * 
 * ALL DATA FROM BACKEND APIs
 */

import { createSignal, createEffect, Show, For, onMount, onCleanup, createMemo } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  RefreshCw,
  Search,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  BarChart2,
  DollarSign,
  Newspaper,
  Zap,
  Shield,
  Info,
  Clock,
  Award,
  ArrowUp,
  ArrowDown,
  Minus,
  Sparkles,
  Briefcase,
  Bot,
  Cpu,
  LineChart,
  PieChart,
  Wallet,
  ArrowRightLeft,
  Plus,
  MinusCircle,
} from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { formatCurrency, formatPercent, formatNumber } from '~/lib/utils/format';
import { apiClient, Position, MLPrediction } from '~/lib/api/client';
import { ExpandableAnalysisSection } from '~/components/analysis/ExpandableAnalysisSection';
import { TradeSetupVisualizer } from '~/components/analysis/TradeSetupVisualizer';

// ============================================================================
// TYPES
// ============================================================================

interface TechnicalAnalysis {
  score: number;
  signal: string;
  rsi_14?: number;
  rsi_signal?: string;
  macd_line?: number;
  macd_signal_line?: number;
  macd_crossover?: string;
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;
  atr_percent?: number;
  volume_vs_avg?: number;
  short_term_trend?: string;
  medium_term_trend?: string;
  long_term_trend?: string;
  support_levels?: number[];
  resistance_levels?: number[];
  bollinger_upper?: number;
  bollinger_lower?: number;
}

interface FundamentalAnalysis {
  score: number;
  signal: string;
  pe_ratio?: number;
  pb_ratio?: number;
  ps_ratio?: number;
  peg_ratio?: number;
  ev_ebitda?: number;
  roe?: number;
  roa?: number;
  profit_margin?: number;
  operating_margin?: number;
  revenue_growth_yoy?: number;
  earnings_growth_yoy?: number;
  debt_to_equity?: number;
  current_ratio?: number;
  dividend_yield?: number;
}

interface SentimentAnalysis {
  score: number;
  signal: string;
  news_sentiment?: number;
  news_volume?: number;
  news_trend?: string;
  analyst_rating?: string;
  analyst_target?: number;
  analyst_target_upside?: number;
  institutional_ownership?: number;
  institutional_change?: number;
  insider_net_shares?: number;
}

interface MomentumAnalysis {
  score: number;
  signal: string;
  return_1d?: number;
  return_1w?: number;
  return_1m?: number;
  return_3m?: number;
  return_6m?: number;
  return_12m?: number;
  return_ytd?: number;
  momentum_12_1?: number;
  momentum_percentile?: number;
  relative_strength?: number;
}

interface RiskAnalysis {
  score: number;
  risk_level: string;
  volatility_30d?: number;
  beta?: number;
  max_drawdown_1y?: number;
  current_drawdown?: number;
  var_95?: number;
}

interface StockAnalysis {
  symbol: string;
  timestamp: string;
  price: number;
  change: number;
  change_percent: number;
  technical: TechnicalAnalysis;
  fundamental: FundamentalAnalysis;
  sentiment: SentimentAnalysis;
  momentum: MomentumAnalysis;
  risk: RiskAnalysis;
  overall_score: number;
  rating: string;
  confidence: number;
  bullish_factors: string[];
  bearish_factors: string[];
  key_risks: string[];
  trade_suggestion?: {
    action: string;
    entry_zone_low?: number;
    entry_zone_high?: number;
    stop_loss?: number;
    target_1?: number;
    target_2?: number;
    risk_reward?: number;
  };
  analysis_latency_ms?: number;
}

// ============================================================================
// HELPER COMPONENTS
// ============================================================================

const ScoreGauge = (props: { score: number; label: string; size?: 'sm' | 'md' | 'lg' }) => {
  const size = props.size || 'md';
  const sizes = {
    sm: { outer: 80, stroke: 8, fontSize: 'text-lg', labelSize: 'text-xs' },
    md: { outer: 120, stroke: 12, fontSize: 'text-2xl', labelSize: 'text-sm' },
    lg: { outer: 160, stroke: 14, fontSize: 'text-4xl', labelSize: 'text-base' },
  };
  
  const config = sizes[size];
  const radius = (config.outer - config.stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (props.score / 100) * circumference;
  
  const getColor = (score: number) => {
    if (score >= 70) return { stroke: '#10b981', text: 'text-emerald-400' };
    if (score >= 55) return { stroke: '#22c55e', text: 'text-green-400' };
    if (score >= 45) return { stroke: '#eab308', text: 'text-yellow-400' };
    if (score >= 30) return { stroke: '#f97316', text: 'text-orange-400' };
    return { stroke: '#ef4444', text: 'text-red-400' };
  };
  
  const colors = getColor(props.score);
  
  return (
    <div class="flex flex-col items-center">
      <div class="relative" style={{ width: `${config.outer}px`, height: `${config.outer}px` }}>
        <svg class="transform -rotate-90" viewBox={`0 0 ${config.outer} ${config.outer}`}>
          <circle
            cx={config.outer / 2}
            cy={config.outer / 2}
            r={radius}
            stroke="currentColor"
            stroke-width={config.stroke}
            fill="none"
            class="text-slate-700/50"
          />
          <circle
            cx={config.outer / 2}
            cy={config.outer / 2}
            r={radius}
            stroke={colors.stroke}
            stroke-width={config.stroke}
            fill="none"
            stroke-linecap="round"
            stroke-dasharray={circumference}
            stroke-dashoffset={circumference - progress}
            class="transition-all duration-1000 ease-out"
          />
        </svg>
        <div class="absolute inset-0 flex flex-col items-center justify-center">
          <span class={`${config.fontSize} font-bold ${colors.text}`}>{Math.round(props.score)}</span>
        </div>
      </div>
      <span class={`${config.labelSize} text-slate-400 mt-1`}>{props.label}</span>
    </div>
  );
};

const RatingBadge = (props: { rating: string }) => {
  const configs: Record<string, { bg: string; text: string; border: string }> = {
    'STRONG_BUY': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/40' },
    'BUY': { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/40' },
    'HOLD': { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/40' },
    'SELL': { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/40' },
    'STRONG_SELL': { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/40' },
  };
  
  const config = configs[props.rating] || configs['HOLD'];
  
  return (
    <span class={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-semibold border ${config.bg} ${config.text} ${config.border}`}>
      <Award class="h-4 w-4" />
      {props.rating.replace('_', ' ')}
    </span>
  );
};

const MLPredictionCard = (props: { prediction: MLPrediction | null; loading: boolean }) => {
  const directionConfig = () => {
    if (!props.prediction) return { icon: Minus, color: 'text-slate-400', bg: 'bg-slate-500/10', label: 'N/A' };
    switch (props.prediction.direction) {
      case 'long':
        return { icon: TrendingUp, color: 'text-emerald-400', bg: 'bg-emerald-500/10', label: 'BULLISH' };
      case 'short':
        return { icon: TrendingDown, color: 'text-red-400', bg: 'bg-red-500/10', label: 'BEARISH' };
      default:
        return { icon: Minus, color: 'text-yellow-400', bg: 'bg-yellow-500/10', label: 'NEUTRAL' };
    }
  };

  const config = directionConfig();
  const Icon = config.icon;

  return (
    <div class="bg-gradient-to-br from-purple-500/10 via-slate-800/50 to-blue-500/10 border border-purple-500/30 rounded-xl p-5">
      <div class="flex items-center gap-2 mb-4">
        <div class="p-2 bg-purple-500/20 rounded-lg">
          <AIIcon class="h-5 w-5 text-purple-400" />
        </div>
        <div>
          <h3 class="text-sm font-semibold text-white">AI Model Prediction</h3>
          <p class="text-xs text-slate-400">Ensemble ML Analysis</p>
        </div>
        <Show when={props.prediction?.inference_latency_ms}>
          <span class="ml-auto text-xs text-slate-500">{props.prediction!.inference_latency_ms.toFixed(0)}ms</span>
        </Show>
      </div>

      <Show when={props.loading}>
        <div class="flex items-center justify-center py-8">
          <RefreshCw class="h-6 w-6 text-purple-400 animate-spin" />
        </div>
      </Show>

      <Show when={!props.loading && !props.prediction}>
        <div class="text-center py-6 text-slate-500">
          <Bot class="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p class="text-sm">ML prediction unavailable</p>
        </div>
      </Show>

      <Show when={!props.loading && props.prediction}>
        <div class="space-y-4">
          {/* Direction & Confidence */}
          <div class="flex items-center justify-between">
            <div class={`flex items-center gap-2 px-3 py-2 rounded-lg ${config.bg}`}>
              <Icon class={`h-5 w-5 ${config.color}`} />
              <span class={`font-bold ${config.color}`}>{config.label}</span>
            </div>
            <div class="text-right">
              <div class="text-xs text-slate-400">Confidence</div>
              <div class={`text-xl font-bold ${
                (props.prediction?.confidence ?? 0) > 0.7 ? 'text-emerald-400' :
                (props.prediction?.confidence ?? 0) > 0.5 ? 'text-yellow-400' : 'text-slate-400'
              }`}>
                {((props.prediction?.confidence ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Should Trade Indicator */}
          <div class={`p-3 rounded-lg border ${
            props.prediction?.should_trade 
              ? 'bg-emerald-500/10 border-emerald-500/30' 
              : 'bg-slate-700/30 border-slate-600/30'
          }`}>
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-300">Trade Signal</span>
              <span class={`text-sm font-semibold ${
                props.prediction?.should_trade ? 'text-emerald-400' : 'text-slate-400'
              }`}>
                {props.prediction?.should_trade ? '✓ Active Signal' : '○ No Signal'}
              </span>
            </div>
          </div>

          {/* Model Details Grid */}
          <div class="grid grid-cols-2 gap-3 pt-2">
            <div class="text-center p-2 bg-slate-800/50 rounded-lg">
              <div class="text-xs text-slate-500">Direction Prob.</div>
              <div class="text-sm font-semibold text-white">
                {((props.prediction?.direction_probability ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div class="text-center p-2 bg-slate-800/50 rounded-lg">
              <div class="text-xs text-slate-500">Model Agreement</div>
              <div class="text-sm font-semibold text-white">
                {props.prediction?.model_agreement ?? 0}/5
              </div>
            </div>
            <div class="text-center p-2 bg-slate-800/50 rounded-lg">
              <div class="text-xs text-slate-500">Market Regime</div>
              <div class="text-sm font-semibold text-white capitalize">
                {props.prediction?.current_regime ?? 'Unknown'}
              </div>
            </div>
            <div class="text-center p-2 bg-slate-800/50 rounded-lg">
              <div class="text-xs text-slate-500">Position Size</div>
              <div class="text-sm font-semibold text-white">
                {((props.prediction?.position_size ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Risk Params */}
          <Show when={props.prediction?.stop_loss_bps || props.prediction?.take_profit_bps}>
            <div class="flex items-center justify-between text-xs pt-2 border-t border-slate-700/50">
              <span class="text-red-400">
                SL: {props.prediction?.stop_loss_bps?.toFixed(0) ?? 'N/A'} bps
              </span>
              <span class="text-emerald-400">
                TP: {props.prediction?.take_profit_bps?.toFixed(0) ?? 'N/A'} bps
              </span>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
};

const PortfolioContextCard = (props: { 
  position: Position | null; 
  analysis: StockAnalysis | null;
  prediction: MLPrediction | null;
}) => {
  const hasPosition = () => !!props.position;
  
  const getRecommendation = () => {
    const pos = props.position;
    const analysis = props.analysis;
    const pred = props.prediction;
    
    if (!analysis) return null;
    
    if (hasPosition()) {
      // User owns this stock
      const pnlPercent = pos!.unrealized_pnl_pct;
      const rating = analysis.rating;
      const shouldTrade = pred?.should_trade;
      const direction = pred?.direction;
      
      if (rating === 'STRONG_SELL' || rating === 'SELL') {
        return {
          action: 'Consider Reducing',
          reason: `Analysis suggests ${rating.replace('_', ' ')} rating. You have ${formatPercent(pnlPercent)} unrealized P&L.`,
          type: 'warning' as const,
        };
      }
      if (pnlPercent < -10 && analysis.overall_score < 45) {
        return {
          action: 'Review Position',
          reason: `Down ${formatPercent(Math.abs(pnlPercent))} with weak score (${analysis.overall_score}/100). Consider your thesis.`,
          type: 'warning' as const,
        };
      }
      if (pnlPercent > 20 && (rating === 'HOLD' || analysis.overall_score < 55)) {
        return {
          action: 'Consider Taking Profits',
          reason: `Up ${formatPercent(pnlPercent)} with ${rating} rating. Lock in gains or set trailing stop.`,
          type: 'info' as const,
        };
      }
      if (rating === 'BUY' || rating === 'STRONG_BUY') {
        return {
          action: 'Hold / Add on Dips',
          reason: `Strong ${rating.replace('_', ' ')} rating supports your position. Consider adding on weakness.`,
          type: 'success' as const,
        };
      }
      return {
        action: 'Monitor Position',
        reason: `Current rating: ${rating.replace('_', ' ')}. Watch for changes in thesis.`,
        type: 'neutral' as const,
      };
    } else {
      // User doesn't own this stock
      const rating = analysis.rating;
      const shouldTrade = pred?.should_trade;
      
      if ((rating === 'STRONG_BUY' || rating === 'BUY') && shouldTrade && pred?.direction === 'long') {
        return {
          action: 'Consider Buying',
          reason: `${rating.replace('_', ' ')} rating with active ML signal. Good entry opportunity.`,
          type: 'success' as const,
        };
      }
      if (rating === 'STRONG_BUY' || rating === 'BUY') {
        return {
          action: 'Watchlist Candidate',
          reason: `${rating.replace('_', ' ')} rating. Add to watchlist and wait for better entry.`,
          type: 'info' as const,
        };
      }
      if (rating === 'STRONG_SELL' || rating === 'SELL') {
        return {
          action: 'Avoid / Short Candidate',
          reason: `${rating.replace('_', ' ')} rating. Not recommended for long positions.`,
          type: 'warning' as const,
        };
      }
      return {
        action: 'Wait for Better Setup',
        reason: `Currently rated ${rating.replace('_', ' ')}. Look for clearer signals.`,
        type: 'neutral' as const,
      };
    }
  };

  const recommendation = getRecommendation();
  
  const typeStyles = {
    success: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    warning: 'bg-orange-500/10 border-orange-500/30 text-orange-400',
    info: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
    neutral: 'bg-slate-500/10 border-slate-600/30 text-slate-400',
  };

  return (
    <div class="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div class={`px-5 py-3 border-b border-slate-700/50 ${hasPosition() ? 'bg-accent-500/10' : 'bg-slate-800/30'}`}>
        <div class="flex items-center gap-2">
          {hasPosition() ? (
            <>
              <Briefcase class="h-5 w-5 text-accent-400" />
              <span class="font-semibold text-white">Your Position</span>
              <span class="ml-2 px-2 py-0.5 rounded text-xs font-medium bg-accent-500/20 text-accent-400">
                {props.position!.side.toUpperCase()}
              </span>
            </>
          ) : (
            <>
              <Wallet class="h-5 w-5 text-slate-400" />
              <span class="font-semibold text-slate-300">Not in Portfolio</span>
            </>
          )}
        </div>
      </div>

      {/* Position Details */}
      <Show when={hasPosition()}>
        <div class="p-5">
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <div class="text-xs text-slate-500 mb-1">Shares</div>
              <div class="text-lg font-mono font-semibold text-white">{formatNumber(props.position!.quantity)}</div>
            </div>
            <div>
              <div class="text-xs text-slate-500 mb-1">Avg Cost</div>
              <div class="text-lg font-mono font-semibold text-white">{formatCurrency(props.position!.avg_cost)}</div>
            </div>
            <div>
              <div class="text-xs text-slate-500 mb-1">Market Value</div>
              <div class="text-lg font-mono font-semibold text-white">{formatCurrency(props.position!.market_value)}</div>
            </div>
            <div>
              <div class="text-xs text-slate-500 mb-1">Total P&L</div>
              <div class={`text-lg font-mono font-semibold ${props.position!.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {formatCurrency(props.position!.unrealized_pnl)}
                <span class="text-sm ml-1">({formatPercent(props.position!.unrealized_pnl_pct)})</span>
              </div>
            </div>
          </div>
          
          {/* Today's P&L */}
          <div class="flex items-center justify-between py-2 px-3 bg-slate-900/50 rounded-lg">
            <span class="text-sm text-slate-400">Today's Change</span>
            <span class={`font-mono font-semibold ${props.position!.day_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {formatCurrency(props.position!.day_pnl)} ({formatPercent(props.position!.day_pnl_pct)})
            </span>
          </div>
        </div>
      </Show>

      {/* Recommendation */}
      <Show when={recommendation}>
        <div class={`mx-5 mb-5 p-4 rounded-lg border ${typeStyles[recommendation!.type]}`}>
          <div class="flex items-start gap-3">
            <div class="mt-0.5">
              {recommendation!.type === 'success' && <TrendingUp class="h-5 w-5" />}
              {recommendation!.type === 'warning' && <AlertTriangle class="h-5 w-5" />}
              {recommendation!.type === 'info' && <Info class="h-5 w-5" />}
              {recommendation!.type === 'neutral' && <Minus class="h-5 w-5" />}
            </div>
            <div>
              <div class="font-semibold text-white">{recommendation!.action}</div>
              <div class="text-sm opacity-80 mt-0.5">{recommendation!.reason}</div>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function AnalysisPage() {
  const params = useParams<{ symbol?: string }>();
  const navigate = useNavigate();
  
  // State
  const [searchSymbol, setSearchSymbol] = createSignal(params.symbol || '');
  const [analysis, setAnalysis] = createSignal<StockAnalysis | null>(null);
  const [position, setPosition] = createSignal<Position | null>(null);
  const [prediction, setPrediction] = createSignal<MLPrediction | null>(null);
  const [news, setNews] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [predictionLoading, setPredictionLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [lastUpdated, setLastUpdated] = createSignal<Date | null>(null);
  
  // Expandable sections state
  const [expandedSections, setExpandedSections] = createSignal<Set<string>>(new Set());
  
  let refreshInterval: number | null = null;
  
  const toggleSection = (section: string) => {
    const expanded = new Set(expandedSections());
    if (expanded.has(section)) {
      expanded.delete(section);
    } else {
      expanded.add(section);
    }
    setExpandedSections(expanded);
  };
  
  const fetchAnalysis = async (symbol: string) => {
    if (!symbol || symbol.trim() === '') {
      setError('Please enter a valid stock symbol');
      return;
    }
    
    const sym = symbol.toUpperCase().trim();
    setLoading(true);
    setError(null);
    
    try {
      // Fetch analysis, position, and news in parallel
      const [analysisData, posData, newsData] = await Promise.all([
        apiClient.get(`/analysis/${sym}`),
        apiClient.getPosition(sym).catch(() => null),
        apiClient.getNews({ symbol: sym, limit: 5 }).catch(() => ({ articles: [] }))
      ]);
      
      if (!analysisData || !analysisData.symbol) {
        throw new Error('Invalid analysis data received');
      }
      
      setAnalysis(analysisData);
      setPosition(posData);
      setNews(Array.isArray(newsData) ? newsData : newsData?.articles || []);
      setLastUpdated(new Date());
      
      // Fetch ML prediction separately (might take longer or fail)
      setPredictionLoading(true);
      try {
        const predData = await apiClient.getMLPrediction(sym);
        setPrediction(predData);
      } catch (e) {
        console.warn('ML prediction unavailable:', e);
        setPrediction(null);
      } finally {
        setPredictionLoading(false);
      }
      
      // Update URL
      if (params.symbol !== sym) {
        navigate(`/analysis/${sym}`, { replace: true });
      }
    } catch (err: any) {
      console.error('Analysis fetch error:', err);
      
      if (err.message?.includes('timeout')) {
        setError('Request timed out. Please try again.');
      } else if (err.status === 404) {
        setError(`Symbol "${sym}" not found.`);
      } else if (err.status === 429) {
        setError('Too many requests. Please wait a moment.');
      } else {
        setError(err.message || 'Failed to fetch analysis.');
      }
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSearch = (e: Event) => {
    e.preventDefault();
    fetchAnalysis(searchSymbol());
  };
  
  onMount(() => {
    if (params.symbol) {
      setSearchSymbol(params.symbol);
      fetchAnalysis(params.symbol);
    }
    
    // Auto-refresh every 60 seconds
    refreshInterval = window.setInterval(() => {
      const current = analysis();
      if (current) {
        fetchAnalysis(current.symbol);
      }
    }, 60000);
  });
  
  onCleanup(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });
  
  createEffect(() => {
    if (params.symbol && params.symbol !== analysis()?.symbol) {
      setSearchSymbol(params.symbol);
      fetchAnalysis(params.symbol);
    }
  });
  
  return (
    <div class="min-h-screen bg-slate-950 text-white">
      {/* Header Section */}
      <div class="border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm sticky top-0 z-10">
        <div class="max-w-7xl mx-auto px-4 py-4">
          <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 class="text-xl font-bold flex items-center gap-2">
                <AIIcon size={24} class="text-accent-400" />
                Stock Analysis
                <span class="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400 text-xs font-medium">
                  <AIIcon class="h-3 w-3" />
                  AI + ML
                </span>
              </h1>
              <p class="text-slate-400 text-sm mt-0.5">Technical, Fundamental, Sentiment & ML-Powered Analysis</p>
            </div>
            
            {/* Search Bar */}
            <form onSubmit={handleSearch} class="flex gap-2 w-full md:w-auto">
              <div class="relative flex-1 md:w-64">
                <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-500" />
                <input
                  type="text"
                  value={searchSymbol()}
                  onInput={(e) => setSearchSymbol(e.currentTarget.value.toUpperCase())}
                  placeholder="Symbol (AAPL, MSFT...)"
                  class="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent text-sm"
                />
              </div>
              <button
                type="submit"
                disabled={loading() || !searchSymbol()}
                class="px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center gap-2 text-sm"
              >
                {loading() ? <RefreshCw class="h-4 w-4 animate-spin" /> : <BarChart2 class="h-4 w-4" />}
                Analyze
              </button>
            </form>
          </div>
        </div>
      </div>
      
      <div class="max-w-7xl mx-auto px-4 py-6">
        {/* Error State */}
        <Show when={error()}>
          <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6">
            <div class="flex items-center gap-2 text-red-400">
              <AlertTriangle class="h-5 w-5" />
              <span>{error()}</span>
            </div>
          </div>
        </Show>
        
        {/* Loading State */}
        <Show when={loading() && !analysis()}>
          <div class="flex flex-col items-center justify-center py-24">
            <div class="relative">
              <RefreshCw class="h-12 w-12 text-accent-400 animate-spin" />
              <Sparkles class="h-5 w-5 text-purple-400 absolute -top-1 -right-1 animate-pulse" />
            </div>
            <p class="text-slate-300 mt-4 font-medium">Analyzing {searchSymbol()}...</p>
            <p class="text-slate-500 text-sm mt-1">Running comprehensive AI analysis</p>
          </div>
        </Show>
        
        {/* Empty State */}
        <Show when={!loading() && !analysis() && !error()}>
          <div class="flex flex-col items-center justify-center py-24 text-center">
            <div class="w-24 h-24 bg-gradient-to-br from-accent-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mb-6 border border-accent-500/30">
              <LineChart class="h-12 w-12 text-accent-400" />
            </div>
            <h3 class="text-xl font-semibold text-slate-200 mb-2">Enter a Stock Symbol</h3>
            <p class="text-slate-500 max-w-md mb-6">
              Get comprehensive AI-powered analysis including technical indicators, fundamentals,
              sentiment, and ML model predictions.
            </p>
            <div class="flex flex-wrap gap-2 justify-center">
              {['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN'].map(sym => (
                <button
                  onClick={() => { setSearchSymbol(sym); fetchAnalysis(sym); }}
                  class="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors border border-slate-700 hover:border-accent-500/50"
                >
                  {sym}
                </button>
              ))}
            </div>
          </div>
        </Show>
        
        {/* Analysis Results */}
        <Show when={analysis()}>
          {(data) => (
            <div class="space-y-6">
              {/* Top Section: Symbol Info + Score + ML Prediction */}
              <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Symbol Info */}
                <div class="lg:col-span-5 bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                  <div class="flex items-start justify-between mb-4">
                    <div>
                      <div class="flex items-center gap-3 mb-2">
                        <h2 class="text-3xl font-bold">{data().symbol}</h2>
                        <RatingBadge rating={data().rating} />
                      </div>
                      <div class="flex items-baseline gap-3">
                        <span class="text-4xl font-light">{formatCurrency(data().price)}</span>
                        <span class={`text-xl font-medium ${(data().change ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(data().change ?? 0) >= 0 ? '+' : ''}{formatCurrency(data().change ?? 0)}
                          <span class="text-sm ml-1">({(data().change_percent ?? 0) >= 0 ? '+' : ''}{(data().change_percent ?? 0).toFixed(2)}%)</span>
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Quick Stats */}
                  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t border-slate-700/50">
                    <div class="text-center p-2 bg-slate-900/50 rounded-lg">
                      <div class="text-xs text-slate-500">Confidence</div>
                      <div class="text-base font-semibold text-white">{((data().confidence ?? 0.75) * 100).toFixed(0)}%</div>
                    </div>
                    <div class="text-center p-2 bg-slate-900/50 rounded-lg">
                      <div class="text-xs text-slate-500">Risk Level</div>
                      <div class={`text-base font-semibold ${
                        data().risk?.risk_level === 'low' ? 'text-emerald-400' :
                        data().risk?.risk_level === 'medium' ? 'text-yellow-400' :
                        'text-red-400'
                      }`}>
                        {(data().risk?.risk_level ?? 'medium').toUpperCase()}
                      </div>
                    </div>
                    <div class="text-center p-2 bg-slate-900/50 rounded-lg">
                      <div class="text-xs text-slate-500">Volatility</div>
                      <div class="text-base font-semibold text-white">{data().risk?.volatility_30d?.toFixed(1) ?? 'N/A'}%</div>
                    </div>
                    <div class="text-center p-2 bg-slate-900/50 rounded-lg">
                      <div class="text-xs text-slate-500">Updated</div>
                      <div class="text-base font-semibold text-white">{lastUpdated()?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                    </div>
                  </div>
                </div>
                
                {/* Overall Score Gauge */}
                <div class="lg:col-span-3 bg-slate-800/50 border border-slate-700 rounded-xl p-6 flex flex-col items-center justify-center">
                  <ScoreGauge score={data().overall_score} label="Overall Score" size="lg" />
                  <div class="mt-3 flex gap-1.5 text-xs flex-wrap justify-center">
                    <span class="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">Tech 35%</span>
                    <span class="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">Fund 25%</span>
                    <span class="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded">Sent 20%</span>
                    <span class="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">Mom 20%</span>
                  </div>
                </div>
                
                {/* ML Prediction */}
                <div class="lg:col-span-4">
                  <MLPredictionCard prediction={prediction()} loading={predictionLoading()} />
                </div>
              </div>

              {/* Portfolio Context Card */}
              <PortfolioContextCard 
                position={position()} 
                analysis={data()} 
                prediction={prediction()} 
              />
              
              {/* Factor Analysis Sections (Expandable) */}
              <div class="space-y-4">
                <h3 class="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <PieChart class="h-4 w-4" />
                  Detailed Analysis
                  <span class="text-xs font-normal text-slate-500">(Click to expand)</span>
                </h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ExpandableAnalysisSection
                    type="technical"
                    score={data().technical?.score ?? 50}
                    signal={data().technical?.signal ?? 'NEUTRAL'}
                    data={data().technical}
                    isExpanded={expandedSections().has('technical')}
                    onToggle={() => toggleSection('technical')}
                    symbol={data().symbol}
                  />
                  
                  <ExpandableAnalysisSection
                    type="fundamental"
                    score={data().fundamental?.score ?? 50}
                    signal={data().fundamental?.signal ?? 'NEUTRAL'}
                    data={data().fundamental}
                    isExpanded={expandedSections().has('fundamental')}
                    onToggle={() => toggleSection('fundamental')}
                    symbol={data().symbol}
                  />
                  
                  <ExpandableAnalysisSection
                    type="sentiment"
                    score={data().sentiment?.score ?? 50}
                    signal={data().sentiment?.signal ?? 'NEUTRAL'}
                    data={data().sentiment}
                    isExpanded={expandedSections().has('sentiment')}
                    onToggle={() => toggleSection('sentiment')}
                    symbol={data().symbol}
                  />
                  
                  <ExpandableAnalysisSection
                    type="momentum"
                    score={data().momentum?.score ?? 50}
                    signal={data().momentum?.signal ?? 'NEUTRAL'}
                    data={data().momentum}
                    isExpanded={expandedSections().has('momentum')}
                    onToggle={() => toggleSection('momentum')}
                    symbol={data().symbol}
                  />
                  
                  <ExpandableAnalysisSection
                    type="risk"
                    score={100 - (data().risk?.score ?? 50)}
                    signal={data().risk?.risk_level === 'low' ? 'BULLISH' : data().risk?.risk_level === 'high' ? 'BEARISH' : 'NEUTRAL'}
                    data={data().risk}
                    isExpanded={expandedSections().has('risk')}
                    onToggle={() => toggleSection('risk')}
                    symbol={data().symbol}
                  />
                </div>
              </div>
              
              {/* Trade Setup (if available) */}
              <Show when={data().trade_suggestion?.stop_loss}>
                <div class="bg-gradient-to-r from-slate-800/80 to-slate-800/40 border border-accent-500/30 rounded-xl p-6">
                  <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-2">
                      <Target class="h-5 w-5 text-accent-400" />
                      <h3 class="text-lg font-semibold">Trade Setup</h3>
                      <span class={`px-2 py-0.5 rounded text-xs font-bold ${
                        data().trade_suggestion?.action?.includes('Buy') || data().rating === 'BUY' || data().rating === 'STRONG_BUY'
                          ? 'bg-emerald-500/20 text-emerald-400' 
                          : data().trade_suggestion?.action?.includes('Sell') || data().rating === 'SELL' || data().rating === 'STRONG_SELL'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {data().trade_suggestion?.action || data().rating?.replace('_', ' ')}
                      </span>
                    </div>
                    <span class="text-xs text-slate-500">Educational purposes only</span>
                  </div>
                  
                  <Show when={data().trade_suggestion?.entry_zone_low && data().trade_suggestion?.stop_loss}>
                    <TradeSetupVisualizer 
                      currentPrice={data().price}
                      entryLow={data().trade_suggestion!.entry_zone_low!}
                      entryHigh={data().trade_suggestion!.entry_zone_high || data().trade_suggestion!.entry_zone_low!}
                      stopLoss={data().trade_suggestion!.stop_loss!}
                      target1={data().trade_suggestion!.target_1 || data().price * 1.05}
                      target2={data().trade_suggestion!.target_2}
                    />
                  </Show>
                  
                  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-4 border-t border-slate-700/50">
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Entry Zone</div>
                      <div class="text-base font-medium text-accent-400">
                        {formatCurrency(data().trade_suggestion?.entry_zone_low ?? data().price * 0.98)} - {formatCurrency(data().trade_suggestion?.entry_zone_high ?? data().price)}
                      </div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Stop Loss</div>
                      <div class="text-base font-medium text-red-400">{formatCurrency(data().trade_suggestion?.stop_loss ?? 0)}</div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Target 1</div>
                      <div class="text-base font-medium text-emerald-400">{formatCurrency(data().trade_suggestion?.target_1 ?? 0)}</div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Risk/Reward</div>
                      <div class={`text-base font-bold ${(data().trade_suggestion?.risk_reward ?? 0) >= 2 ? 'text-emerald-400' : 'text-yellow-400'}`}>
                        1:{data().trade_suggestion?.risk_reward?.toFixed(1) ?? 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              </Show>
              
              {/* Bullish/Bearish/Risks Grid */}
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                  <div class="flex items-center gap-2 mb-3">
                    <TrendingUp class="h-4 w-4 text-emerald-400" />
                    <h4 class="font-medium text-emerald-400">Bullish Factors</h4>
                  </div>
                  <ul class="space-y-2">
                    <For each={data().bullish_factors}>
                      {(factor) => (
                        <li class="flex items-start gap-2 text-sm">
                          <ChevronRight class="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                          <span class="text-slate-300">{factor}</span>
                        </li>
                      )}
                    </For>
                    <Show when={!data().bullish_factors?.length}>
                      <li class="text-slate-500 text-sm">No bullish factors identified</li>
                    </Show>
                  </ul>
                </div>
                
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                  <div class="flex items-center gap-2 mb-3">
                    <TrendingDown class="h-4 w-4 text-red-400" />
                    <h4 class="font-medium text-red-400">Bearish Factors</h4>
                  </div>
                  <ul class="space-y-2">
                    <For each={data().bearish_factors}>
                      {(factor) => (
                        <li class="flex items-start gap-2 text-sm">
                          <ChevronRight class="h-4 w-4 text-red-400 mt-0.5 flex-shrink-0" />
                          <span class="text-slate-300">{factor}</span>
                        </li>
                      )}
                    </For>
                    <Show when={!data().bearish_factors?.length}>
                      <li class="text-slate-500 text-sm">No bearish factors identified</li>
                    </Show>
                  </ul>
                </div>
                
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                  <div class="flex items-center gap-2 mb-3">
                    <AlertTriangle class="h-4 w-4 text-yellow-400" />
                    <h4 class="font-medium text-yellow-400">Key Risks</h4>
                  </div>
                  <ul class="space-y-2">
                    <For each={data().key_risks}>
                      {(risk) => (
                        <li class="flex items-start gap-2 text-sm">
                          <AlertTriangle class="h-4 w-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                          <span class="text-slate-300">{risk}</span>
                        </li>
                      )}
                    </For>
                    <Show when={!data().key_risks?.length}>
                      <li class="text-slate-500 text-sm">No significant risks identified</li>
                    </Show>
                  </ul>
                </div>
              </div>
              
              {/* Recent News */}
              <Show when={news().length > 0}>
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-5">
                  <div class="flex items-center gap-2 mb-4">
                    <Newspaper class="h-5 w-5 text-accent-400" />
                    <h3 class="text-base font-bold text-white">Recent News</h3>
                  </div>
                  <div class="space-y-3">
                    <For each={news().slice(0, 4)}>
                      {(article) => (
                        <a 
                          href={article.url} 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          class="block p-3 bg-slate-900/50 rounded-lg border border-slate-700/50 hover:border-accent-500/50 transition-colors"
                        >
                          <div class="flex items-center gap-2 mb-1">
                            <span class="text-xs font-medium text-accent-400">{article.source}</span>
                            <span class="text-xs text-slate-500">•</span>
                            <span class="text-xs text-slate-500">{new Date(article.published_at).toLocaleDateString()}</span>
                          </div>
                          <div class="text-sm font-medium text-white line-clamp-1">{article.title || article.headline}</div>
                        </a>
                      )}
                    </For>
                  </div>
                </div>
              </Show>

              {/* Disclaimer */}
              <div class="bg-slate-800/30 border border-slate-700/50 rounded-lg p-4 text-center">
                <p class="text-xs text-slate-500">
                  <Info class="inline h-3 w-3 mr-1" />
                  This analysis is for educational purposes only and does not constitute financial advice.
                  Always do your own research and consult with a licensed financial advisor.
                </p>
              </div>
            </div>
          )}
        </Show>
      </div>
    </div>
  );
}
