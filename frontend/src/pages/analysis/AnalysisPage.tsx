/**
 * CIFT Markets - Professional Stock Analysis Page
 * 
 * Beautiful, comprehensive stock analysis interface with:
 * - Overall Score Gauge with color-coded sentiment
 * - Technical/Fundamental/Sentiment/Momentum breakdowns
 * - AI-powered insights with reasoning
 * - Trade suggestion cards with risk/reward
 * - Interactive factor weight visualization
 * - Real-time price and chart integration
 * 
 * ALL DATA FROM BACKEND /api/v1/analysis/{symbol}
 * 
 * UI Inspired by:
 * - Bloomberg Terminal analysis views
 * - TradingView technicals overlay
 * - Refinitiv StarMine scoring
 */

import { createSignal, createEffect, Show, For, onMount, onCleanup } from 'solid-js';
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
  Layers,
  ArrowUp,
  ArrowDown,
  Minus,
  ExternalLink,
  Copy,
  Star,
  Sparkles,
  Briefcase,
  Wallet,
} from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { formatCurrency, formatPercent, formatNumber } from '~/lib/utils/format';
import { apiClient, Position } from '~/lib/api/client';
import { Card, CardHeader, CardTitle, CardContent } from '~/components/ui/Card';
import { Input } from '~/components/ui/Input';
import { Button } from '~/components/ui/Button';
import { TradeSetupVisualizer } from '~/components/analysis/TradeSetupVisualizer';
import { AnalysisDetailModal } from '~/components/analysis/AnalysisDetailModal';

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/** Safely format a number to fixed decimal places, returns 'N/A' if null/undefined */
const safeToFixed = (value: number | null | undefined, decimals: number = 2, fallback: string = 'N/A'): string => {
  if (value == null || isNaN(value)) return fallback;
  return value.toFixed(decimals);
};

/** Safely format a percent value */
const safePercent = (value: number | null | undefined, decimals: number = 2): string => {
  if (value == null || isNaN(value)) return 'N/A';
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
};

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
}

interface FundamentalAnalysis {
  score: number;
  signal: string;
  pe_ratio?: number;
  pb_ratio?: number;
  ps_ratio?: number;
  peg_ratio?: number;
  roe?: number;
  roa?: number;
  profit_margin?: number;
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
  momentum_12_1?: number;
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
  
  suggested_action?: string;
  entry_zone?: [number, number];
  stop_loss?: number;
  target_1?: number;
  target_2?: number;
  risk_reward?: number;
  
  analysis_latency_ms?: number;
}

// ============================================================================
// COMPONENT HELPERS
// ============================================================================

const ScoreGauge = (props: { score: number; label: string; size?: 'sm' | 'md' | 'lg' }) => {
  const size = props.size || 'md';
  const sizes = {
    sm: { outer: 80, stroke: 8, fontSize: 'text-lg', labelSize: 'text-xs' },
    md: { outer: 120, stroke: 12, fontSize: 'text-2xl', labelSize: 'text-sm' },
    lg: { outer: 180, stroke: 16, fontSize: 'text-4xl', labelSize: 'text-base' },
  };
  
  const config = sizes[size];
  const radius = (config.outer - config.stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (props.score / 100) * circumference;
  
  // Color based on score
  const getColor = (score: number) => {
    if (score >= 70) return { stroke: '#10b981', bg: 'bg-emerald-500/10', text: 'text-emerald-400' };
    if (score >= 55) return { stroke: '#22c55e', bg: 'bg-green-500/10', text: 'text-green-400' };
    if (score >= 45) return { stroke: '#eab308', bg: 'bg-yellow-500/10', text: 'text-yellow-400' };
    if (score >= 30) return { stroke: '#f97316', bg: 'bg-orange-500/10', text: 'text-orange-400' };
    return { stroke: '#ef4444', bg: 'bg-red-500/10', text: 'text-red-400' };
  };
  
  const colors = getColor(props.score);
  
  return (
    <div class="flex flex-col items-center">
      <div class="relative" style={{ width: `${config.outer}px`, height: `${config.outer}px` }}>
        <svg class="transform -rotate-90" viewBox={`0 0 ${config.outer} ${config.outer}`}>
          {/* Background circle */}
          <circle
            cx={config.outer / 2}
            cy={config.outer / 2}
            r={radius}
            stroke="currentColor"
            stroke-width={config.stroke}
            fill="none"
            class="text-slate-700"
          />
          {/* Progress circle */}
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
        {/* Center text */}
        <div class="absolute inset-0 flex flex-col items-center justify-center">
          <span class={`${config.fontSize} font-bold ${colors.text}`}>{Math.round(props.score)}</span>
        </div>
      </div>
      <span class={`${config.labelSize} text-slate-400 mt-1`}>{props.label}</span>
    </div>
  );
};

const SignalBadge = (props: { signal: string }) => {
  const configs: Record<string, { bg: string; text: string; icon: any }> = {
    'VERY_BULLISH': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', icon: TrendingUp },
    'BULLISH': { bg: 'bg-green-500/20', text: 'text-green-400', icon: ArrowUp },
    'NEUTRAL': { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: Minus },
    'BEARISH': { bg: 'bg-orange-500/20', text: 'text-orange-400', icon: ArrowDown },
    'VERY_BEARISH': { bg: 'bg-red-500/20', text: 'text-red-400', icon: TrendingDown },
  };
  
  const config = configs[props.signal] || configs['NEUTRAL'];
  const Icon = config.icon;
  
  return (
    <span class={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text}`}>
      <Icon class="h-3 w-3" />
      {props.signal.replace('_', ' ')}
    </span>
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

const FactorBar = (props: { label: string; value: number | null | undefined; max?: number; format?: 'percent' | 'number' | 'currency' }) => {
  const value = props.value ?? 0;
  const max = props.max || 100;
  const pct = Math.min(100, (Math.abs(value) / max) * 100);
  const isPositive = value >= 0;
  
  const formatValue = () => {
    if (props.value == null) return 'N/A';
    if (props.format === 'percent') return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    if (props.format === 'currency') return formatCurrency(value);
    return value.toFixed(2);
  };
  
  return (
    <div class="mb-2">
      <div class="flex justify-between text-xs mb-1">
        <span class="text-slate-400">{props.label}</span>
        <span class={isPositive ? 'text-green-400' : 'text-red-400'}>{formatValue()}</span>
      </div>
      <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          class={`h-full rounded-full transition-all duration-500 ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
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
  const [news, setNews] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [lastUpdated, setLastUpdated] = createSignal<Date | null>(null);
  const [expandedSections, setExpandedSections] = createSignal<Set<string>>(new Set(['technical', 'fundamental', 'sentiment', 'momentum']));
  const [activeModal, setActiveModal] = createSignal<{ type: 'technical' | 'fundamental' | 'sentiment' | 'risk', data: any } | null>(null);
  
  // Auto-refresh interval
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
    
    setLoading(true);
    setError(null);
    
    try {
      const [data, posData, newsData] = await Promise.all([
        apiClient.get(`/analysis/${symbol.toUpperCase().trim()}`),
        apiClient.getPosition(symbol.toUpperCase().trim()).catch(() => null),
        apiClient.getNews({ symbol: symbol.toUpperCase().trim(), limit: 5 }).catch(() => [])
      ]);
      
      // Validate response has required fields
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid response from server');
      }
      
      if (!data.symbol) {
        throw new Error('Invalid analysis data received');
      }
      
      setAnalysis(data);
      setPosition(posData);
      setNews(Array.isArray(newsData) ? newsData : []);
      setLastUpdated(new Date());
      
      // Update URL if needed
      if (params.symbol !== symbol.toUpperCase()) {
        navigate(`/analysis/${symbol.toUpperCase()}`, { replace: true });
      }
    } catch (err: any) {
      console.error('Analysis fetch error:', err);
      
      // Better error messages based on error type
      if (err.message?.includes('timeout') || err.message?.includes('Network')) {
        setError('Request timed out. The AI analysis may take longer - please try again.');
      } else if (err.status === 404) {
        setError(`Symbol "${symbol.toUpperCase()}" not found. Please check the symbol and try again.`);
      } else if (err.status === 429) {
        setError('Too many requests. Please wait a moment and try again.');
      } else if (err.status >= 500) {
        setError('Server error. Our team has been notified. Please try again later.');
      } else {
        setError(err.message || 'Failed to fetch analysis. Please try again.');
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
  
  // Initial load
  onMount(() => {
    if (params.symbol) {
      setSearchSymbol(params.symbol);
      fetchAnalysis(params.symbol);
    }
    
    // Auto-refresh every 60 seconds if we have a symbol
    refreshInterval = window.setInterval(() => {
      const current = analysis();
      if (current) {
        fetchAnalysis(current.symbol);
      }
    }, 60000);
  });
  
  onCleanup(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
  
  // Effect to handle URL changes
  createEffect(() => {
    if (params.symbol && params.symbol !== analysis()?.symbol) {
      setSearchSymbol(params.symbol);
      fetchAnalysis(params.symbol);
    }
  });
  
  return (
    <div class="min-h-screen bg-slate-900 text-white p-4 md:p-6">
      {/* Header */}
      <div class="mb-6">
        <h1 class="text-2xl font-bold mb-2 flex items-center gap-2">
          <AIIcon size={24} class="text-accent-400" />
          Stock Analysis
          <span class="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-accent-500/20 text-accent-400 text-xs font-medium">
            <Sparkles class="h-3 w-3" />
            AI-Powered
          </span>
        </h1>
        <p class="text-slate-400 text-sm">Comprehensive technical, fundamental, sentiment & momentum analysis</p>
      </div>
      
      {/* Search Bar */}
      <form onSubmit={handleSearch} class="mb-6">
        <div class="flex gap-2 max-w-xl">
          <div class="relative flex-1">
            <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              type="text"
              value={searchSymbol()}
              onInput={(e) => setSearchSymbol(e.currentTarget.value.toUpperCase())}
              placeholder="Enter symbol (e.g., AAPL, MSFT, TSLA)"
              class="w-full pl-10 pr-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent"
            />
          </div>
          <button
            type="submit"
            disabled={loading() || !searchSymbol()}
            class="px-6 py-2.5 bg-accent-500 hover:bg-accent-600 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center gap-2"
          >
            {loading() ? <RefreshCw class="h-4 w-4 animate-spin" /> : <BarChart2 class="h-4 w-4" />}
            Analyze
          </button>
        </div>
      </form>
      
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
        <div class="flex flex-col items-center justify-center py-20">
          <RefreshCw class="h-12 w-12 text-accent-400 animate-spin mb-4" />
          <p class="text-slate-400">Analyzing {searchSymbol()}...</p>
          <p class="text-slate-500 text-sm mt-1">Running technical, fundamental, sentiment & momentum analysis</p>
        </div>
      </Show>
      
      {/* No Symbol State */}
      <Show when={!loading() && !analysis() && !error()}>
        <div class="flex flex-col items-center justify-center py-20 text-center">
          <div class="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mb-4">
            <BarChart2 class="h-10 w-10 text-slate-600" />
          </div>
          <h3 class="text-lg font-medium text-slate-300 mb-2">Enter a Stock Symbol</h3>
          <p class="text-slate-500 max-w-md">
            Get comprehensive AI-powered analysis including technical indicators, fundamental metrics,
            sentiment analysis, and trade suggestions.
          </p>
          <div class="flex gap-2 mt-6">
            {['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'].map(sym => (
              <button
                onClick={() => { setSearchSymbol(sym); fetchAnalysis(sym); }}
                class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
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
            {/* Top Row: Symbol Info + Overall Score */}
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Symbol Info Card */}
              <div class="lg:col-span-2 bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                <div class="flex items-start justify-between mb-4">
                  <div>
                    <div class="flex items-center gap-3 mb-2">
                      <h2 class="text-3xl font-bold">{data().symbol}</h2>
                      <RatingBadge rating={data().rating} />
                    </div>
                    <div class="flex items-baseline gap-4">
                      <span class="text-4xl font-light">{formatCurrency(data().price)}</span>
                      <span class={`text-xl font-medium ${(data().change ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(data().change ?? 0) >= 0 ? '+' : ''}{formatCurrency(data().change ?? 0)}
                        ({(data().change_percent ?? 0) >= 0 ? '+' : ''}{(data().change_percent ?? 0).toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                  <div class="text-right text-sm text-slate-400">
                    <div class="flex items-center gap-1">
                      <Clock class="h-3.5 w-3.5" />
                      {lastUpdated()?.toLocaleTimeString()}
                    </div>
                    <Show when={data().analysis_latency_ms}>
                      <div class="text-xs mt-1">
                        Latency: {data().analysis_latency_ms!.toFixed(0)}ms
                      </div>
                    </Show>
                  </div>
                </div>
                
                {/* Quick Stats */}
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-slate-700">
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Confidence</div>
                    <div class="text-lg font-semibold">{((data().confidence ?? 0.75) * 100).toFixed(0)}%</div>
                  </div>
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Risk Level</div>
                    <div class={`text-lg font-semibold ${
                      data().risk?.risk_level === 'low' ? 'text-green-400' :
                      data().risk?.risk_level === 'medium' ? 'text-yellow-400' :
                      data().risk?.risk_level === 'high' ? 'text-orange-400' : 'text-red-400'
                    }`}>
                      {(data().risk?.risk_level ?? 'medium').toUpperCase()}
                    </div>
                  </div>
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Volatility</div>
                    <div class="text-lg font-semibold">{data().risk?.volatility_30d?.toFixed(1) ?? 'N/A'}%</div>
                  </div>
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Beta</div>
                    <div class="text-lg font-semibold">{data().risk?.beta?.toFixed(2) ?? 'N/A'}</div>
                  </div>
                </div>
              </div>
              
              {/* Overall Score Gauge */}
              <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-6 flex flex-col items-center justify-center">
                <ScoreGauge score={data().overall_score} label="Overall Score" size="lg" />
                <div class="mt-4 text-center">
                  <div class="text-slate-400 text-sm mb-2">Factor Weights</div>
                  <div class="flex gap-2 text-xs flex-wrap justify-center">
                    <span class="px-2 py-1 bg-accent-500/20 text-accent-400 rounded">Tech 35%</span>
                    <span class="px-2 py-1 bg-purple-500/20 text-purple-400 rounded">Fund 25%</span>
                    <span class="px-2 py-1 bg-green-500/20 text-green-400 rounded">Sent 20%</span>
                    <span class="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded">Mom 20%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Portfolio Context (If Owned) */}
            <Show when={position()}>
              {(pos) => (
                <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                  <div class="flex items-center gap-2 mb-4">
                    <Briefcase class="h-5 w-5 text-accent-400" />
                    <h3 class="text-lg font-bold text-white">Your Position</h3>
                    <span class="px-2 py-0.5 rounded text-xs font-medium bg-accent-500/20 text-accent-400">
                      {pos().side.toUpperCase()}
                    </span>
                  </div>
                  <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Quantity</div>
                      <div class="text-lg font-mono font-semibold">{formatNumber(pos().quantity)}</div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Avg Cost</div>
                      <div class="text-lg font-mono font-semibold">{formatCurrency(pos().avg_cost)}</div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Market Value</div>
                      <div class="text-lg font-mono font-semibold">{formatCurrency(pos().market_value)}</div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Total P&L</div>
                      <div class={`text-lg font-mono font-semibold ${pos().unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatCurrency(pos().unrealized_pnl)}
                        <span class="text-sm ml-1">({formatPercent(pos().unrealized_pnl_pct)})</span>
                      </div>
                    </div>
                    <div>
                      <div class="text-slate-400 text-xs mb-1">Today's P&L</div>
                      <div class={`text-lg font-mono font-semibold ${pos().day_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatCurrency(pos().day_pnl)}
                        <span class="text-sm ml-1">({formatPercent(pos().day_pnl_pct)})</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </Show>
            
            {/* Factor Scores Row */}
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <button 
                onClick={() => setActiveModal({ type: 'technical', data: data().technical })}
                class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex items-center gap-4 hover:bg-slate-700/50 transition-colors text-left group"
              >
                <ScoreGauge score={data().technical?.score ?? 50} label="" size="sm" />
                <div>
                  <div class="text-slate-400 text-xs font-medium mb-1">Technical</div>
                  <SignalBadge signal={data().technical?.signal ?? 'NEUTRAL'} />
                  <div class="text-[10px] text-accent-400 mt-1 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    View Details <ChevronRight class="w-3 h-3" />
                  </div>
                </div>
              </button>
              
              <button 
                onClick={() => setActiveModal({ type: 'fundamental', data: data().fundamental })}
                class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex items-center gap-4 hover:bg-slate-700/50 transition-colors text-left group"
              >
                <ScoreGauge score={data().fundamental?.score ?? 50} label="" size="sm" />
                <div>
                  <div class="text-slate-400 text-xs font-medium mb-1">Fundamental</div>
                  <SignalBadge signal={data().fundamental?.signal ?? 'NEUTRAL'} />
                  <div class="text-[10px] text-accent-400 mt-1 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    View Details <ChevronRight class="w-3 h-3" />
                  </div>
                </div>
              </button>
              
              <button 
                onClick={() => setActiveModal({ type: 'sentiment', data: data().sentiment })}
                class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex items-center gap-4 hover:bg-slate-700/50 transition-colors text-left group"
              >
                <ScoreGauge score={data().sentiment?.score ?? 50} label="" size="sm" />
                <div>
                  <div class="text-slate-400 text-xs font-medium mb-1">Sentiment</div>
                  <SignalBadge signal={data().sentiment?.signal ?? 'NEUTRAL'} />
                  <div class="text-[10px] text-accent-400 mt-1 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    View Details <ChevronRight class="w-3 h-3" />
                  </div>
                </div>
              </button>
              
              <button 
                onClick={() => setActiveModal({ type: 'risk', data: data().risk })}
                class="bg-slate-800/50 border border-slate-700 rounded-xl p-4 flex items-center gap-4 hover:bg-slate-700/50 transition-colors text-left group"
              >
                <ScoreGauge score={data().momentum?.score ?? 50} label="" size="sm" />
                <div>
                  <div class="text-slate-400 text-xs font-medium mb-1">Risk & Mom.</div>
                  <SignalBadge signal={data().momentum?.signal ?? 'NEUTRAL'} />
                  <div class="text-[10px] text-accent-400 mt-1 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    View Details <ChevronRight class="w-3 h-3" />
                  </div>
                </div>
              </button>
            </div>
            
            {/* Trade Suggestion Card */}
            <Show when={data().suggested_action}>
              <div class="bg-gradient-to-r from-slate-800 to-slate-800/50 border border-accent-500/30 rounded-xl p-6">
                <div class="flex items-center justify-between mb-4">
                  <div class="flex items-center gap-2">
                    <Target class="h-5 w-5 text-accent-400" />
                    <h3 class="text-lg font-semibold">Trade Setup</h3>
                    <span class={`px-2 py-0.5 rounded text-xs font-bold ${
                      data().suggested_action === 'BUY' ? 'bg-green-500/20 text-green-400' :
                      data().suggested_action === 'SELL' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {data().suggested_action}
                    </span>
                  </div>
                  <span class="text-xs text-slate-500">Educational purposes only</span>
                </div>
                
                <Show when={data().entry_zone && data().stop_loss && data().target_1}>
                  <TradeSetupVisualizer 
                    currentPrice={data().price}
                    entryLow={data().entry_zone![0]}
                    entryHigh={data().entry_zone![1]}
                    stopLoss={data().stop_loss!}
                    target1={data().target_1!}
                    target2={data().target_2}
                  />
                </Show>
                
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-4 border-t border-slate-700/50">
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Entry Zone</div>
                    <div class="text-base font-medium text-accent-400">
                      {formatCurrency(data().entry_zone?.[0] ?? 0)} - {formatCurrency(data().entry_zone?.[1] ?? 0)}
                    </div>
                  </div>
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Stop Loss</div>
                    <div class="text-base font-medium text-red-400">{formatCurrency(data().stop_loss ?? 0)}</div>
                  </div>
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Target 1</div>
                    <div class="text-base font-medium text-green-400">{formatCurrency(data().target_1 ?? 0)}</div>
                  </div>
                  <div>
                    <div class="text-slate-400 text-xs mb-1">Risk/Reward</div>
                    <div class={`text-base font-bold ${data().risk_reward! >= 2 ? 'text-green-400' : 'text-yellow-400'}`}>
                      1:{data().risk_reward?.toFixed(1) ?? 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            </Show>
            
            {/* Bullish/Bearish Factors */}
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Bullish Factors */}
              <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
                <div class="flex items-center gap-2 mb-3">
                  <TrendingUp class="h-4 w-4 text-green-400" />
                  <h4 class="font-medium text-green-400">Bullish Factors</h4>
                </div>
                <ul class="space-y-2">
                  <For each={data().bullish_factors}>
                    {(factor) => (
                      <li class="flex items-start gap-2 text-sm">
                        <ChevronRight class="h-4 w-4 text-green-400 mt-0.5 flex-shrink-0" />
                        <span class="text-slate-300">{factor}</span>
                      </li>
                    )}
                  </For>
                  <Show when={!data().bullish_factors?.length}>
                    <li class="text-slate-500 text-sm">No significant bullish factors</li>
                  </Show>
                </ul>
              </div>
              
              {/* Bearish Factors */}
              <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
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
                    <li class="text-slate-500 text-sm">No significant bearish factors</li>
                  </Show>
                </ul>
              </div>
              
              {/* Key Risks */}
              <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
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
              <div class="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                <div class="flex items-center gap-2 mb-4">
                  <Newspaper class="h-5 w-5 text-accent-400" />
                  <h3 class="text-lg font-bold text-white">Recent News</h3>
                </div>
                <div class="space-y-4">
                  <For each={news()}>
                    {(article) => (
                      <div class="flex items-start gap-4 p-3 bg-slate-900/50 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                        <div class="flex-1">
                          <div class="flex items-center gap-2 mb-1">
                            <span class="text-xs font-medium text-accent-400">{article.source}</span>
                            <span class="text-xs text-slate-500">•</span>
                            <span class="text-xs text-slate-500">{new Date(article.published_at).toLocaleDateString()}</span>
                          </div>
                          <a href={article.url} target="_blank" rel="noopener noreferrer" class="text-sm font-medium text-white hover:text-accent-400 transition-colors">
                            {article.title}
                          </a>
                          <p class="text-xs text-slate-400 mt-1 line-clamp-2">{article.summary}</p>
                        </div>
                        <Show when={article.sentiment_score}>
                          <div class={`px-2 py-1 rounded text-xs font-bold ${
                            article.sentiment_score > 0.2 ? 'bg-green-500/20 text-green-400' :
                            article.sentiment_score < -0.2 ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'
                          }`}>
                            {article.sentiment_score > 0.2 ? 'BULLISH' : article.sentiment_score < -0.2 ? 'BEARISH' : 'NEUTRAL'}
                          </div>
                        </Show>
                      </div>
                    )}
                  </For>
                </div>
              </div>
            </Show>

            {/* Disclaimer */}
            <div class="bg-slate-800/30 border border-slate-700/50 rounded-lg p-4 text-center">
              <p class="text-xs text-slate-500">
                <Info class="inline h-3 w-3 mr-1" />
                This analysis is for educational purposes only. It does not constitute financial advice.
                Always do your own research and consult with a licensed financial advisor before making investment decisions.
                Past performance is not indicative of future results.
              </p>
            </div>

            <AnalysisDetailModal 
              isOpen={!!activeModal()}
              onClose={() => setActiveModal(null)}
              type={activeModal()?.type || null}
              data={activeModal()?.data}
              symbol={data().symbol}
            />
          </div>
        )}
      </Show>
    </div>
  );
}
