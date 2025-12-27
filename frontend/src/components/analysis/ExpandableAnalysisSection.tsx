/**
 * CIFT Markets - Expandable Analysis Section Component
 * 
 * Replaces modal-based "View More" with smooth inline expansion.
 * Maintains brand consistency with gradient borders and glass effects.
 */

import { Show, For, createSignal, createMemo } from 'solid-js';
import { 
  ChevronDown, ChevronUp, Activity, DollarSign, 
  Newspaper, Shield, TrendingUp, TrendingDown,
  AlertTriangle, CheckCircle2, MinusCircle
} from 'lucide-solid';
import { formatCurrency, formatPercent } from '~/lib/utils/format';

// ============================================================================
// HELPER COMPONENTS
// ============================================================================

const MetricRow = (props: { 
  label: string; 
  value: string | number | null | undefined;
  status?: 'bullish' | 'bearish' | 'neutral' | null;
  format?: 'currency' | 'percent' | 'number';
  tooltip?: string;
}) => {
  const formattedValue = () => {
    if (props.value == null) return 'N/A';
    if (typeof props.value === 'string') return props.value;
    if (props.format === 'currency') return formatCurrency(props.value);
    if (props.format === 'percent') return `${props.value >= 0 ? '+' : ''}${props.value.toFixed(2)}%`;
    return props.value.toFixed(2);
  };

  const statusColor = () => {
    if (props.status === 'bullish') return 'text-emerald-400';
    if (props.status === 'bearish') return 'text-red-400';
    return 'text-slate-300';
  };

  return (
    <div class="flex justify-between items-center py-1 border-b border-slate-800 last:border-0">
      <span class="text-xs text-slate-400">{props.label}</span>
      <span class={`text-xs font-medium ${statusColor()}`}>{formattedValue()}</span>
    </div>
  );
};

const TrendIndicator = (props: { label: string; trend?: string | null }) => {
  const trendConfig = () => {
    switch (props.trend?.toLowerCase()) {
      case 'up':
      case 'uptrend':
      case 'bullish':
        return { icon: TrendingUp, color: 'text-emerald-400 bg-emerald-500/10', text: 'Uptrend' };
      case 'down':
      case 'downtrend':
      case 'bearish':
        return { icon: TrendingDown, color: 'text-red-400 bg-red-500/10', text: 'Downtrend' };
      default:
        return { icon: MinusCircle, color: 'text-slate-400 bg-slate-500/10', text: 'Sideways' };
    }
  };

  const config = trendConfig();
  const Icon = config.icon;

  return (
    <div class="text-center p-2 rounded-lg bg-slate-900 border border-slate-800">
      <div class="text-[10px] text-slate-500 mb-1">{props.label}</div>
      <div class={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium ${config.color}`}>
        <Icon class="w-3 h-3" />
        {config.text}
      </div>
    </div>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

interface ExpandableAnalysisSectionProps {
  type: 'technical' | 'fundamental' | 'sentiment' | 'risk' | 'momentum';
  score: number;
  signal: string;
  data: any;
  isExpanded: boolean;
  onToggle: () => void;
  symbol: string;
}

export function ExpandableAnalysisSection(props: ExpandableAnalysisSectionProps) {
  const config = createMemo(() => {
    switch (props.type) {
      case 'technical':
        return {
          title: 'Technical Analysis',
          icon: Activity,
          gradient: 'from-blue-500/20 to-cyan-500/20',
          border: 'border-blue-500/30',
          iconColor: 'text-blue-400',
        };
      case 'fundamental':
        return {
          title: 'Fundamental Analysis',
          icon: DollarSign,
          gradient: 'from-emerald-500/20 to-green-500/20',
          border: 'border-emerald-500/30',
          iconColor: 'text-emerald-400',
        };
      case 'sentiment':
        return {
          title: 'Market Sentiment',
          icon: Newspaper,
          gradient: 'from-purple-500/20 to-pink-500/20',
          border: 'border-purple-500/30',
          iconColor: 'text-purple-400',
        };
      case 'risk':
        return {
          title: 'Risk Assessment',
          icon: Shield,
          gradient: 'from-orange-500/20 to-amber-500/20',
          border: 'border-orange-500/30',
          iconColor: 'text-orange-400',
        };
      case 'momentum':
        return {
          title: 'Momentum Indicators',
          icon: TrendingUp,
          gradient: 'from-yellow-500/20 to-lime-500/20',
          border: 'border-yellow-500/30',
          iconColor: 'text-yellow-400',
        };
    }
  });

  const scoreColor = () => {
    if (props.score >= 70) return 'text-emerald-400';
    if (props.score >= 55) return 'text-green-400';
    if (props.score >= 45) return 'text-yellow-400';
    if (props.score >= 30) return 'text-orange-400';
    return 'text-red-400';
  };

  const signalConfig = () => {
    const signal = props.signal?.toUpperCase() || 'NEUTRAL';
    if (signal.includes('BULLISH')) return { bg: 'bg-emerald-500/20', text: 'text-emerald-400' };
    if (signal.includes('BEARISH')) return { bg: 'bg-red-500/20', text: 'text-red-400' };
    return { bg: 'bg-yellow-500/20', text: 'text-yellow-400' };
  };

  const Icon = config().icon;

  return (
    <div class={`bg-slate-950 border ${props.isExpanded ? config().border : 'border-slate-800'} rounded-xl overflow-hidden transition-all duration-300 h-full`}>
      {/* Header - Vertical Layout for Grid */}
      <button
        onClick={props.onToggle}
        class={`w-full p-4 flex flex-col items-center justify-center gap-3 hover:bg-slate-900 transition-colors ${props.isExpanded ? `bg-gradient-to-b ${config().gradient}` : ''}`}
      >
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider text-center h-8 flex items-center justify-center">{config().title}</div>
        
        <div class="flex items-baseline gap-1 my-1">
          <span class={`text-3xl font-bold ${scoreColor()}`}>{Math.round(props.score)}</span>
          <span class="text-slate-600 text-[10px] font-medium">/100</span>
        </div>
        
        <div class={`px-3 py-1 rounded-md text-[10px] font-bold tracking-wide uppercase ${signalConfig().bg} ${signalConfig().text} border border-white/5`}>
          {props.signal?.replace('_', ' ') || 'NEUTRAL'}
        </div>

        <Show when={props.isExpanded}>
          <ChevronUp class="w-3 h-3 text-slate-500 mt-1" />
        </Show>
      </button>

      {/* Expanded Content */}
      <Show when={props.isExpanded}>
        <div class="px-3 pb-3 animate-in slide-in-from-top-2 duration-200 bg-slate-950">
          <div class="h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent mb-4" />
          
          {/* Technical Analysis Details */}
          <Show when={props.type === 'technical'}>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Oscillators */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Oscillators</h4>
                <MetricRow 
                  label="RSI (14)" 
                  value={props.data?.rsi_14} 
                  status={props.data?.rsi_14 < 30 ? 'bullish' : props.data?.rsi_14 > 70 ? 'bearish' : 'neutral'}
                />
                <MetricRow label="MACD Line" value={props.data?.macd_line} />
                <MetricRow label="MACD Signal" value={props.data?.macd_signal_line} />
                <MetricRow label="ATR %" value={props.data?.atr_percent} format="percent" />
                <MetricRow 
                  label="Volume vs Avg" 
                  value={props.data?.volume_vs_avg ? `${props.data.volume_vs_avg.toFixed(1)}x` : null}
                />
              </div>
              
              {/* Moving Averages */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Moving Averages</h4>
                <MetricRow label="SMA 20" value={props.data?.sma_20} format="currency" />
                <MetricRow label="SMA 50" value={props.data?.sma_50} format="currency" />
                <MetricRow label="SMA 200" value={props.data?.sma_200} format="currency" />
                <MetricRow label="Bollinger Upper" value={props.data?.bollinger_upper} format="currency" />
                <MetricRow label="Bollinger Lower" value={props.data?.bollinger_lower} format="currency" />
              </div>
              
              {/* Trend Analysis */}
              <div class="col-span-1 md:col-span-2">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Trend Analysis</h4>
                <div class="grid grid-cols-3 gap-3">
                  <TrendIndicator label="Short Term" trend={props.data?.short_term_trend} />
                  <TrendIndicator label="Medium Term" trend={props.data?.medium_term_trend} />
                  <TrendIndicator label="Long Term" trend={props.data?.long_term_trend} />
                </div>
              </div>
              
              {/* Support/Resistance */}
              <Show when={props.data?.support_levels?.length || props.data?.resistance_levels?.length}>
                <div class="col-span-1 md:col-span-2 grid grid-cols-2 gap-4 pt-3 border-t border-slate-700/50">
                  <div>
                    <h4 class="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-2">Support Levels</h4>
                    <div class="flex flex-wrap gap-2">
                      <For each={props.data?.support_levels?.slice(0, 3) || []}>
                        {(level) => (
                          <span class="px-2 py-1 bg-emerald-500/10 text-emerald-400 rounded text-sm font-mono">
                            {formatCurrency(level)}
                          </span>
                        )}
                      </For>
                    </div>
                  </div>
                  <div>
                    <h4 class="text-xs font-semibold text-red-400 uppercase tracking-wider mb-2">Resistance Levels</h4>
                    <div class="flex flex-wrap gap-2">
                      <For each={props.data?.resistance_levels?.slice(0, 3) || []}>
                        {(level) => (
                          <span class="px-2 py-1 bg-red-500/10 text-red-400 rounded text-sm font-mono">
                            {formatCurrency(level)}
                          </span>
                        )}
                      </For>
                    </div>
                  </div>
                </div>
              </Show>
            </div>
          </Show>

          {/* Fundamental Analysis Details */}
          <Show when={props.type === 'fundamental'}>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Valuation */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Valuation Metrics</h4>
                <MetricRow 
                  label="P/E Ratio" 
                  value={props.data?.pe_ratio}
                  status={props.data?.pe_ratio < 15 ? 'bullish' : props.data?.pe_ratio > 30 ? 'bearish' : 'neutral'}
                />
                <MetricRow label="P/B Ratio" value={props.data?.pb_ratio} />
                <MetricRow label="P/S Ratio" value={props.data?.ps_ratio} />
                <MetricRow label="PEG Ratio" value={props.data?.peg_ratio} />
                <MetricRow label="EV/EBITDA" value={props.data?.ev_ebitda} />
              </div>
              
              {/* Profitability */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Profitability</h4>
                <MetricRow 
                  label="ROE" 
                  value={props.data?.roe}
                  format="percent"
                  status={props.data?.roe > 15 ? 'bullish' : props.data?.roe < 5 ? 'bearish' : 'neutral'}
                />
                <MetricRow label="ROA" value={props.data?.roa} format="percent" />
                <MetricRow label="Profit Margin" value={props.data?.profit_margin} format="percent" />
                <MetricRow label="Operating Margin" value={props.data?.operating_margin} format="percent" />
              </div>
              
              {/* Growth */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Growth</h4>
                <MetricRow 
                  label="Revenue Growth (YoY)" 
                  value={props.data?.revenue_growth_yoy}
                  format="percent"
                  status={props.data?.revenue_growth_yoy > 10 ? 'bullish' : props.data?.revenue_growth_yoy < 0 ? 'bearish' : 'neutral'}
                />
                <MetricRow 
                  label="EPS Growth (YoY)" 
                  value={props.data?.earnings_growth_yoy}
                  format="percent"
                  status={props.data?.earnings_growth_yoy > 10 ? 'bullish' : props.data?.earnings_growth_yoy < 0 ? 'bearish' : 'neutral'}
                />
              </div>
              
              {/* Financial Health */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Financial Health</h4>
                <MetricRow 
                  label="Debt/Equity" 
                  value={props.data?.debt_to_equity}
                  status={props.data?.debt_to_equity < 0.5 ? 'bullish' : props.data?.debt_to_equity > 2 ? 'bearish' : 'neutral'}
                />
                <MetricRow 
                  label="Current Ratio" 
                  value={props.data?.current_ratio}
                  status={props.data?.current_ratio > 1.5 ? 'bullish' : props.data?.current_ratio < 1 ? 'bearish' : 'neutral'}
                />
                <MetricRow label="Dividend Yield" value={props.data?.dividend_yield} format="percent" />
              </div>
            </div>
          </Show>

          {/* Sentiment Analysis Details */}
          <Show when={props.type === 'sentiment'}>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* News Sentiment */}
              <div class="p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">News Sentiment</h4>
                <div class="text-center">
                  <div class={`text-4xl font-bold ${
                    (props.data?.news_sentiment ?? 0.5) > 0.6 ? 'text-emerald-400' :
                    (props.data?.news_sentiment ?? 0.5) < 0.4 ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {Math.round((props.data?.news_sentiment ?? 0.5) * 100)}
                  </div>
                  <div class="text-slate-500 text-xs mt-1">out of 100</div>
                  <div class="mt-3 text-sm text-slate-400">
                    Based on <span class="text-white font-medium">{props.data?.news_volume ?? 0}</span> recent articles
                  </div>
                </div>
              </div>
              
              {/* Analyst Ratings */}
              <div class="p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Analyst Consensus</h4>
                <Show when={props.data?.analyst_rating} fallback={
                  <div class="text-center text-slate-500 py-4">No analyst data available</div>
                }>
                  <div class="text-center">
                    <div class="text-2xl font-bold text-white">
                      {props.data?.analyst_rating?.replace('_', ' ')}
                    </div>
                    <Show when={props.data?.analyst_target}>
                      <div class="mt-2 text-sm text-slate-400">
                        Target: <span class="text-accent-400 font-medium">{formatCurrency(props.data.analyst_target)}</span>
                      </div>
                    </Show>
                    <Show when={props.data?.analyst_target_upside}>
                      <div class={`text-sm font-medium mt-1 ${props.data.analyst_target_upside > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {props.data.analyst_target_upside > 0 ? '↑' : '↓'} {Math.abs(props.data.analyst_target_upside).toFixed(1)}% upside
                      </div>
                    </Show>
                  </div>
                </Show>
              </div>
              
              {/* Social Metrics */}
              <div class="col-span-1 md:col-span-2">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Ownership & Activity</h4>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <MetricRow label="Institutional Own." value={props.data?.institutional_ownership} format="percent" />
                  <MetricRow label="Inst. Change" value={props.data?.institutional_change} format="percent" />
                  <MetricRow label="Insider Net Shares" value={props.data?.insider_net_shares} />
                  <MetricRow label="News Trend" value={props.data?.news_trend} />
                </div>
              </div>
            </div>
          </Show>

          {/* Risk Analysis Details */}
          <Show when={props.type === 'risk'}>
            <div class="space-y-6">
              {/* Risk Level Banner */}
              <div class={`p-4 rounded-lg border ${
                props.data?.risk_level === 'low' ? 'bg-emerald-500/10 border-emerald-500/30' :
                props.data?.risk_level === 'medium' ? 'bg-yellow-500/10 border-yellow-500/30' :
                props.data?.risk_level === 'high' ? 'bg-orange-500/10 border-orange-500/30' :
                'bg-red-500/10 border-red-500/30'
              }`}>
                <div class="flex items-center gap-3">
                  <AlertTriangle class={`w-6 h-6 ${
                    props.data?.risk_level === 'low' ? 'text-emerald-400' :
                    props.data?.risk_level === 'medium' ? 'text-yellow-400' :
                    props.data?.risk_level === 'high' ? 'text-orange-400' : 'text-red-400'
                  }`} />
                  <div>
                    <div class="text-sm font-medium text-white">Risk Level: {(props.data?.risk_level ?? 'Medium').toUpperCase()}</div>
                    <div class="text-xs text-slate-400">Based on volatility, drawdown, and market conditions</div>
                  </div>
                </div>
              </div>
              
              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="space-y-1">
                  <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Volatility Metrics</h4>
                  <MetricRow label="30-Day Volatility" value={props.data?.volatility_30d} format="percent" />
                  <MetricRow label="Beta" value={props.data?.beta} />
                  <MetricRow label="VaR (95%)" value={props.data?.var_95} format="percent" />
                </div>
                <div class="space-y-1">
                  <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Drawdown Analysis</h4>
                  <MetricRow 
                    label="Max Drawdown (1Y)" 
                    value={props.data?.max_drawdown_1y} 
                    format="percent"
                    status={props.data?.max_drawdown_1y < 10 ? 'bullish' : props.data?.max_drawdown_1y > 30 ? 'bearish' : 'neutral'}
                  />
                  <MetricRow 
                    label="Current Drawdown" 
                    value={props.data?.current_drawdown} 
                    format="percent"
                    status={props.data?.current_drawdown < 5 ? 'bullish' : props.data?.current_drawdown > 15 ? 'bearish' : 'neutral'}
                  />
                </div>
              </div>
            </div>
          </Show>

          {/* Momentum Analysis Details */}
          <Show when={props.type === 'momentum'}>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Returns */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Period Returns</h4>
                <MetricRow label="1 Day" value={props.data?.return_1d} format="percent" status={props.data?.return_1d > 0 ? 'bullish' : 'bearish'} />
                <MetricRow label="1 Week" value={props.data?.return_1w} format="percent" status={props.data?.return_1w > 0 ? 'bullish' : 'bearish'} />
                <MetricRow label="1 Month" value={props.data?.return_1m} format="percent" status={props.data?.return_1m > 0 ? 'bullish' : 'bearish'} />
                <MetricRow label="3 Months" value={props.data?.return_3m} format="percent" status={props.data?.return_3m > 0 ? 'bullish' : 'bearish'} />
                <MetricRow label="6 Months" value={props.data?.return_6m} format="percent" status={props.data?.return_6m > 0 ? 'bullish' : 'bearish'} />
                <MetricRow label="12 Months" value={props.data?.return_12m} format="percent" status={props.data?.return_12m > 0 ? 'bullish' : 'bearish'} />
              </div>
              
              {/* Momentum Indicators */}
              <div class="space-y-1">
                <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Momentum Indicators</h4>
                <MetricRow label="12-1 Momentum" value={props.data?.momentum_12_1} format="percent" />
                <MetricRow label="Momentum Percentile" value={props.data?.momentum_percentile} />
                <MetricRow label="Relative Strength" value={props.data?.relative_strength} />
                <MetricRow label="YTD Return" value={props.data?.return_ytd} format="percent" status={props.data?.return_ytd > 0 ? 'bullish' : 'bearish'} />
              </div>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}
