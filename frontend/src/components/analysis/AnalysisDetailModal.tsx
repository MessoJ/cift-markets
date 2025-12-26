import { Show, For } from 'solid-js';
import { Modal } from '~/components/ui/Modal';
import { 
  TrendingUp, TrendingDown, Activity, BarChart2, 
  DollarSign, Newspaper, Shield, AlertTriangle,
  CheckCircle2, XCircle
} from 'lucide-solid';
import { formatCurrency, formatPercent } from '~/lib/utils/format';

// Re-defining types locally for component independence
// In a full refactor, these should move to types/analysis.ts
interface AnalysisDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: 'technical' | 'fundamental' | 'sentiment' | 'risk' | null;
  data: any; // Using any for flexibility with the complex backend response
  symbol: string;
}

export function AnalysisDetailModal(props: AnalysisDetailModalProps) {
  const getTitle = () => {
    switch (props.type) {
      case 'technical': return `Technical Analysis: ${props.symbol}`;
      case 'fundamental': return `Fundamental Data: ${props.symbol}`;
      case 'sentiment': return `Market Sentiment: ${props.symbol}`;
      case 'risk': return `Risk Assessment: ${props.symbol}`;
      default: return 'Analysis Details';
    }
  };

  const getIcon = () => {
    switch (props.type) {
      case 'technical': return <Activity class="w-6 h-6 text-blue-400" />;
      case 'fundamental': return <DollarSign class="w-6 h-6 text-emerald-400" />;
      case 'sentiment': return <Newspaper class="w-6 h-6 text-purple-400" />;
      case 'risk': return <Shield class="w-6 h-6 text-orange-400" />;
      default: return <BarChart2 class="w-6 h-6 text-slate-400" />;
    }
  };

  return (
    <Modal
      open={props.isOpen}
      onClose={props.onClose}
      title={getTitle()}
      size="lg"
      className="bg-slate-900 border-slate-800"
    >
      <div class="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
        {/* Header Summary */}
        <div class="flex items-center justify-between bg-slate-800/50 p-4 rounded-lg border border-slate-700">
          <div class="flex items-center gap-4">
            <div class="p-3 bg-slate-800 rounded-full border border-slate-700">
              {getIcon()}
            </div>
            <div>
              <div class="text-sm text-slate-400">Overall Score</div>
              <div class="text-2xl font-bold text-white">{props.data?.score ?? 0}/100</div>
            </div>
          </div>
          <div class="text-right">
            <div class="text-sm text-slate-400">Signal</div>
            <div class={`text-xl font-bold ${
              props.data?.signal?.includes('BULLISH') ? 'text-emerald-400' : 
              props.data?.signal?.includes('BEARISH') ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {props.data?.signal?.replace('_', ' ') ?? 'NEUTRAL'}
            </div>
          </div>
        </div>

        {/* Content based on type */}
        <Show when={props.type === 'technical'}>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="space-y-4">
              <h3 class="text-lg font-semibold text-white border-b border-slate-700 pb-2">Oscillators</h3>
              <MetricRow label="RSI (14)" value={props.data?.rsi_14?.toFixed(2)} status={props.data?.rsi_signal} />
              <MetricRow label="MACD" value={props.data?.macd_line?.toFixed(2)} />
              <MetricRow label="MACD Signal" value={props.data?.macd_signal_line?.toFixed(2)} />
              <MetricRow label="ATR %" value={formatPercent(props.data?.atr_percent)} />
            </div>
            <div class="space-y-4">
              <h3 class="text-lg font-semibold text-white border-b border-slate-700 pb-2">Moving Averages</h3>
              <MetricRow label="SMA 20" value={formatCurrency(props.data?.sma_20)} />
              <MetricRow label="SMA 50" value={formatCurrency(props.data?.sma_50)} />
              <MetricRow label="SMA 200" value={formatCurrency(props.data?.sma_200)} />
              <MetricRow label="Volume vs Avg" value={`${props.data?.volume_vs_avg?.toFixed(1)}x`} />
            </div>
            <div class="col-span-1 md:col-span-2">
              <h3 class="text-lg font-semibold text-white border-b border-slate-700 pb-2 mb-4">Trend Analysis</h3>
              <div class="grid grid-cols-3 gap-4">
                <TrendBox label="Short Term" trend={props.data?.short_term_trend} />
                <TrendBox label="Medium Term" trend={props.data?.medium_term_trend} />
                <TrendBox label="Long Term" trend={props.data?.long_term_trend} />
              </div>
            </div>
          </div>
        </Show>

        <Show when={props.type === 'fundamental'}>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="space-y-4">
              <h3 class="text-lg font-semibold text-white border-b border-slate-700 pb-2">Valuation</h3>
              <MetricRow label="P/E Ratio" value={props.data?.pe_ratio?.toFixed(2)} />
              <MetricRow label="P/B Ratio" value={props.data?.pb_ratio?.toFixed(2)} />
              <MetricRow label="P/S Ratio" value={props.data?.ps_ratio?.toFixed(2)} />
              <MetricRow label="PEG Ratio" value={props.data?.peg_ratio?.toFixed(2)} />
            </div>
            <div class="space-y-4">
              <h3 class="text-lg font-semibold text-white border-b border-slate-700 pb-2">Profitability & Growth</h3>
              <MetricRow label="ROE" value={formatPercent(props.data?.roe)} />
              <MetricRow label="Profit Margin" value={formatPercent(props.data?.profit_margin)} />
              <MetricRow label="Rev Growth (YoY)" value={formatPercent(props.data?.revenue_growth_yoy)} />
              <MetricRow label="EPS Growth (YoY)" value={formatPercent(props.data?.earnings_growth_yoy)} />
            </div>
          </div>
        </Show>

        <Show when={props.type === 'sentiment'}>
          <div class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="bg-slate-800/30 p-4 rounded-lg border border-slate-700">
                <h3 class="text-sm text-slate-400 mb-2">News Sentiment</h3>
                <div class="text-3xl font-bold text-white mb-1">
                  {(props.data?.news_sentiment * 100).toFixed(0)}/100
                </div>
                <div class="text-sm text-slate-500">
                  Based on {props.data?.news_volume} recent articles
                </div>
              </div>
              <div class="bg-slate-800/30 p-4 rounded-lg border border-slate-700">
                <h3 class="text-sm text-slate-400 mb-2">Analyst Consensus</h3>
                <div class="text-3xl font-bold text-white mb-1">
                  {props.data?.analyst_rating?.replace('_', ' ')}
                </div>
                <div class="text-sm text-slate-500">
                  Target: {formatCurrency(props.data?.analyst_target)} ({formatPercent(props.data?.analyst_target_upside)})
                </div>
              </div>
            </div>
          </div>
        </Show>

        <Show when={props.type === 'risk'}>
          <div class="space-y-6">
            <div class="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
              <div class="flex items-center gap-3 mb-2">
                <AlertTriangle class="w-5 h-5 text-red-400" />
                <h3 class="font-semibold text-red-400">Risk Assessment</h3>
              </div>
              <p class="text-slate-300">
                Risk Level: <span class="font-bold text-white">{props.data?.risk_level}</span>
              </p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MetricRow label="Volatility (30d)" value={formatPercent(props.data?.volatility_30d)} />
              <MetricRow label="Beta" value={props.data?.beta?.toFixed(2)} />
              <MetricRow label="Max Drawdown (1y)" value={formatPercent(props.data?.max_drawdown_1y)} />
              <MetricRow label="VaR (95%)" value={formatPercent(props.data?.var_95)} />
            </div>
          </div>
        </Show>
      </div>
    </Modal>
  );
}

// Helper Components
function MetricRow(props: { label: string; value: string | undefined; status?: string }) {
  return (
    <div class="flex justify-between items-center py-2 border-b border-slate-800/50 last:border-0">
      <span class="text-slate-400">{props.label}</span>
      <div class="flex items-center gap-2">
        <span class="font-mono text-white">{props.value ?? 'N/A'}</span>
        <Show when={props.status}>
          <span class={`text-xs px-1.5 py-0.5 rounded ${
            props.status === 'BULLISH' ? 'bg-emerald-500/20 text-emerald-400' :
            props.status === 'BEARISH' ? 'bg-red-500/20 text-red-400' :
            'bg-slate-700 text-slate-300'
          }`}>
            {props.status}
          </span>
        </Show>
      </div>
    </div>
  );
}

function TrendBox(props: { label: string; trend: string | undefined }) {
  const isBullish = props.trend?.includes('UP') || props.trend?.includes('BULLISH');
  const isBearish = props.trend?.includes('DOWN') || props.trend?.includes('BEARISH');
  
  return (
    <div class={`p-3 rounded-lg border ${
      isBullish ? 'bg-emerald-500/10 border-emerald-500/20' :
      isBearish ? 'bg-red-500/10 border-red-500/20' :
      'bg-slate-800 border-slate-700'
    }`}>
      <div class="text-xs text-slate-400 mb-1">{props.label}</div>
      <div class={`font-bold flex items-center gap-1 ${
        isBullish ? 'text-emerald-400' :
        isBearish ? 'text-red-400' :
        'text-slate-300'
      }`}>
        <Show when={isBullish}><TrendingUp class="w-4 h-4" /></Show>
        <Show when={isBearish}><TrendingDown class="w-4 h-4" /></Show>
        {props.trend?.replace('_', ' ') ?? 'NEUTRAL'}
      </div>
    </div>
  );
}
