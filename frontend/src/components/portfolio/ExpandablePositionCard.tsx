/**
 * CIFT Markets - Portfolio Position Card (Expandable)
 * 
 * Enhanced portfolio position card that expands to show:
 * - ML recommendation (Buy More, Hold, Trim, Exit)
 * - Detailed reasoning from AI
 * - Technical/Fundamental/Sentiment scores
 * - Price targets and risk levels
 * - Suggested actions with percentages
 * 
 * Used on the Portfolio page to give users actionable insights.
 */

import { createSignal, Show, For } from 'solid-js';
import { 
  TrendingUp, TrendingDown, Activity, Target, AlertTriangle,
  ChevronDown, ChevronUp, Plus, Minus, DollarSign, Shield,
  Brain, Zap, BarChart2, PieChart, MessageCircle, ArrowRight,
  Info, Check, X
} from 'lucide-solid';
import { AIIcon } from '~/components/icons/AIIcon';
import { formatCurrency, formatPercent } from '~/lib/utils/format';

interface PortfolioRecommendation {
  symbol: string;
  current_price: number;
  avg_cost: number;
  quantity: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  action: string;
  confidence: number;
  target_price: number | null;
  stop_loss: number | null;
  technical_score: number;
  fundamental_score: number;
  sentiment_score: number;
  ml_prediction: string;
  summary: string;
  bullish_factors: string[];
  bearish_factors: string[];
  key_risks: string[];
  should_add: boolean;
  should_trim: boolean;
  should_hold: boolean;
  should_exit: boolean;
  suggested_add_pct: number;
  suggested_trim_pct: number;
}

interface ExpandablePositionCardProps {
  recommendation: PortfolioRecommendation;
  onTrade?: (symbol: string, action: 'buy' | 'sell') => void;
  className?: string;
}

export function ExpandablePositionCard(props: ExpandablePositionCardProps) {
  const [expanded, setExpanded] = createSignal(false);

  const rec = () => props.recommendation;

  const getActionConfig = (action: string) => {
    switch (action) {
      case 'strong_buy':
        return { 
          label: 'STRONG BUY', 
          color: 'text-success-400', 
          bg: 'bg-success-500/20',
          icon: TrendingUp,
          actionText: 'Add More',
        };
      case 'buy':
        return { 
          label: 'BUY', 
          color: 'text-success-400', 
          bg: 'bg-success-500/10',
          icon: TrendingUp,
          actionText: 'Consider Adding',
        };
      case 'sell':
        return { 
          label: 'SELL', 
          color: 'text-danger-400', 
          bg: 'bg-danger-500/10',
          icon: TrendingDown,
          actionText: 'Trim Position',
        };
      case 'strong_sell':
        return { 
          label: 'STRONG SELL', 
          color: 'text-danger-400', 
          bg: 'bg-danger-500/20',
          icon: TrendingDown,
          actionText: 'Exit Position',
        };
      default:
        return { 
          label: 'HOLD', 
          color: 'text-warning-400', 
          bg: 'bg-warning-500/10',
          icon: Activity,
          actionText: 'Maintain',
        };
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-success-400';
    if (score >= 50) return 'text-warning-400';
    return 'text-danger-400';
  };

  const getScoreBarColor = (score: number) => {
    if (score >= 70) return 'bg-success-500';
    if (score >= 50) return 'bg-warning-500';
    return 'bg-danger-500';
  };

  return (
    <div class={`bg-terminal-900 border border-terminal-700 rounded-lg overflow-hidden hover:border-terminal-600 transition-colors ${props.className || ''}`}>
      {/* Main Row - Always Visible */}
      <div 
        class="flex items-center justify-between p-4 cursor-pointer"
        onClick={() => setExpanded(!expanded())}
      >
        {/* Left: Symbol & Basic Info */}
        <div class="flex items-center gap-4">
          <div class="w-10 h-10 rounded-lg bg-terminal-800 flex items-center justify-center">
            <span class="text-sm font-bold text-white">{rec().symbol.slice(0, 2)}</span>
          </div>
          <div>
            <div class="font-semibold text-white">{rec().symbol}</div>
            <div class="text-xs text-gray-400">
              {rec().quantity.toFixed(4)} @ {formatCurrency(rec().avg_cost)}
            </div>
          </div>
        </div>

        {/* Center: P&L */}
        <div class="text-right">
          <div class={`font-mono font-semibold ${rec().unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
            {rec().unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(rec().unrealized_pnl)}
          </div>
          <div class={`text-xs font-mono ${rec().unrealized_pnl_pct >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
            {rec().unrealized_pnl_pct >= 0 ? '+' : ''}{rec().unrealized_pnl_pct.toFixed(2)}%
          </div>
        </div>

        {/* Right: Action Badge & Expand */}
        <div class="flex items-center gap-3">
          {(() => {
            const config = getActionConfig(rec().action);
            return (
              <span class={`px-2 py-1 rounded text-xs font-bold ${config.bg} ${config.color}`}>
                {config.label}
              </span>
            );
          })()}
          
          {expanded() ? (
            <ChevronUp class="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown class="w-5 h-5 text-gray-400" />
          )}
        </div>
      </div>

      {/* Expanded Details */}
      <Show when={expanded()}>
        <div class="border-t border-terminal-700 p-4 space-y-4 bg-terminal-950/50">
          {/* Action Recommendation */}
          {(() => {
            const config = getActionConfig(rec().action);
            const ActionIcon = config.icon;
            return (
              <div class={`p-3 rounded-lg ${config.bg} border border-terminal-700`}>
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <ActionIcon class={`w-5 h-5 ${config.color}`} />
                    <span class={`font-semibold ${config.color}`}>{config.actionText}</span>
                  </div>
                  <div class="text-right">
                    <span class="text-xs text-gray-400">Confidence</span>
                    <span class={`ml-2 font-mono font-bold ${config.color}`}>
                      {(rec().confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                
                {/* Suggested Action */}
                <Show when={rec().should_add}>
                  <div class="mt-2 flex items-center gap-2 text-sm text-success-400">
                    <Plus class="w-4 h-4" />
                    <span>Consider adding {rec().suggested_add_pct.toFixed(0)}% to position</span>
                  </div>
                </Show>
                <Show when={rec().should_trim}>
                  <div class="mt-2 flex items-center gap-2 text-sm text-warning-400">
                    <Minus class="w-4 h-4" />
                    <span>Consider trimming {rec().suggested_trim_pct.toFixed(0)}% of position</span>
                  </div>
                </Show>
                <Show when={rec().should_exit}>
                  <div class="mt-2 flex items-center gap-2 text-sm text-danger-400">
                    <X class="w-4 h-4" />
                    <span>Consider exiting entire position</span>
                  </div>
                </Show>
              </div>
            );
          })()}

          {/* Scores Grid */}
          <div class="grid grid-cols-3 gap-3">
            <div class="bg-terminal-800 rounded-lg p-3">
              <div class="text-[10px] text-gray-500 uppercase mb-1">Technical</div>
              <div class={`text-xl font-bold font-mono ${getScoreColor(rec().technical_score)}`}>
                {rec().technical_score.toFixed(0)}
              </div>
              <div class="mt-1 h-1 bg-terminal-700 rounded-full overflow-hidden">
                <div 
                  class={`h-full ${getScoreBarColor(rec().technical_score)} rounded-full transition-all`}
                  style={{ width: `${rec().technical_score}%` }}
                />
              </div>
            </div>
            
            <div class="bg-terminal-800 rounded-lg p-3">
              <div class="text-[10px] text-gray-500 uppercase mb-1">Fundamental</div>
              <div class={`text-xl font-bold font-mono ${getScoreColor(rec().fundamental_score)}`}>
                {rec().fundamental_score.toFixed(0)}
              </div>
              <div class="mt-1 h-1 bg-terminal-700 rounded-full overflow-hidden">
                <div 
                  class={`h-full ${getScoreBarColor(rec().fundamental_score)} rounded-full transition-all`}
                  style={{ width: `${rec().fundamental_score}%` }}
                />
              </div>
            </div>
            
            <div class="bg-terminal-800 rounded-lg p-3">
              <div class="text-[10px] text-gray-500 uppercase mb-1">Sentiment</div>
              <div class={`text-xl font-bold font-mono ${getScoreColor(rec().sentiment_score)}`}>
                {rec().sentiment_score.toFixed(0)}
              </div>
              <div class="mt-1 h-1 bg-terminal-700 rounded-full overflow-hidden">
                <div 
                  class={`h-full ${getScoreBarColor(rec().sentiment_score)} rounded-full transition-all`}
                  style={{ width: `${rec().sentiment_score}%` }}
                />
              </div>
            </div>
          </div>

          {/* Price Targets */}
          <div class="grid grid-cols-2 gap-3">
            <Show when={rec().target_price}>
              <div class="bg-terminal-800 rounded-lg p-3">
                <div class="flex items-center gap-2 text-[10px] text-gray-500 uppercase mb-1">
                  <Target class="w-3 h-3" /> Price Target
                </div>
                <div class="text-lg font-mono text-success-400">
                  {formatCurrency(rec().target_price!)}
                </div>
                <div class="text-xs text-success-400">
                  {((rec().target_price! - rec().current_price) / rec().current_price * 100).toFixed(1)}% upside
                </div>
              </div>
            </Show>
            
            <Show when={rec().stop_loss}>
              <div class="bg-terminal-800 rounded-lg p-3">
                <div class="flex items-center gap-2 text-[10px] text-gray-500 uppercase mb-1">
                  <Shield class="w-3 h-3" /> Stop Loss
                </div>
                <div class="text-lg font-mono text-danger-400">
                  {formatCurrency(rec().stop_loss!)}
                </div>
                <div class="text-xs text-danger-400">
                  {((rec().stop_loss! - rec().current_price) / rec().current_price * 100).toFixed(1)}% risk
                </div>
              </div>
            </Show>
          </div>

          {/* Bullish Factors */}
          <Show when={rec().bullish_factors.length > 0}>
            <div>
              <h4 class="text-xs text-gray-500 uppercase font-semibold mb-2 flex items-center gap-1">
                <TrendingUp class="w-3 h-3 text-success-400" /> Bullish Factors
              </h4>
              <ul class="space-y-1">
                <For each={rec().bullish_factors}>
                  {(factor) => (
                    <li class="flex items-start gap-2 text-xs text-success-400">
                      <Check class="w-3 h-3 mt-0.5 flex-shrink-0" />
                      <span>{factor}</span>
                    </li>
                  )}
                </For>
              </ul>
            </div>
          </Show>

          {/* Bearish Factors */}
          <Show when={rec().bearish_factors.length > 0}>
            <div>
              <h4 class="text-xs text-gray-500 uppercase font-semibold mb-2 flex items-center gap-1">
                <TrendingDown class="w-3 h-3 text-danger-400" /> Bearish Factors
              </h4>
              <ul class="space-y-1">
                <For each={rec().bearish_factors}>
                  {(factor) => (
                    <li class="flex items-start gap-2 text-xs text-danger-400">
                      <X class="w-3 h-3 mt-0.5 flex-shrink-0" />
                      <span>{factor}</span>
                    </li>
                  )}
                </For>
              </ul>
            </div>
          </Show>

          {/* Key Risks */}
          <Show when={rec().key_risks.length > 0}>
            <div>
              <h4 class="text-xs text-gray-500 uppercase font-semibold mb-2 flex items-center gap-1">
                <AlertTriangle class="w-3 h-3 text-warning-400" /> Key Risks
              </h4>
              <ul class="space-y-1">
                <For each={rec().key_risks}>
                  {(risk) => (
                    <li class="flex items-start gap-2 text-xs text-warning-400">
                      <Info class="w-3 h-3 mt-0.5 flex-shrink-0" />
                      <span>{risk}</span>
                    </li>
                  )}
                </For>
              </ul>
            </div>
          </Show>

          {/* AI Summary */}
          <Show when={rec().summary}>
            <div class="bg-gradient-to-r from-primary-500/5 to-accent-500/5 border border-primary-500/20 rounded-lg p-3">
              <div class="flex items-center gap-2 mb-2">
                <AIIcon size={14} />
                <span class="text-xs text-primary-400 font-semibold">AI Analysis</span>
              </div>
              <p class="text-xs text-gray-300 leading-relaxed">
                {rec().summary}
              </p>
            </div>
          </Show>

          {/* Action Buttons */}
          <div class="flex gap-2 pt-2">
            <Show when={rec().should_add}>
              <button
                onClick={() => props.onTrade?.(rec().symbol, 'buy')}
                class="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-success-500/20 hover:bg-success-500/30 text-success-400 rounded-lg transition-colors"
              >
                <Plus class="w-4 h-4" />
                <span>Add Position</span>
              </button>
            </Show>
            
            <Show when={rec().should_trim || rec().should_exit}>
              <button
                onClick={() => props.onTrade?.(rec().symbol, 'sell')}
                class={`flex-1 flex items-center justify-center gap-2 px-4 py-2 ${
                  rec().should_exit 
                    ? 'bg-danger-500/20 hover:bg-danger-500/30 text-danger-400' 
                    : 'bg-warning-500/20 hover:bg-warning-500/30 text-warning-400'
                } rounded-lg transition-colors`}
              >
                <Minus class="w-4 h-4" />
                <span>{rec().should_exit ? 'Close Position' : 'Reduce Position'}</span>
              </button>
            </Show>
            
            <Show when={!rec().should_add && !rec().should_trim && !rec().should_exit}>
              <button
                class="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-terminal-700 text-gray-400 rounded-lg cursor-default"
              >
                <Activity class="w-4 h-4" />
                <span>Hold Position</span>
              </button>
            </Show>
          </div>
        </div>
      </Show>
    </div>
  );
}

export default ExpandablePositionCard;
