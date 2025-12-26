import { createSignal, createEffect, Show, For } from 'solid-js';
import { AIIcon } from '~/components/icons/AIIcon';
import { 
  X, Brain, Sparkles, Cpu, Building2, Activity, Map as MapIcon, 
  TrendingUp, TrendingDown, Shield, AlertTriangle, Zap, Globe,
  ChevronRight, BarChart3, Clock, DollarSign
} from 'lucide-solid';
import { AssetMarkerData, AssetCategory, formatLargeNumber, formatPercentage } from '../../config/assetColors';
import { GlobeExchange } from '../../hooks/useGlobeData';

interface UnifiedRightPanelProps {
  isVisible: boolean;
  onClose: () => void;
  selectedEntity: any | null; // Can be Exchange, Asset, or Country
  entityType: 'exchange' | 'asset' | 'country' | null;
  aiInsights: any[];
  marketPulse: any;
  activeAssetsCount: number;
}

export function UnifiedRightPanel(props: UnifiedRightPanelProps) {
  const [activeTab, setActiveTab] = createSignal<'insights' | 'details'>('insights');

  // Auto-switch to details tab when an entity is selected
  createEffect(() => {
    if (props.selectedEntity) {
      setActiveTab('details');
    } else {
      setActiveTab('insights');
    }
  });

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'bullish': return TrendingUp;
      case 'bearish': return TrendingDown;
      case 'alert': return Shield;
      default: return Activity;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'bullish': return 'text-success-400 bg-success-500/10 border-success-500/30';
      case 'bearish': return 'text-danger-400 bg-danger-500/10 border-danger-500/30';
      case 'alert': return 'text-warning-400 bg-warning-500/10 border-warning-500/30';
      default: return 'text-gray-400 bg-gray-500/10 border-gray-500/30';
    }
  };

  return (
    <Show when={props.isVisible}>
      <div class="absolute top-16 right-4 bottom-4 w-80 z-20 flex flex-col gap-3 animate-in slide-in-from-right duration-300 pointer-events-auto">
        
        {/* Header / Tabs */}
        <div class="bg-black/80 backdrop-blur-xl border border-terminal-700/50 p-2 rounded-xl shadow-2xl flex gap-1">
          <button
            onClick={() => setActiveTab('insights')}
            class={`flex-1 py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all ${
              activeTab() === 'insights' 
                ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' 
                : 'text-gray-500 hover:text-gray-300 hover:bg-terminal-800'
            }`}
          >
            <AIIcon size={14} />
            AI Intel
          </button>
          <button
            onClick={() => setActiveTab('details')}
            disabled={!props.selectedEntity}
            class={`flex-1 py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all ${
              activeTab() === 'details'
                ? 'bg-accent-500/20 text-accent-300 border border-accent-500/30'
                : !props.selectedEntity 
                  ? 'text-gray-700 cursor-not-allowed'
                  : 'text-gray-500 hover:text-gray-300 hover:bg-terminal-800'
            }`}
          >
            <Activity class="w-3.5 h-3.5" />
            Details
          </button>
          <button 
            onClick={props.onClose}
            class="p-2 text-gray-500 hover:text-white hover:bg-terminal-800 rounded-lg transition-colors"
          >
            <X class="w-4 h-4" />
          </button>
        </div>

        {/* CONTENT: AI INSIGHTS */}
        <Show when={activeTab() === 'insights'}>
          <div class="flex-1 flex flex-col gap-3 overflow-hidden">
            {/* Status Card */}
            <div class="bg-black/70 backdrop-blur-xl border border-terminal-700/50 p-4 rounded-xl shadow-xl">
              <div class="flex items-center gap-2 text-[10px] text-violet-400 bg-violet-500/10 px-3 py-2 rounded-lg border border-violet-500/20 mb-3">
                <Cpu class="w-3 h-3 animate-pulse" />
                <span class="font-mono">Processing {props.activeAssetsCount} data streams...</span>
              </div>
              <div class="grid grid-cols-2 gap-2">
                <div class="bg-terminal-800/50 p-2 rounded-lg text-center">
                  <div class="text-[9px] text-gray-500 uppercase">Volatility</div>
                  <div class={`text-sm font-bold ${props.marketPulse.volatility > 20 ? 'text-warning-400' : 'text-success-400'}`}>
                    {props.marketPulse.volatility.toFixed(1)}
                  </div>
                </div>
                <div class="bg-terminal-800/50 p-2 rounded-lg text-center">
                  <div class="text-[9px] text-gray-500 uppercase">Sentiment</div>
                  <div class={`text-sm font-bold ${props.marketPulse.sentiment > 0 ? 'text-success-400' : 'text-danger-400'}`}>
                    {(props.marketPulse.sentiment * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Insights List */}
            <div class="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
              <For each={props.aiInsights}>
                {(insight) => {
                  const Icon = getInsightIcon(insight.type);
                  return (
                    <div class={`bg-black/70 backdrop-blur-xl border p-3 rounded-xl shadow-lg ${getInsightColor(insight.type)}`}>
                      <div class="flex items-start gap-3">
                        <div class={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${getInsightColor(insight.type)}`}>
                          <Icon class="w-3.5 h-3.5" />
                        </div>
                        <div class="flex-1 min-w-0">
                          <h3 class="text-xs font-bold text-white leading-tight mb-1">{insight.title}</h3>
                          <p class="text-[10px] text-gray-400 leading-relaxed mb-2">{insight.description}</p>
                          <div class="flex items-center gap-2 text-[9px] font-mono opacity-70">
                            <span>{insight.region}</span>
                            <span>•</span>
                            <span>{Math.round(insight.confidence * 100)}% conf</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                }}
              </For>
              <Show when={props.aiInsights.length === 0}>
                <div class="text-center py-10 text-gray-500 text-xs">
                  <Brain class="w-8 h-8 mx-auto mb-2 opacity-20" />
                  Analyzing market data...
                </div>
              </Show>
            </div>
          </div>
        </Show>

        {/* CONTENT: ENTITY DETAILS */}
        <Show when={activeTab() === 'details' && props.selectedEntity}>
          <div class="flex-1 bg-black/80 backdrop-blur-xl border border-terminal-700/50 rounded-xl shadow-2xl overflow-y-auto custom-scrollbar flex flex-col">
            
            {/* Header Image/Icon */}
            <div class="h-24 bg-gradient-to-br from-terminal-800 to-terminal-900 relative flex items-center justify-center overflow-hidden">
              <div class="absolute inset-0 opacity-20 bg-[url('/grid.png')] bg-repeat opacity-10" />
              <Show when={props.entityType === 'exchange'}>
                <Building2 class="w-10 h-10 text-accent-500/50" />
              </Show>
              <Show when={props.entityType === 'asset'}>
                <Zap class="w-10 h-10 text-yellow-500/50" />
              </Show>
              <Show when={props.entityType === 'country'}>
                <Globe class="w-10 h-10 text-blue-500/50" />
              </Show>
              <div class="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-black/80 to-transparent" />
              <div class="absolute bottom-3 left-4 right-4 flex items-end justify-between">
                <div>
                  <div class="text-[10px] font-bold text-accent-400 uppercase tracking-wider mb-0.5">
                    {props.entityType}
                  </div>
                  <h2 class="text-lg font-black text-white leading-none truncate max-w-[200px]">
                    {props.selectedEntity.name}
                  </h2>
                </div>
                <Show when={props.selectedEntity.flag || props.selectedEntity.code}>
                  <div class="text-2xl">{props.selectedEntity.flag}</div>
                </Show>
              </div>
            </div>

            <div class="p-4 space-y-4">
              {/* Common Stats Grid */}
              <div class="grid grid-cols-2 gap-2">
                <Show when={props.selectedEntity.market_cap_usd || props.selectedEntity.value}>
                  <div class="bg-terminal-800/30 p-2 rounded-lg border border-terminal-700/50">
                    <div class="text-[9px] text-gray-500 uppercase mb-1">Value</div>
                    <div class="text-xs font-bold text-white">
                      {formatLargeNumber(props.selectedEntity.market_cap_usd || props.selectedEntity.value || 0)}
                    </div>
                  </div>
                </Show>
                
                <Show when={props.selectedEntity.change_pct !== undefined}>
                  <div class="bg-terminal-800/30 p-2 rounded-lg border border-terminal-700/50">
                    <div class="text-[9px] text-gray-500 uppercase mb-1">Change (24h)</div>
                    <div class={`text-xs font-bold ${props.selectedEntity.change_pct >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                      {props.selectedEntity.change_pct > 0 ? '+' : ''}{props.selectedEntity.change_pct.toFixed(2)}%
                    </div>
                  </div>
                </Show>

                <Show when={props.selectedEntity.news_count !== undefined}>
                  <div class="bg-terminal-800/30 p-2 rounded-lg border border-terminal-700/50">
                    <div class="text-[9px] text-gray-500 uppercase mb-1">News Volume</div>
                    <div class="text-xs font-bold text-white">{props.selectedEntity.news_count}</div>
                  </div>
                </Show>

                <Show when={props.selectedEntity.sentiment_score !== undefined || props.selectedEntity.sentiment !== undefined}>
                  <div class="bg-terminal-800/30 p-2 rounded-lg border border-terminal-700/50">
                    <div class="text-[9px] text-gray-500 uppercase mb-1">Sentiment</div>
                    <div class={`text-xs font-bold ${(props.selectedEntity.sentiment_score || props.selectedEntity.sentiment) > 0 ? 'text-success-400' : 'text-danger-400'}`}>
                      {((props.selectedEntity.sentiment_score || props.selectedEntity.sentiment || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                </Show>
              </div>

              {/* Description / Categories */}
              <Show when={props.selectedEntity.categories}>
                <div>
                  <div class="text-[10px] text-gray-500 uppercase mb-2">Categories</div>
                  <div class="flex flex-wrap gap-1.5">
                    <For each={props.selectedEntity.categories.slice(0, 5)}>
                      {(cat: string) => (
                        <span class="px-2 py-1 bg-terminal-800 text-gray-300 rounded text-[10px] border border-terminal-700">
                          {cat}
                        </span>
                      )}
                    </For>
                  </div>
                </div>
              </Show>

              {/* Action Button */}
              <button class="w-full py-2.5 bg-accent-600 hover:bg-accent-500 text-white rounded-lg text-xs font-bold transition-colors flex items-center justify-center gap-2">
                Full Analytics Report
                <ChevronRight class="w-3 h-3" />
              </button>

            </div>
          </div>
        </Show>

      </div>
    </Show>
  );
}
