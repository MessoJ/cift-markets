/**
 * Bloomberg-Style Asset Detail Modal
 * 
 * Professional modal for displaying:
 * - Real-time asset data
 * - Latest news & events
 * - Price charts
 * - Related assets
 * - Action buttons
 */

import { createSignal, Show, For, onMount, onCleanup } from 'solid-js';
import { X, AlertTriangle, Activity, Globe, Clock, DollarSign, BarChart3, Newspaper } from 'lucide-solid';
import { AssetMarkerData, NewsEvent, formatLargeNumber, ASSET_COLORS, EVENT_STATUS_COLORS } from '../../config/assetColors';
import { assetWebSocket } from '../../services/assetWebSocket';

interface AssetDetailModalProps {
  asset: AssetMarkerData;
  onClose: () => void;
}

export function AssetDetailModal(props: AssetDetailModalProps) {
  const [activeTab, setActiveTab] = createSignal<'overview' | 'news' | 'related' | 'chart'>('overview');
  const [liveAsset, setLiveAsset] = createSignal(props.asset);
  const [liveNews, setLiveNews] = createSignal<NewsEvent[]>(props.asset.news || []);

  onMount(() => {
    // Subscribe to real-time updates for this asset
    const unsubAsset = assetWebSocket.onAssetUpdate((updated) => {
      if (updated.id === props.asset.id) {
        setLiveAsset(updated);
      }
    });

    const unsubNews = assetWebSocket.onNewsUpdate((news) => {
      if (news.relatedAssets.includes(props.asset.id)) {
        setLiveNews(prev => [news, ...prev].slice(0, 20));
      }
    });

    onCleanup(() => {
      unsubAsset();
      unsubNews();
    });
  });

  const asset = liveAsset;
  const news = liveNews;

  const statusColor = () => EVENT_STATUS_COLORS[asset().eventStatus];
  const categoryColor = () => ASSET_COLORS[asset().category];
  const changeClass = () => {
    const change = asset().change_pct || 0;
    if (change > 0) return 'text-success-400';
    if (change < 0) return 'text-danger-400';
    return 'text-gray-400';
  };

  return (
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div class="bg-gray-900 rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden border border-gray-800 animate-slide-up">
        {/* Header */}
        <div class="relative px-6 py-4 border-b border-gray-800 bg-gradient-to-r from-gray-900 to-gray-800">
          <button
            onClick={props.onClose}
            class="absolute top-4 right-4 p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X class="w-5 h-5" />
          </button>

          <div class="flex items-start gap-4">
            {/* Category indicator */}
            <div 
              class="w-2 h-16 rounded-full"
              style={{ background: categoryColor() }}
            />

            <div class="flex-1">
              <div class="flex items-center gap-3 mb-2">
                <h2 class="text-2xl font-bold text-white">
                  {asset().name}
                </h2>
                {asset().symbol && (
                  <span class="px-2 py-1 text-xs font-mono font-semibold bg-gray-800 text-gray-300 rounded">
                    {asset().symbol}
                  </span>
                )}
                <span 
                  class="px-2 py-1 text-xs font-semibold rounded uppercase"
                  style={{ 
                    background: `${categoryColor()}20`,
                    color: categoryColor()
                  }}
                >
                  {asset().category}
                </span>
              </div>

              <div class="flex items-center gap-4 text-sm">
                <div class="flex items-center gap-2">
                  <Globe class="w-4 h-4 text-gray-400" />
                  <span class="text-gray-300">
                    {asset().lat.toFixed(2)}°, {asset().lng.toFixed(2)}°
                  </span>
                </div>

                <Show when={asset().value}>
                  <div class="flex items-center gap-2">
                    <DollarSign class="w-4 h-4 text-gray-400" />
                    <span class="text-gray-300 font-semibold">
                      {formatLargeNumber(asset().value!)}
                    </span>
                  </div>
                </Show>

                <div class="flex items-center gap-2">
                  <Clock class="w-4 h-4 text-gray-400" />
                  <span class="text-gray-400 text-xs">
                    {new Date(asset().lastUpdate).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>

            {/* Price change indicator */}
            <Show when={asset().change_pct !== undefined}>
              <div class="text-right">
                <div class={`text-3xl font-bold ${changeClass()}`}>
                  {asset().change_pct! > 0 ? '+' : ''}
                  {asset().change_pct!.toFixed(2)}%
                </div>
                <Show when={asset().change !== undefined}>
                  <div class={`text-sm ${changeClass()}`}>
                    {asset().change! > 0 ? '+' : ''}
                    {asset().change!.toFixed(2)}
                  </div>
                </Show>
              </div>
            </Show>
          </div>

          {/* Event status indicator */}
          <Show when={asset().eventStatus !== 'normal'}>
            <div 
              class="mt-3 px-3 py-2 rounded-lg flex items-center gap-2 text-sm font-medium"
              style={{ 
                background: `${statusColor()}20`,
                color: statusColor()
              }}
            >
              <AlertTriangle class="w-4 h-4" />
              <span class="uppercase tracking-wide">
                {asset().eventStatus.replace('_', ' ')}
              </span>
            </div>
          </Show>
        </div>

        {/* Tabs */}
        <div class="flex border-b border-gray-800 bg-gray-900/50">
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'news', label: 'News & Events', icon: Newspaper },
            { id: 'chart', label: 'Chart', icon: BarChart3 },
            { id: 'related', label: 'Related Assets', icon: Globe },
          ].map(tab => (
            <button
              onClick={() => setActiveTab(tab.id as any)}
              class={`flex-1 px-4 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                activeTab() === tab.id
                  ? 'text-primary-400 border-b-2 border-primary-400 bg-primary-500/5'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <tab.icon class="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div class="p-6 overflow-y-auto max-h-[calc(90vh-300px)]">
          <Show when={activeTab() === 'overview'}>
            <OverviewTab asset={asset()} />
          </Show>

          <Show when={activeTab() === 'news'}>
            <NewsTab news={news()} />
          </Show>

          <Show when={activeTab() === 'chart'}>
            <ChartTab asset={asset()} />
          </Show>

          <Show when={activeTab() === 'related'}>
            <RelatedTab asset={asset()} />
          </Show>
        </div>

        {/* Footer Actions */}
        <div class="px-6 py-4 border-t border-gray-800 bg-gray-900/50 flex items-center justify-between">
          <div class="flex gap-2">
            <button class="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors">
              Add to Watchlist
            </button>
            <button class="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors">
              Set Alert
            </button>
          </div>

          <button 
            onClick={props.onClose}
            class="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

function OverviewTab(props: { asset: AssetMarkerData }) {
  return (
    <div class="space-y-6">
      <div class="grid grid-cols-2 gap-4">
        <StatCard label="Market Value" value={props.asset.value ? formatLargeNumber(props.asset.value) : 'N/A'} />
        <StatCard label="24h Change" value={props.asset.change_pct !== undefined ? `${props.asset.change_pct > 0 ? '+' : ''}${props.asset.change_pct.toFixed(2)}%` : 'N/A'} />
        <StatCard label="Category" value={props.asset.category.toUpperCase()} />
        <StatCard label="Status" value={props.asset.eventStatus.toUpperCase()} />
      </div>

      <Show when={props.asset.news && props.asset.news.length > 0}>
        <div>
          <h3 class="text-lg font-semibold text-white mb-3">Recent Activity</h3>
          <div class="space-y-2">
            <For each={props.asset.news?.slice(0, 3)}>
              {(item) => (
                <NewsItem news={item} compact />
              )}
            </For>
          </div>
        </div>
      </Show>
    </div>
  );
}

function NewsTab(props: { news: NewsEvent[] }) {
  return (
    <div class="space-y-3">
      <Show when={props.news.length === 0}>
        <div class="text-center py-12 text-gray-400">
          <Newspaper class="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No news available</p>
        </div>
      </Show>

      <For each={props.news}>
        {(item) => <NewsItem news={item} />}
      </For>
    </div>
  );
}

function NewsItem(props: { news: NewsEvent; compact?: boolean }) {
  const importanceColor = () => {
    switch (props.news.importance) {
      case 'critical': return 'text-danger-400';
      case 'high': return 'text-warning-400';
      case 'medium': return 'text-info-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div class={`p-4 bg-gray-800/50 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors ${props.compact ? '' : 'space-y-2'}`}>
      <div class="flex items-start justify-between gap-3">
        <h4 class={`font-semibold text-white ${props.compact ? 'text-sm' : 'text-base'}`}>
          {props.news.headline}
        </h4>
        <span class={`text-xs font-semibold uppercase ${importanceColor()}`}>
          {props.news.importance}
        </span>
      </div>

      <Show when={!props.compact}>
        <p class="text-sm text-gray-400">{props.news.summary}</p>
      </Show>

      <div class="flex items-center justify-between text-xs text-gray-500">
        <span>{props.news.source}</span>
        <span>{new Date(props.news.timestamp).toLocaleString()}</span>
      </div>
    </div>
  );
}

function ChartTab(_props: { asset: AssetMarkerData }) {
  return (
    <div class="space-y-4">
      <div class="h-80 bg-gray-800/30 rounded-lg border border-gray-700 flex items-center justify-center">
        <div class="text-center text-gray-400">
          <BarChart3 class="w-16 h-16 mx-auto mb-3 opacity-50" />
          <p>Chart integration coming soon</p>
          <p class="text-sm mt-2">Connect to TradingView or custom chart service</p>
        </div>
      </div>
    </div>
  );
}

function RelatedTab(_props: { asset: AssetMarkerData }) {
  return (
    <div class="text-center py-12 text-gray-400">
      <Globe class="w-12 h-12 mx-auto mb-3 opacity-50" />
      <p>Related assets integration coming soon</p>
    </div>
  );
}

function StatCard(props: { label: string; value: string }) {
  return (
    <div class="p-4 bg-gray-800/30 rounded-lg border border-gray-700">
      <div class="text-xs text-gray-400 uppercase tracking-wide mb-1">{props.label}</div>
      <div class="text-xl font-bold text-white">{props.value}</div>
    </div>
  );
}
