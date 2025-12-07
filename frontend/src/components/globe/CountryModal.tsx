import { createSignal, Show, For } from 'solid-js';
import { Portal } from 'solid-js/web';

export interface CountryData {
  code: string;
  name: string;
  flag: string;
  gdp?: number | null;
  gdp_growth?: number | null;
  inflation?: number | null;
  unemployment?: number | null;
  sentiment: number | null;
  news_count: number;
  exchanges_count: number;
  assets_count: number;
  top_news?: {
    title: string;
    sentiment: number;
    source: string;
    published_at: string;
  } | null;
  recent_news?: Array<{
    title: string;
    sentiment: number;
    source: string;
  }> | null;
}

interface CountryModalProps {
  country: CountryData | null;
  onClose: () => void;
}

export function CountryModal(props: CountryModalProps) {
  const [showMoreNews, setShowMoreNews] = createSignal(false);

  const formatGDP = (gdp: number | undefined) => {
    if (!gdp) return 'N/A';
    if (gdp >= 1e12) return `$${(gdp / 1e12).toFixed(2)}T`;
    if (gdp >= 1e9) return `$${(gdp / 1e9).toFixed(1)}B`;
    return `$${(gdp / 1e6).toFixed(0)}M`;
  };

  const formatPercent = (value: number | undefined | null) => {
    if (value === undefined || value === null) return 'N/A';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  };

  const getSentimentColor = (sentiment: number | null) => {
    if (sentiment === null) return 'text-gray-400';
    if (sentiment > 0.3) return 'text-green-400';
    if (sentiment < -0.3) return 'text-red-400';
    return 'text-blue-400';
  };

  const getSentimentLabel = (sentiment: number | null) => {
    if (sentiment === null) return 'Unknown';
    if (sentiment > 0.3) return 'Positive';
    if (sentiment < -0.3) return 'Negative';
    return 'Neutral';
  };

  return (
    <Show when={props.country}>
      {(country) => (
        <Portal>
          <div
            class="fixed inset-0 bg-black/50 backdrop-blur-sm z-[9999] flex items-center justify-center p-4"
            onClick={props.onClose}
            style={{ animation: 'fadeIn 0.15s ease-out' }}
          >
            <div
              class="bg-terminal-900/98 backdrop-blur-xl border border-terminal-600 rounded-lg shadow-2xl w-full max-w-md max-h-[55vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
              style={{ animation: 'modalSlideIn 0.25s ease-out' }}
            >
              {/* Compact Header */}
              <div class="bg-gradient-to-r from-accent-600/10 to-blue-600/10 px-4 py-3 border-b border-white/5 flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class="text-3xl">{country().flag}</span>
                  <div>
                    <h2 class="text-lg font-bold text-white">{country().name}</h2>
                    <p class="text-xs text-gray-500">{country().code}</p>
                  </div>
                </div>
                <button
                  onClick={props.onClose}
                  class="text-gray-500 hover:text-white transition-colors"
                  title="Close"
                >
                  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Compact Content */}
              <div class="p-4 overflow-y-auto max-h-[calc(55vh-70px)] modal-content space-y-3">
                {/* Quick Stats */}
                <div class="grid grid-cols-2 gap-2">
                  <div class="bg-white/5 rounded p-2 border border-white/5">
                    <p class="text-xs text-gray-500">GDP</p>
                    <p class="text-sm font-bold text-white">{formatGDP(country().gdp)}</p>
                  </div>
                  <div class="bg-white/5 rounded p-2 border border-white/5">
                    <p class="text-xs text-gray-500">Growth</p>
                    <p class={`text-sm font-bold ${(country().gdp_growth ?? 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatPercent(country().gdp_growth)}
                    </p>
                  </div>
                  <div class="bg-white/5 rounded p-2 border border-white/5">
                    <p class="text-xs text-gray-500">Inflation</p>
                    <p class={`text-sm font-bold ${(country().inflation ?? 0) > 5 ? 'text-red-400' : 'text-white'}`}>
                      {formatPercent(country().inflation)}
                    </p>
                  </div>
                  <div class="bg-white/5 rounded p-2 border border-white/5">
                    <p class="text-xs text-gray-500">Unemployment</p>
                    <p class="text-sm font-bold text-white">{formatPercent(country().unemployment)}</p>
                  </div>
                </div>

                {/* Market Presence */}
                <div class="flex gap-2">
                  <div class="flex-1 bg-accent-500/10 rounded p-2 border border-accent-500/20">
                    <p class="text-xs text-gray-400">Exchanges</p>
                    <p class="text-lg font-bold text-accent-400">{country().exchanges_count || 0}</p>
                  </div>
                  <div class="flex-1 bg-blue-500/10 rounded p-2 border border-blue-500/20">
                    <p class="text-xs text-gray-400">Assets</p>
                    <p class="text-lg font-bold text-blue-400">{country().assets_count || 0}</p>
                  </div>
                </div>

                {/* News Sentiment Compact */}
                <div class="bg-white/5 rounded p-3 border border-white/5">
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-400">News Sentiment</span>
                    <span class={`text-sm font-bold ${getSentimentColor(country().sentiment)}`}>
                      {getSentimentLabel(country().sentiment)}
                    </span>
                  </div>
                  <div class="flex items-center justify-between">
                    <span class="text-xs text-gray-400">Articles</span>
                    <span class="text-sm font-bold text-white">{country().news_count || 0}</span>
                  </div>
                </div>

                {/* Top News Compact */}
                <Show when={country().top_news}>
                  <div class="bg-accent-500/5 rounded p-3 border border-accent-500/20">
                    <div class="flex items-start gap-2">
                      <div class="text-lg">🔥</div>
                      <div class="flex-1 min-w-0">
                        <p class="text-xs text-accent-400 mb-1">Top News</p>
                        <p class="text-sm text-white font-medium line-clamp-2 mb-1">{country().top_news!.title}</p>
                        <div class="flex items-center gap-2 text-xs text-gray-500">
                          <span class="truncate">{country().top_news!.source}</span>
                          <span>•</span>
                          <span class={getSentimentColor(country().top_news!.sentiment)}>
                            {getSentimentLabel(country().top_news!.sentiment)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </Show>

                {/* Recent News Compact */}
                <Show when={country().recent_news && country().recent_news!.length > 0}>
                  <button
                    onClick={() => setShowMoreNews(!showMoreNews())}
                    class="text-accent-400 hover:text-accent-300 text-xs font-medium flex items-center gap-1 w-full"
                  >
                    {showMoreNews() ? '▼' : '▶'} {country().recent_news!.length} More Articles
                  </button>

                  <Show when={showMoreNews()}>
                    <div class="space-y-2">
                      <For each={country().recent_news}>
                        {(news) => (
                          <div class="bg-white/5 rounded p-2 border border-white/5">
                            <p class="text-xs text-white mb-1 line-clamp-1">{news.title}</p>
                            <div class="flex items-center gap-2 text-xs text-gray-500">
                              <span class="truncate">{news.source}</span>
                              <span>•</span>
                              <span class={getSentimentColor(news.sentiment)}>
                                {getSentimentLabel(news.sentiment)}
                              </span>
                            </div>
                          </div>
                        )}
                      </For>
                    </div>
                  </Show>
                </Show>

                <Show when={!country().top_news && (!country().recent_news || country().recent_news!.length === 0)}>
                  <div class="text-center py-3 text-gray-500">
                    <p class="text-xs">No recent market news</p>
                  </div>
                </Show>
              </div>

              {/* Compact Footer */}
              <div class="bg-white/5 px-4 py-2 border-t border-white/5 flex justify-end">
                <button
                  onClick={props.onClose}
                  class="px-4 py-1.5 bg-white/10 hover:bg-white/20 text-white text-sm rounded transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>

          <style>{`
            @keyframes fadeIn {
              from { opacity: 0; }
              to { opacity: 1; }
            }
            @keyframes modalSlideIn {
              from {
                opacity: 0;
                transform: scale(0.95) translateY(20px);
              }
              to {
                opacity: 1;
                transform: scale(1) translateY(0);
              }
            }
            .modal-content::-webkit-scrollbar {
              width: 6px;
            }
            .modal-content::-webkit-scrollbar-track {
              background: rgba(255, 255, 255, 0.05);
              border-radius: 3px;
            }
            .modal-content::-webkit-scrollbar-thumb {
              background: rgba(255, 255, 255, 0.2);
              border-radius: 3px;
            }
          `}</style>
        </Portal>
      )}
    </Show>
  );
}
