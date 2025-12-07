/**
 * Globe Filter Panel
 * Floating control panel for toggling globe features and filtering assets
 */

import { createSignal, For, Show } from 'solid-js';

export interface GlobeFilters {
  showExchanges: boolean;
  showAssets: boolean;
  showArcs: boolean;
  showBoundaries: boolean;
  assetTypes: {
    central_bank: boolean;
    commodity_market: boolean;
    government: boolean;
    tech_hq: boolean;
    energy: boolean;
  };
  assetStatus: {
    operational: boolean;
    unknown: boolean;
    issue: boolean;
  };
}

interface GlobeFilterPanelProps {
  filters: GlobeFilters;
  onFiltersChange: (filters: GlobeFilters) => void;
  exchangeCount?: number;
  assetCount?: number;
  statusCounts?: {
    operational: number;
    unknown: number;
    issue: number;
  };
}

export function GlobeFilterPanel(props: GlobeFilterPanelProps) {
  const [isExpanded, setIsExpanded] = createSignal(false);

  const toggleFeature = (feature: keyof Pick<GlobeFilters, 'showExchanges' | 'showAssets' | 'showArcs' | 'showBoundaries'>) => {
    props.onFiltersChange({
      ...props.filters,
      [feature]: !props.filters[feature],
    });
  };

  const toggleAssetType = (type: keyof GlobeFilters['assetTypes']) => {
    props.onFiltersChange({
      ...props.filters,
      assetTypes: {
        ...props.filters.assetTypes,
        [type]: !props.filters.assetTypes[type],
      },
    });
  };

  const toggleAssetStatus = (status: keyof GlobeFilters['assetStatus']) => {
    props.onFiltersChange({
      ...props.filters,
      assetStatus: {
        ...props.filters.assetStatus,
        [status]: !props.filters.assetStatus[status],
      },
    });
  };

  const resetFilters = () => {
    props.onFiltersChange({
      showExchanges: true,
      showAssets: true,
      showArcs: true,
      showBoundaries: true,
      assetTypes: {
        central_bank: true,
        commodity_market: true,
        government: true,
        tech_hq: true,
        energy: true,
      },
      assetStatus: {
        operational: true,
        unknown: true,
        issue: true,
      },
    });
  };

  const assetTypeLabels = {
    central_bank: '🏦 Central Banks',
    commodity_market: '🛢️ Commodities',
    government: '🏛️ Government',
    tech_hq: '🏢 Tech HQs',
    energy: '⚡ Energy',
  };

  return (
    <div class="absolute top-4 right-4 z-40">
      {/* Collapse/Expand Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded())}
        class="w-full bg-terminal-900/95 backdrop-blur-xl border border-terminal-700 rounded-lg p-3 hover:bg-terminal-800/95 transition-colors flex items-center justify-between"
      >
        <div class="flex items-center gap-2">
          <span class="text-lg">🔍</span>
          <span class="font-semibold text-white">Globe Filters</span>
        </div>
        <span class="text-gray-400 text-xl">
          {isExpanded() ? '−' : '+'}
        </span>
      </button>

      {/* Expanded Panel */}
      <Show when={isExpanded()}>
        <div 
          class="mt-2 bg-terminal-900/95 backdrop-blur-xl border border-terminal-700 rounded-lg p-4 w-64 max-h-[70vh] overflow-y-auto"
          style={{
            animation: 'slideDown 0.3s ease-out',
          }}
        >
          <style>{`
            @keyframes slideDown {
              from {
                opacity: 0;
                transform: translateY(-10px);
              }
              to {
                opacity: 1;
                transform: translateY(0);
              }
            }
          `}</style>

          {/* Main Features */}
          <div class="mb-4">
            <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-2">
              Show/Hide
            </h3>
            <div class="space-y-2">
              <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                <div class="flex items-center gap-2">
                  <span class="text-sm text-white">Exchanges</span>
                  <span class="text-xs text-gray-500">({props.exchangeCount || 0})</span>
                </div>
                <input
                  type="checkbox"
                  checked={props.filters.showExchanges}
                  onChange={() => toggleFeature('showExchanges')}
                  class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                />
              </label>

              <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                <div class="flex items-center gap-2">
                  <span class="text-sm text-white">Assets</span>
                  <span class="text-xs text-gray-500">({props.assetCount || 0})</span>
                </div>
                <input
                  type="checkbox"
                  checked={props.filters.showAssets}
                  onChange={() => toggleFeature('showAssets')}
                  class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                />
              </label>

              <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                <span class="text-sm text-white">Connections</span>
                <input
                  type="checkbox"
                  checked={props.filters.showArcs}
                  onChange={() => toggleFeature('showArcs')}
                  class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                />
              </label>

              <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                <span class="text-sm text-white">Boundaries</span>
                <input
                  type="checkbox"
                  checked={props.filters.showBoundaries}
                  onChange={() => toggleFeature('showBoundaries')}
                  class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                />
              </label>
            </div>
          </div>

          {/* Asset Types Filter */}
          <Show when={props.filters.showAssets}>
            <div class="mb-4 pt-4 border-t border-terminal-700">
              <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-2">
                Asset Types
              </h3>
              <div class="space-y-2">
                <For each={Object.entries(assetTypeLabels)}>
                  {([type, label]) => (
                    <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                      <span class="text-sm text-white">{label}</span>
                      <input
                        type="checkbox"
                        checked={props.filters.assetTypes[type as keyof typeof props.filters.assetTypes]}
                        onChange={() => toggleAssetType(type as keyof GlobeFilters['assetTypes'])}
                        class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                      />
                    </label>
                  )}
                </For>
              </div>
            </div>

            {/* Asset Status Filter */}
            <div class="mb-4 pt-4 border-t border-terminal-700">
              <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-2">
                Asset Status
              </h3>
              <div class="space-y-2">
                <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                  <div class="flex items-center gap-2">
                    <span class="text-sm text-white">🟢 Operational</span>
                    <span class="text-xs text-gray-500">({props.statusCounts?.operational || 0})</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={props.filters.assetStatus.operational}
                    onChange={() => toggleAssetStatus('operational')}
                    class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                  />
                </label>

                <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                  <div class="flex items-center gap-2">
                    <span class="text-sm text-white">⚪ Unknown</span>
                    <span class="text-xs text-gray-500">({props.statusCounts?.unknown || 0})</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={props.filters.assetStatus.unknown}
                    onChange={() => toggleAssetStatus('unknown')}
                    class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                  />
                </label>

                <label class="flex items-center justify-between cursor-pointer hover:bg-terminal-800/50 p-2 rounded">
                  <div class="flex items-center gap-2">
                    <span class="text-sm text-white">🔴 Issues</span>
                    <span class="text-xs text-gray-500">({props.statusCounts?.issue || 0})</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={props.filters.assetStatus.issue}
                    onChange={() => toggleAssetStatus('issue')}
                    class="w-4 h-4 rounded border-gray-600 text-accent-500 focus:ring-accent-500"
                  />
                </label>
              </div>
            </div>
          </Show>

          {/* Reset Button */}
          <button
            onClick={resetFilters}
            class="w-full bg-terminal-800 hover:bg-terminal-700 text-white text-sm font-medium py-2 px-4 rounded transition-colors"
          >
            Reset All Filters
          </button>
        </div>
      </Show>
    </div>
  );
}
