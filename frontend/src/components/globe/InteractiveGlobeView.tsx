/**
 * Interactive Globe View
 * Complete globe visualization with search panel and controls
 */

import { createSignal } from 'solid-js';
import { EnhancedFinancialGlobe } from './EnhancedFinancialGlobe';
import { GlobeSearchPanel } from './GlobeSearchPanel';
import { useGlobeData } from '~/hooks/useGlobeData';

export function InteractiveGlobeView() {
  const [showArcs, setShowArcs] = createSignal(true);
  const [showBoundaries, setShowBoundaries] = createSignal(false);
  const [autoRotate, setAutoRotate] = createSignal(true);

  const globeData = useGlobeData({
    timeframe: '24h',
    min_articles: 1,
    min_strength: 0.3,
  });

  return (
    <div class="min-h-screen bg-terminal-950 text-white">
      {/* Header */}
      <div class="border-b border-terminal-800 bg-terminal-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div class="container mx-auto px-3 sm:px-4 py-3 sm:py-4">
          <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-0">
            <div>
              <h1 class="text-xl sm:text-2xl font-bold text-white">Global Financial News</h1>
              <p class="text-gray-400 text-xs sm:text-sm mt-1">
                Interactive 3D visualization of worldwide market activity
              </p>
            </div>
            
            {/* View Controls */}
            <div class="flex flex-wrap items-center gap-2 sm:gap-4 w-full sm:w-auto">
              <button
                onClick={() => setShowArcs(!showArcs())}
                class={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  showArcs()
                    ? 'bg-accent-500 text-white'
                    : 'bg-terminal-800 text-gray-400 hover:bg-terminal-750'
                }`}
              >
                🌈 Arcs {showArcs() ? 'ON' : 'OFF'}
              </button>
              
              <button
                onClick={() => setShowBoundaries(!showBoundaries())}
                class={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  showBoundaries()
                    ? 'bg-accent-500 text-white'
                    : 'bg-terminal-800 text-gray-400 hover:bg-terminal-750'
                }`}
              >
                🗺️ Boundaries {showBoundaries() ? 'ON' : 'OFF'}
              </button>
              
              <button
                onClick={() => setAutoRotate(!autoRotate())}
                class={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  autoRotate()
                    ? 'bg-accent-500 text-white'
                    : 'bg-terminal-800 text-gray-400 hover:bg-terminal-750'
                }`}
              >
                🔄 Auto-Rotate {autoRotate() ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div class="container mx-auto px-3 sm:px-4 py-4 sm:py-6">
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-3 sm:gap-6">
          {/* Search Panel - Left Side */}
          <div class="lg:col-span-1">
            <div class="sticky top-20 sm:top-24">
              <GlobeSearchPanel
                filters={globeData.filters()}
                onFilterChange={globeData.updateFilters}
                onReset={globeData.resetFilters}
              />
              
              {/* Stats Card */}
              <div class="mt-4 bg-terminal-900/95 backdrop-blur-lg border border-terminal-750 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-3">Global Stats</h3>
                <div class="space-y-2">
                  <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-400">Exchanges</span>
                    <span class="text-lg font-bold text-white">{globeData.exchanges().length}</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-400">Connections</span>
                    <span class="text-lg font-bold text-white">{globeData.arcs().length}</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-sm text-gray-400">Countries</span>
                    <span class="text-lg font-bold text-white">{globeData.boundaries().length}</span>
                  </div>
                </div>
              </div>

              {/* Legend */}
              <div class="mt-4 bg-terminal-900/95 backdrop-blur-lg border border-terminal-750 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-3">Legend</h3>
                <div class="space-y-2 text-sm">
                  <div class="flex items-center gap-2">
                    <div class="w-3 h-3 rounded-full bg-[#00ff88]"></div>
                    <span class="text-gray-300">Positive Sentiment</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <div class="w-3 h-3 rounded-full bg-[#0088ff]"></div>
                    <span class="text-gray-300">Neutral Sentiment</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <div class="w-3 h-3 rounded-full bg-[#ff0088]"></div>
                    <span class="text-gray-300">Negative Sentiment</span>
                  </div>
                  <div class="h-px bg-terminal-750 my-2"></div>
                  <div class="flex items-center gap-2">
                    <div class="w-8 h-0.5 bg-gradient-to-r from-[#00ff88] to-[#0088ff]"></div>
                    <span class="text-gray-300">Trade Connection</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <div class="w-8 h-0.5 bg-gradient-to-r from-[#ff8800] to-[#ff0088]"></div>
                    <span class="text-gray-300">Impact Connection</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <div class="w-8 h-0.5 bg-gradient-to-r from-[#8800ff] to-[#00ffff]"></div>
                    <span class="text-gray-300">Correlation</span>
                  </div>
                </div>
              </div>

              {/* Tips */}
              <div class="mt-4 bg-terminal-900/95 backdrop-blur-lg border border-terminal-750 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-3">💡 Tips</h3>
                <ul class="space-y-2 text-sm text-gray-300">
                  <li>• <strong>Click</strong> a marker to view details</li>
                  <li>• <strong>Drag</strong> to rotate the globe</li>
                  <li>• <strong>Scroll</strong> to zoom in/out</li>
                  <li>• <strong>Hover</strong> for quick stats</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Globe - Right Side */}
          <div class="lg:col-span-3">
            <div class="bg-terminal-900/50 rounded-xl sm:rounded-2xl border border-terminal-800 overflow-hidden h-96 sm:h-[500px] lg:h-[800px]">
              <EnhancedFinancialGlobe
                autoRotate={autoRotate()}
                showArcs={showArcs()}
                showBoundaries={showBoundaries()}
                onExchangeClick={(exchange) => {
                  console.log('Exchange clicked:', exchange);
                }}
              />
            </div>

            {/* Instructions */}
            <div class="mt-4 text-center text-sm text-gray-400">
              <p>
                Visualization based on real-time news data from {globeData.exchanges().length} global stock exchanges
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
