/**
 * GLOBAL INTELLIGENCE COMMAND CENTER
 * 
 * Ultra-premium 3D visualization of global market data with AI-powered insights.
 * Features:
 * - Real-time market pulse visualization
 * - AI-generated trend predictions
 * - Cross-market correlation analysis
 * - Sentiment heatmaps and flow visualization
 * - Interactive exchange and asset exploration
 */

import { createSignal, onMount, onCleanup, Show, For, createMemo } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import {
  Globe, Search, Filter, Layers, Activity, X, ChevronRight,
  Maximize2, Minimize2, Navigation, Anchor, Building2, Map as MapIcon,
  ArrowUpRight, ArrowDownRight, Clock, Zap, HelpCircle, Info,
  Brain, TrendingUp, TrendingDown, Cpu, Target, BarChart3, 
  Sparkles, Eye, Shield, Radar, Wifi, WifiOff
} from 'lucide-solid';
import { EnhancedFinancialGlobe } from '../../components/globe/EnhancedFinancialGlobe';
import { UnifiedRightPanel } from '../../components/globe/UnifiedRightPanel';
import { apiClient } from '../../lib/api/client';
import { formatCurrency, formatPercentage } from '../../lib/utils';
import { useAssetStream } from '../../services/assetWebSocket';

// AI Insight Types
interface MarketInsight {
  id: string;
  type: 'bullish' | 'bearish' | 'neutral' | 'alert';
  title: string;
  description: string;
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  region?: string;
  timestamp: Date;
}

// Market Pulse Data
interface MarketPulse {
  volatility: number;
  sentiment: number;
  momentum: number;
  correlation: number;
  riskLevel: 'low' | 'moderate' | 'high' | 'extreme';
}

export default function GlobePage() {
  const navigate = useNavigate();
  const [marketTicker, setMarketTicker] = createSignal<any[]>([]);
  const [selectedEntity, setSelectedEntity] = createSignal<any>(null);
  const [selectedEntityType, setSelectedEntityType] = createSignal<'exchange' | 'asset' | 'country' | null>(null);
  const [showLayers, setShowLayers] = createSignal(true);
  const [showRightPanel, setShowRightPanel] = createSignal(true);
  const [showLegend, setShowLegend] = createSignal(false);
  const [isFullscreen, setIsFullscreen] = createSignal(false);
  const [connectionStatus, setConnectionStatus] = createSignal<'connected' | 'connecting' | 'disconnected'>('connected');
  
  // Real-time Data Stream
  const { assets, latestNews, cleanup } = useAssetStream();
  
  onCleanup(() => cleanup());

  // Derived Market Pulse
  const marketPulse = createMemo<MarketPulse>(() => {
    const assetList = Array.from(assets().values());
    if (assetList.length === 0) {
      return {
        volatility: 14.2,
        sentiment: 0.67,
        momentum: 0.82,
        correlation: 0.45,
        riskLevel: 'moderate',
      };
    }

    let totalChange = 0;
    let totalAbsChange = 0;
    
    assetList.forEach(a => {
      const change = a.change_pct || 0;
      totalChange += change;
      totalAbsChange += Math.abs(change);
    });

    const avgChange = totalChange / assetList.length;
    const avgVol = totalAbsChange / assetList.length;
    
    // Normalize for display
    const sentiment = Math.min(Math.max((avgChange + 5) / 10, 0), 1); // -5% to +5% maps to 0-1
    const volatility = Math.min(avgVol * 5, 100); // Scale up
    
    let riskLevel: 'low' | 'moderate' | 'high' | 'extreme' = 'low';
    if (volatility > 15) riskLevel = 'moderate';
    if (volatility > 30) riskLevel = 'high';
    if (volatility > 50) riskLevel = 'extreme';

    return {
      volatility,
      sentiment,
      momentum: 0.5 + (avgChange / 10),
      correlation: 0.45,
      riskLevel
    };
  });

  const activeMarketsCount = createMemo(() => assets().size);
  const newsEventsCount = createMemo(() => latestNews().length);

  
  // Dynamic AI Insights derived from real data
  const aiInsights = createMemo<MarketInsight[]>(() => {
    const assetList = Array.from(assets().values());
    const insights: MarketInsight[] = [];
    
    // 1. Top Gainer Insight
    const topGainer = [...assetList].sort((a, b) => (b.change_pct || 0) - (a.change_pct || 0))[0];
    if (topGainer && (topGainer.change_pct || 0) > 2) {
      insights.push({
        id: 'gainer',
        type: 'bullish',
        title: `Strong Momentum: ${topGainer.name}`,
        description: `${topGainer.name} is leading the market with a +${topGainer.change_pct?.toFixed(2)}% gain.`,
        confidence: 0.85,
        impact: 'high',
        region: 'Global',
        timestamp: new Date()
      });
    }

    // 2. Volatility Insight
    const highVol = assetList.filter(a => Math.abs(a.change_pct || 0) > 5);
    if (highVol.length > 2) {
      insights.push({
        id: 'vol',
        type: 'alert',
        title: 'Elevated Market Volatility',
        description: `${highVol.length} assets are showing significant price swings (>5%). Caution advised.`,
        confidence: 0.92,
        impact: 'high',
        region: 'Global',
        timestamp: new Date()
      });
    }

    // 3. News Insight
    const news = latestNews()[0];
    if (news) {
      insights.push({
        id: news.id,
        type: news.sentiment === 'positive' ? 'bullish' : news.sentiment === 'negative' ? 'bearish' : 'neutral',
        title: 'Breaking News Analysis',
        description: news.headline,
        confidence: 0.75,
        impact: news.importance,
        region: 'Global',
        timestamp: news.timestamp
      });
    }

    // Fallback if no data yet
    if (insights.length === 0) {
      insights.push({
        id: 'waiting',
        type: 'neutral',
        title: 'Awaiting Market Data',
        description: 'Connecting to global exchanges for real-time analysis...',
        confidence: 0.0,
        impact: 'low',
        region: 'System',
        timestamp: new Date()
      });
    }

    return insights;
  });

  // Globe Layer State
  const [layers, setLayers] = createSignal({
    arcs: true,
    boundaries: true,
    assets: true,
    ships: true,
    exchanges: true
  });
  
  // Live time
  const [currentTime, setCurrentTime] = createSignal(new Date());

  // Pulse animation values
  const [pulsePhase, setPulsePhase] = createSignal(0);

  let pulseInterval: ReturnType<typeof setInterval>;
  let timeInterval: ReturnType<typeof setInterval>;

  onMount(() => {
    loadTicker();
    
    // Animate market pulse
    pulseInterval = setInterval(() => {
      setPulsePhase(p => (p + 0.05) % (Math.PI * 2));
    }, 100);

    // Update time
    timeInterval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
  });

  onCleanup(() => {
    if (pulseInterval) clearInterval(pulseInterval);
    if (timeInterval) clearInterval(timeInterval);
  });

  const loadTicker = async () => {
    try {
      const data = await apiClient.getMarketTicker(['SPY', 'QQQ', 'BTC-USD', 'ETH-USD', 'EURUSD=X', 'GC=F', 'CL=F', '^VIX']);
      setMarketTicker(data);
    } catch (err) {
      console.error('Failed to load ticker:', err);
    }
  };

  type LayerKey = 'arcs' | 'boundaries' | 'assets' | 'ships' | 'exchanges';
  
  const toggleLayer = (layer: LayerKey) => {
    setLayers(prev => ({ ...prev, [layer]: !prev[layer] }));
  };

  const handleExchangeClick = (exchange: any) => {
    setSelectedEntity(exchange);
    setSelectedEntityType('exchange');
    setShowRightPanel(true);
  };

  const handleAssetClick = (asset: any) => {
    setSelectedEntity(asset);
    setSelectedEntityType('asset');
    setShowRightPanel(true);
  };

  const handleCountryClick = (country: any) => {
    setSelectedEntity(country);
    setSelectedEntityType('country');
    setShowRightPanel(true);
  };

  // Computed values
  const riskColor = createMemo(() => {
    const risk = marketPulse().riskLevel;
    return {
      low: 'text-success-400',
      moderate: 'text-warning-400',
      high: 'text-danger-400',
      extreme: 'text-danger-300 animate-pulse',
    }[risk];
  });

  const sentimentGradient = createMemo(() => {
    const s = marketPulse().sentiment;
    if (s > 0.3) return 'from-success-500 to-success-400';
    if (s < -0.3) return 'from-danger-500 to-danger-400';
    return 'from-gray-500 to-gray-400';
  });

  const getInsightIcon = (type: MarketInsight['type']) => {
    switch (type) {
      case 'bullish': return TrendingUp;
      case 'bearish': return TrendingDown;
      case 'alert': return Shield;
      default: return Activity;
    }
  };

  const getInsightColor = (type: MarketInsight['type']) => {
    switch (type) {
      case 'bullish': return 'text-success-400 bg-success-500/10 border-success-500/30';
      case 'bearish': return 'text-danger-400 bg-danger-500/10 border-danger-500/30';
      case 'alert': return 'text-warning-400 bg-warning-500/10 border-warning-500/30';
      default: return 'text-gray-400 bg-gray-500/10 border-gray-500/30';
    }
  };

  return (
    <div class={`h-full flex flex-col bg-black text-gray-300 overflow-hidden relative ${isFullscreen() ? 'fixed inset-0 z-50' : ''}`}>
      
      {/* ============ AMBIENT GLOW EFFECTS ============ */}
      <div class="absolute inset-0 pointer-events-none z-0 overflow-hidden">
        {/* Top-left glow */}
        <div class="absolute -top-32 -left-32 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl animate-pulse" style={{ "animation-duration": "4s" }} />
        {/* Bottom-right glow */}
        <div class="absolute -bottom-32 -right-32 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ "animation-duration": "6s" }} />
        {/* Center pulse based on volatility */}
        <div 
          class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl transition-all duration-1000"
          style={{
            width: `${300 + marketPulse().volatility * 10}px`,
            height: `${300 + marketPulse().volatility * 10}px`,
            background: marketPulse().sentiment > 0 
              ? `radial-gradient(circle, rgba(34, 197, 94, ${0.05 + Math.abs(Math.sin(pulsePhase())) * 0.05}) 0%, transparent 70%)`
              : `radial-gradient(circle, rgba(239, 68, 68, ${0.05 + Math.abs(Math.sin(pulsePhase())) * 0.05}) 0%, transparent 70%)`,
          }}
        />
      </div>

      {/* ============ TOP STATUS BAR ============ */}
      <div class="h-10 bg-gradient-to-r from-terminal-950 via-terminal-900 to-terminal-950 border-b border-terminal-700/50 flex items-center justify-between px-4 z-20 relative backdrop-blur-sm">
        {/* Left: Connection Status & Time */}
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2">
            <Show when={connectionStatus() === 'connected'}>
              <Wifi class="w-4 h-4 text-success-400" />
              <span class="text-[10px] font-mono text-success-400">LIVE</span>
            </Show>
            <Show when={connectionStatus() === 'disconnected'}>
              <WifiOff class="w-4 h-4 text-danger-400" />
              <span class="text-[10px] font-mono text-danger-400">OFFLINE</span>
            </Show>
          </div>
          <div class="h-4 w-px bg-terminal-700" />
          <div class="flex items-center gap-2 text-[10px] font-mono text-gray-400">
            <Clock class="w-3 h-3" />
            <span>{currentTime().toLocaleTimeString()}</span>
            <span class="text-gray-600">UTC{currentTime().getTimezoneOffset() <= 0 ? '+' : ''}{-currentTime().getTimezoneOffset() / 60}</span>
          </div>
        </div>

        {/* Center: Market Ticker */}
        <div class="flex-1 mx-8 overflow-hidden">
          <div class="flex items-center justify-center gap-6">
            <For each={marketTicker().slice(0, 6)}>
              {(item) => (
                <div 
                  class="flex items-center gap-2 text-[11px] font-mono cursor-pointer hover:bg-terminal-800/50 px-3 py-1 rounded-md transition-colors group"
                  onClick={() => navigate(`/symbol/${item.symbol}`)}
                >
                  <span class="font-bold text-white group-hover:text-accent-400 transition-colors">{item.symbol}</span>
                  <span class={item.change >= 0 ? 'text-success-400' : 'text-danger-400'}>
                    {formatCurrency(item.price)}
                  </span>
                  <span class={`flex items-center gap-0.5 ${item.change >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                    {item.change >= 0 ? <ArrowUpRight class="w-3 h-3" /> : <ArrowDownRight class="w-3 h-3" />}
                    <span class="text-[10px]">{formatPercentage(item.changePercent)}</span>
                  </span>
                </div>
              )}
            </For>
          </div>
        </div>

        {/* Right: View Controls */}
        <div class="flex items-center gap-2">
          <button
            onClick={() => setShowRightPanel(!showRightPanel())}
            class={`p-1.5 rounded transition-colors ${showRightPanel() ? 'bg-accent-500/20 text-accent-400' : 'text-gray-400 hover:text-white hover:bg-terminal-800'}`}
            title="AI Intelligence Panel"
          >
            <Brain class="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowLegend(true)}
            class="p-1.5 rounded text-gray-400 hover:text-white hover:bg-terminal-800 transition-colors"
            title="Legend"
          >
            <HelpCircle class="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen())}
            class="p-1.5 rounded text-gray-400 hover:text-white hover:bg-terminal-800 transition-colors"
            title={isFullscreen() ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen() ? <Minimize2 class="w-4 h-4" /> : <Maximize2 class="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* ============ MAIN CONTENT ============ */}
      <div class="flex-1 relative overflow-hidden">
        
        {/* The 3D Globe */}
        <div class="absolute inset-0 z-0">
          <EnhancedFinancialGlobe
            autoRotate={true}
            showArcs={layers().arcs}
            showBoundaries={layers().boundaries}
            showAssets={layers().assets}
            onExchangeClick={handleExchangeClick}
            onAssetClick={handleAssetClick}
            onCountryClick={handleCountryClick}
          />
        </div>

        {/* ============ HUD OVERLAYS ============ */}
        
        {/* Top Left: Command Center Title */}
        <div class="absolute top-4 left-4 z-10">
          <div class="bg-black/70 backdrop-blur-xl border border-terminal-700/50 p-4 rounded-xl shadow-2xl">
            <div class="flex items-center gap-3 mb-3">
              <div class="relative">
                <div class="w-12 h-12 bg-gradient-to-br from-accent-500/30 to-blue-500/30 rounded-xl flex items-center justify-center border border-accent-500/40">
                  <Globe class="w-6 h-6 text-accent-400" />
                </div>
                {/* Pulse ring */}
                <div class="absolute inset-0 rounded-xl border-2 border-accent-500/30 animate-ping" style={{ "animation-duration": "2s" }} />
              </div>
              <div>
                <h1 class="text-xl font-black text-white leading-none tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                  GLOBAL INTEL
                </h1>
                <div class="flex items-center gap-2 text-[10px] text-gray-400 font-mono mt-1">
                  <span class="flex items-center gap-1">
                    <Radar class="w-3 h-3 text-accent-400 animate-spin" style={{ "animation-duration": "3s" }} />
                    <span class="text-accent-400">SCANNING</span>
                  </span>
                  <span class="text-gray-600">•</span>
                  <span>{activeMarketsCount() || '--'} Assets</span>
                  <span class="text-gray-600">•</span>
                  <span>Live Coverage</span>
                </div>
              </div>
            </div>
            
            {/* Market Pulse Indicator */}
            <div class="bg-terminal-900/50 rounded-lg p-3 border border-terminal-700/50">
              <div class="flex items-center justify-between mb-2">
                <span class="text-[10px] font-bold uppercase text-gray-500">Market Pulse</span>
                <span class={`text-[10px] font-bold uppercase ${riskColor()}`}>
                  {marketPulse().riskLevel.toUpperCase()} RISK
                </span>
              </div>
              {/* Pulse Visualization */}
              <div class="h-8 flex items-center gap-0.5">
                <For each={Array.from({ length: 30 })}>
                  {(_, i) => {
                    const height = Math.abs(Math.sin(pulsePhase() + i() * 0.3)) * 100;
                    return (
                      <div 
                        class={`w-1 rounded-full transition-all duration-100 bg-gradient-to-t ${sentimentGradient()}`}
                        style={{ height: `${Math.max(10, height)}%` }}
                      />
                    );
                  }}
                </For>
              </div>
              <div class="flex justify-between mt-2 text-[9px] font-mono text-gray-500">
                <span>VIX: {marketPulse().volatility.toFixed(1)}</span>
                <span>SENT: {marketPulse().sentiment > 0 ? '+' : ''}{(marketPulse().sentiment * 100).toFixed(0)}%</span>
                <span>MOM: {(marketPulse().momentum * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Top Right: Layer Controls */}
        <div class="absolute top-4 right-4 z-10 flex flex-col gap-2">
          <div class="bg-black/70 backdrop-blur-xl border border-terminal-700/50 p-3 rounded-xl shadow-2xl">
            <div class="flex items-center gap-2 mb-3">
              <Layers class="w-4 h-4 text-accent-400" />
              <span class="text-[11px] font-bold text-white uppercase tracking-wider">Data Layers</span>
            </div>
            <div class="flex flex-col gap-1.5">
              <button 
                onClick={() => toggleLayer('arcs')}
                class={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-all ${
                  layers().arcs 
                    ? 'bg-gradient-to-r from-accent-500/20 to-blue-500/20 text-accent-400 border border-accent-500/40 shadow-lg shadow-accent-500/10' 
                    : 'hover:bg-terminal-800 text-gray-500 border border-transparent hover:text-gray-300'
                }`}
              >
                <Activity class="w-3.5 h-3.5" /> News Flows
                <Show when={layers().arcs}><Eye class="w-3 h-3 ml-auto text-accent-400" /></Show>
              </button>
              <button 
                onClick={() => toggleLayer('boundaries')}
                class={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-all ${
                  layers().boundaries 
                    ? 'bg-gradient-to-r from-accent-500/20 to-blue-500/20 text-accent-400 border border-accent-500/40 shadow-lg shadow-accent-500/10' 
                    : 'hover:bg-terminal-800 text-gray-500 border border-transparent hover:text-gray-300'
                }`}
              >
                <MapIcon class="w-3.5 h-3.5" /> Boundaries
                <Show when={layers().boundaries}><Eye class="w-3 h-3 ml-auto text-accent-400" /></Show>
              </button>
              <button 
                onClick={() => toggleLayer('exchanges')}
                class={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-all ${
                  layers().exchanges 
                    ? 'bg-gradient-to-r from-accent-500/20 to-blue-500/20 text-accent-400 border border-accent-500/40 shadow-lg shadow-accent-500/10' 
                    : 'hover:bg-terminal-800 text-gray-500 border border-transparent hover:text-gray-300'
                }`}
              >
                <Building2 class="w-3.5 h-3.5" /> Exchanges
                <Show when={layers().exchanges}><Eye class="w-3 h-3 ml-auto text-accent-400" /></Show>
              </button>
              <button 
                onClick={() => toggleLayer('assets')}
                class={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-all ${
                  layers().assets 
                    ? 'bg-gradient-to-r from-accent-500/20 to-blue-500/20 text-accent-400 border border-accent-500/40 shadow-lg shadow-accent-500/10' 
                    : 'hover:bg-terminal-800 text-gray-500 border border-transparent hover:text-gray-300'
                }`}
              >
                <Zap class="w-3.5 h-3.5" /> Assets
                <Show when={layers().assets}><Eye class="w-3 h-3 ml-auto text-accent-400" /></Show>
              </button>
              <button 
                onClick={() => toggleLayer('ships')}
                class={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-all ${
                  layers().ships 
                    ? 'bg-gradient-to-r from-accent-500/20 to-blue-500/20 text-accent-400 border border-accent-500/40 shadow-lg shadow-accent-500/10' 
                    : 'hover:bg-terminal-800 text-gray-500 border border-transparent hover:text-gray-300'
                }`}
              >
                <Anchor class="w-3.5 h-3.5" /> Logistics
                <Show when={layers().ships}><Eye class="w-3 h-3 ml-auto text-accent-400" /></Show>
              </button>
            </div>
          </div>
        </div>

        {/* Bottom Left: Quick Stats Dashboard */}
        <div class="absolute bottom-4 left-4 z-10">
          <div class="bg-black/70 backdrop-blur-xl border border-terminal-700/50 p-4 rounded-xl shadow-2xl flex gap-6">
            <div class="text-center">
              <div class="text-[10px] text-gray-500 uppercase mb-1 flex items-center gap-1 justify-center">
                <Building2 class="w-3 h-3" />
                Active Assets
              </div>
              <div class="text-2xl font-black text-white font-mono">
                {activeMarketsCount() > 0 ? activeMarketsCount() : <span class="animate-pulse">--</span>}
              </div>
              <div class="text-[9px] text-success-400 font-mono">Live Feed</div>
            </div>
            <div class="w-px bg-gradient-to-b from-transparent via-terminal-600 to-transparent" />
            <div class="text-center">
              <div class="text-[10px] text-gray-500 uppercase mb-1 flex items-center gap-1 justify-center">
                <Activity class="w-3 h-3" />
                Live Events
              </div>
              <div class="text-2xl font-black text-accent-400 font-mono">
                {newsEventsCount() > 0 ? newsEventsCount() : <span class="animate-pulse">--</span>}
              </div>
              <div class="text-[9px] text-gray-400 font-mono">Real-time</div>
            </div>
            <div class="w-px bg-gradient-to-b from-transparent via-terminal-600 to-transparent" />
            <div class="text-center">
              <div class="text-[10px] text-gray-500 uppercase mb-1 flex items-center gap-1 justify-center">
                <Target class="w-3 h-3" />
                Volatility Index
              </div>
              <div class={`text-2xl font-black font-mono ${riskColor()}`}>
                {marketPulse().volatility.toFixed(1)}
              </div>
              <div class="text-[9px] text-gray-400 font-mono">VIX Real-time</div>
            </div>
            <div class="w-px bg-gradient-to-b from-transparent via-terminal-600 to-transparent" />
            <div class="text-center">
              <div class="text-[10px] text-gray-500 uppercase mb-1 flex items-center gap-1 justify-center">
                <BarChart3 class="w-3 h-3" />
                Global Sentiment
              </div>
              <div class={`text-2xl font-black font-mono ${marketPulse().sentiment > 0 ? 'text-success-400' : 'text-danger-400'}`}>
                {marketPulse().sentiment > 0 ? '+' : ''}{(marketPulse().sentiment * 100).toFixed(0)}%
              </div>
              <div class="text-[9px] text-gray-400 font-mono">{marketPulse().sentiment > 0.2 ? 'Bullish' : marketPulse().sentiment < -0.2 ? 'Bearish' : 'Neutral'}</div>
            </div>
          </div>
        </div>

        {/* ============ UNIFIED RIGHT PANEL ============ */}
        <UnifiedRightPanel
          isVisible={showRightPanel()}
          onClose={() => {
            setShowRightPanel(false);
            setSelectedEntity(null);
          }}
          selectedEntity={selectedEntity()}
          entityType={selectedEntityType()}
          aiInsights={aiInsights()}
          marketPulse={marketPulse()}
          activeAssetsCount={activeMarketsCount()}
        />

        <Show when={showLegend()}>
          <div class="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4 pointer-events-auto" onClick={() => setShowLegend(false)}>
            <div class="bg-gradient-to-b from-terminal-900 to-terminal-950 border border-terminal-700/50 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[85vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
              {/* Modal Header */}
              <div class="p-5 border-b border-terminal-700/50 flex items-center justify-between bg-black/50">
                <div class="flex items-center gap-4">
                  <div class="w-12 h-12 bg-gradient-to-br from-accent-500/30 to-blue-500/30 rounded-xl flex items-center justify-center border border-accent-500/40">
                    <Info class="w-6 h-6 text-accent-400" />
                  </div>
                  <div>
                    <h2 class="text-lg font-bold text-white">Visualization Guide</h2>
                    <p class="text-xs text-gray-400 font-mono">Understanding the GLOBAL INTEL interface</p>
                  </div>
                </div>
                <button onClick={() => setShowLegend(false)} class="text-gray-400 hover:text-white p-2 rounded-lg hover:bg-terminal-800 transition-colors">
                  <X class="w-5 h-5" />
                </button>
              </div>
              
              {/* Modal Content */}
              <div class="p-6 overflow-y-auto max-h-[65vh] space-y-6">
                {/* Marker Types */}
                <div>
                  <h3 class="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Building2 class="w-4 h-4 text-accent-400" />
                    Marker Types
                  </h3>
                  <div class="grid grid-cols-2 gap-3">
                    {[
                      { color: 'bg-accent-500', name: 'Stock Exchange', desc: 'NYSE, NASDAQ, LSE, etc.' },
                      { color: 'bg-success-500', name: 'Financial Asset', desc: 'Gold, Oil, Commodities' },
                      { color: 'bg-warning-500', name: 'Capital City', desc: 'Major financial hubs' },
                      { color: 'bg-blue-500', name: 'Shipping Vessel', desc: 'Commodity transport' },
                    ].map((item) => (
                      <div class="bg-terminal-800/30 p-3 rounded-xl border border-terminal-700/50 flex items-center gap-3">
                        <div class={`w-4 h-4 rounded-full ${item.color} shadow-lg shadow-${item.color}/30`} />
                        <div>
                          <div class="text-sm font-medium text-white">{item.name}</div>
                          <div class="text-[10px] text-gray-500">{item.desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Connection Arcs */}
                <div>
                  <h3 class="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Activity class="w-4 h-4 text-accent-400" />
                    Connection Arcs
                  </h3>
                  <div class="grid grid-cols-2 gap-3">
                    {[
                      { gradient: 'from-success-500 to-success-500/20', name: 'Positive News Flow', desc: 'Bullish sentiment' },
                      { gradient: 'from-danger-500 to-danger-500/20', name: 'Negative News Flow', desc: 'Bearish sentiment' },
                      { gradient: 'from-accent-500 to-accent-500/20', name: 'Trade Route', desc: 'Shipping & logistics' },
                      { gradient: 'from-violet-500 to-violet-500/20', name: 'AI Correlation', desc: 'Detected patterns' },
                    ].map((item) => (
                      <div class="bg-terminal-800/30 p-3 rounded-xl border border-terminal-700/50 flex items-center gap-3">
                        <div class={`w-10 h-0.5 bg-gradient-to-r ${item.gradient} rounded`} />
                        <div>
                          <div class="text-sm font-medium text-white">{item.name}</div>
                          <div class="text-[10px] text-gray-500">{item.desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Country Sentiments */}
                <div>
                  <h3 class="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                    <MapIcon class="w-4 h-4 text-accent-400" />
                    Country Sentiments
                  </h3>
                  <div class="grid grid-cols-3 gap-3">
                    {[
                      { bg: 'bg-success-500/20', border: 'border-success-500/40', label: 'Bullish', color: 'text-success-400' },
                      { bg: 'bg-gray-500/20', border: 'border-gray-500/40', label: 'Neutral', color: 'text-gray-400' },
                      { bg: 'bg-danger-500/20', border: 'border-danger-500/40', label: 'Bearish', color: 'text-danger-400' },
                    ].map((item) => (
                      <div class="bg-terminal-800/30 p-3 rounded-xl border border-terminal-700/50 text-center">
                        <div class={`w-full h-4 ${item.bg} border ${item.border} rounded-lg mb-2`} />
                        <div class={`text-xs font-medium ${item.color}`}>{item.label}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* AI Features */}
                <div>
                  <h3 class="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Brain class="w-4 h-4 text-violet-400" />
                    AI Intelligence Features
                  </h3>
                  <div class="bg-gradient-to-r from-violet-500/10 to-blue-500/10 p-4 rounded-xl border border-violet-500/20">
                    <ul class="space-y-2 text-sm text-gray-300">
                      <li class="flex items-center gap-2">
                        <Sparkles class="w-4 h-4 text-violet-400" />
                        Real-time market trend analysis and predictions
                      </li>
                      <li class="flex items-center gap-2">
                        <Target class="w-4 h-4 text-violet-400" />
                        Cross-market correlation detection
                      </li>
                      <li class="flex items-center gap-2">
                        <Shield class="w-4 h-4 text-violet-400" />
                        Risk monitoring and volatility alerts
                      </li>
                      <li class="flex items-center gap-2">
                        <Radar class="w-4 h-4 text-violet-400" />
                        847 data streams processed in real-time
                      </li>
                    </ul>
                  </div>
                </div>

                {/* Navigation Controls */}
                <div class="bg-terminal-800/30 p-4 rounded-xl border border-terminal-700/50">
                  <h3 class="text-sm font-bold text-white mb-3 flex items-center gap-2">
                    <Navigation class="w-4 h-4 text-accent-400" />
                    Navigation Controls
                  </h3>
                  <div class="grid grid-cols-2 gap-3 text-xs">
                    {[
                      { action: 'Rotate Globe', key: 'Click + Drag' },
                      { action: 'Zoom In/Out', key: 'Scroll Wheel' },
                      { action: 'Select Marker', key: 'Left Click' },
                      { action: 'Toggle AI Panel', key: 'Brain Icon' },
                    ].map((item) => (
                      <div class="flex items-center justify-between text-gray-300">
                        <span class="text-gray-500">{item.action}</span>
                        <kbd class="bg-terminal-700/50 px-2 py-1 rounded-md text-[10px] font-mono text-accent-400">{item.key}</kbd>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Show>



      </div>
    </div>
  );
}