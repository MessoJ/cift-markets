/**
 * MARKET NEWS INTELLIGENCE TERMINAL
 * 
 * Professional-grade financial news aggregator with:
 * - Real-time market ticker with live prices
 * - Breaking news banner with alerts
 * - Category-based filtering (Markets, Earnings, Economy, Crypto, Tech)
 * - Sentiment analysis visualization
 * - Stock impact badges with price changes
 * - Market movers sidebar (gainers/losers)
 * - Economic calendar widget
 * - Interactive 3D globe showing global news hotspots
 * 
 * Industry Standard Comparisons:
 * - Bloomberg Terminal: Professional dark UI, real-time updates, sentiment
 * - Reuters: Category tabs, regional filtering, breaking news
 * - TradingView: Stock badges, sentiment, clean design
 * - SeekingAlpha: In-article stock changes, comment counts
 * - Benzinga: Breaking news, movers, economic calendar
 */

import { createSignal, onMount, createEffect, For, Show, createMemo, onCleanup, lazy, Suspense } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import {
  Newspaper, TrendingUp, TrendingDown, Activity, Calendar,
  RefreshCw, Search, Globe, Clock, Zap, BarChart3, ArrowUpRight, 
  ArrowDownRight, AlertTriangle, X, Flame, Radio, Cpu, Bitcoin, 
  DollarSign, Building2, Briefcase, Crown
} from 'lucide-solid';
import { apiClient, NewsArticle, MarketMover, EconomicEvent } from '../../lib/api/client';
import { formatCurrency, formatPercentage } from '../../lib/utils';
import { authStore } from '../../stores/auth.store';

// Lazy load the Globe widget to avoid blocking initial render
const NewsGlobeWidget = lazy(() => import('../../components/globe/NewsGlobeWidget'));

// ============================================================================
// UTILITIES
// ============================================================================

const timeAgo = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (seconds < 60) return 'Just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return 'Yesterday';
  return `${days}d ago`;
};

const getSentimentColor = (sentiment: string): string => {
  switch (sentiment) {
    case 'positive': return 'text-success-400 bg-success-500/10 border-success-500/30';
    case 'negative': return 'text-danger-400 bg-danger-500/10 border-danger-500/30';
    default: return 'text-gray-400 bg-gray-500/10 border-gray-500/30';
  }
};

const getSentimentIcon = (sentiment: string) => {
  switch (sentiment) {
    case 'positive': return TrendingUp;
    case 'negative': return TrendingDown;
    default: return Activity;
  }
};

const getImpactColor = (impact: string): string => {
  switch (impact) {
    case 'high': return 'text-danger-400 bg-danger-500/10';
    case 'medium': return 'text-warning-400 bg-warning-500/10';
    default: return 'text-gray-400 bg-gray-500/10';
  }
};

const getCategoryIcon = (category: string) => {
  switch (category) {
    case 'market': return BarChart3;
    case 'earnings': return Briefcase;
    case 'economics': return Building2;
    case 'crypto': return Bitcoin;
    case 'technology': return Cpu;
    default: return Newspaper;
  }
};

// ============================================================================
// COMPONENT
// ============================================================================

export default function NewsPage() {
  const navigate = useNavigate();
  
  // UI State
  const [loading, setLoading] = createSignal(false);
  const [refreshing, setRefreshing] = createSignal(false);
  const [searchQuery, setSearchQuery] = createSignal('');
  const [showBreakingBanner, setShowBreakingBanner] = createSignal(true);
  const [selectedCategory, setSelectedCategory] = createSignal<string>('all');
  const [viewMode, setViewMode] = createSignal<'feed' | 'grid'>('feed');
  const [lastUpdate, setLastUpdate] = createSignal<Date>(new Date());
  
  // Data State
  const [articles, setArticles] = createSignal<NewsArticle[]>([]);
  const [marketTicker, setMarketTicker] = createSignal<any[]>([]);
  const [gainers, setGainers] = createSignal<MarketMover[]>([]);
  const [losers, setLosers] = createSignal<MarketMover[]>([]);
  const [economicEvents, setEconomicEvents] = createSignal<EconomicEvent[]>([]);
  const [breakingNews, setBreakingNews] = createSignal<NewsArticle | null>(null);

  // Categories with icons - matching DB categories
  const categories = [
    { id: 'all', label: 'All News', icon: Newspaper, color: 'accent' },
    { id: 'market', label: 'Markets', icon: BarChart3, color: 'blue' },
    { id: 'earnings', label: 'Earnings', icon: Briefcase, color: 'yellow' },
    { id: 'economics', label: 'Economy', icon: Building2, color: 'green' },
    { id: 'technology', label: 'Tech', icon: Cpu, color: 'purple' },
    { id: 'crypto', label: 'Crypto', icon: Bitcoin, color: 'orange' },
  ];

  // Derived State
  const filteredArticles = createMemo(() => {
    const query = searchQuery().toLowerCase();
    let filtered = articles();
    
    if (query) {
      filtered = filtered.filter(a => 
        a.title.toLowerCase().includes(query) || 
        a.summary.toLowerCase().includes(query) ||
        a.symbols.some(s => s.toLowerCase().includes(query))
      );
    }
    
    return filtered;
  });

  const marketSentiment = createMemo(() => {
    if (articles().length === 0) return { score: 0, label: 'Neutral', color: 'gray' };
    
    const total = articles().length;
    const positive = articles().filter(a => a.sentiment === 'positive').length;
    const negative = articles().filter(a => a.sentiment === 'negative').length;
    const score = ((positive - negative) / total) * 100;
    
    if (score > 15) return { score, label: 'Bullish', color: 'success' };
    if (score < -15) return { score, label: 'Bearish', color: 'danger' };
    return { score, label: 'Neutral', color: 'gray' };
  });

  const featuredArticle = createMemo(() => {
    const arts = filteredArticles();
    if (arts.length === 0) return null;
    // Prefer articles with images and strong sentiment
    return arts.find(a => a.image_url && a.sentiment !== 'neutral') || arts[0];
  });

  // Auto-refresh timer
  let refreshInterval: ReturnType<typeof setInterval>;
  
  onMount(() => {
    loadData();
    loadTicker();
    
    // Refresh data every 60 seconds
    refreshInterval = setInterval(() => {
      loadData(true);
      loadTicker();
    }, 60000);
  });
  
  onCleanup(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });

  // Reload when category changes
  createEffect(() => {
    if (selectedCategory()) {
      loadData();
    }
  });

  const loadTicker = async () => {
    try {
      const data = await apiClient.getMarketTicker([
        'SPY', 'QQQ', 'DIA', 'IWM', 'BTC-USD', 'ETH-USD', 'GC=F', 'CL=F'
      ]);
      setMarketTicker(data);
    } catch (err) {
      console.error('Failed to load ticker:', err);
    }
  };

  const loadData = async (silent = false) => {
    if (loading() && !silent) return;
    if (!silent) setLoading(true);
    setRefreshing(true);
    
    try {
      // Load news for category
      const newsPromise = apiClient.getNews({
        category: selectedCategory() === 'all' ? undefined : selectedCategory(),
        limit: 50,
      });

      // Load sidebar data in parallel
      const sidebarPromises = Promise.all([
        apiClient.getMarketMovers('gainers', 5),
        apiClient.getMarketMovers('losers', 5),
        apiClient.getEconomicCalendar({ limit: 10 })
      ]);

      const [newsData, [gainersData, losersData, eventsData]] = await Promise.all([
        newsPromise,
        sidebarPromises
      ]);
      
      setArticles(newsData?.articles || []);
      setGainers(gainersData || []);
      setLosers(losersData || []);
      setEconomicEvents(eventsData || []);
      setLastUpdate(new Date());
      
      // Set breaking news (most recent high-impact article)
      const breaking = newsData?.articles?.find(a => 
        a.sentiment !== 'neutral' && 
        new Date(a.published_at).getTime() > Date.now() - 3600000 // Within last hour
      );
      if (breaking) setBreakingNews(breaking);

    } catch (err) {
      console.error('Failed to load news data:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div class="h-full flex flex-col bg-black text-gray-300 overflow-hidden">
      
      {/* ===== TOP MARKET TICKER ===== */}
      <div class="h-8 bg-gradient-to-r from-terminal-950 via-terminal-900 to-terminal-950 border-b border-terminal-800 flex items-center overflow-hidden relative">
        {/* Live indicator */}
        <div class="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-terminal-950 to-transparent z-10 flex items-center pl-2">
          <span class="flex items-center gap-1 text-[9px] font-bold text-success-400">
            <div class="w-1.5 h-1.5 rounded-full bg-success-500 animate-pulse" />
            LIVE
          </span>
        </div>
        
        {/* Scrolling ticker */}
        <div class="flex items-center animate-scroll-left px-16 gap-6">
          <For each={[...marketTicker(), ...marketTicker(), ...marketTicker()]}>
            {(item) => (
              <div 
                class="flex items-center gap-2 text-[11px] font-mono cursor-pointer hover:bg-white/5 px-2 py-0.5 rounded transition-colors"
                onClick={() => navigate(`/symbol/${item.symbol}`)}
              >
                <span class="font-bold text-white">{item.symbol}</span>
                <span class={item.change >= 0 ? 'text-success-400' : 'text-danger-400'}>
                  {formatCurrency(item.price)}
                </span>
                <span class={`flex items-center ${item.change >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                  {item.change >= 0 ? <ArrowUpRight class="w-3 h-3" /> : <ArrowDownRight class="w-3 h-3" />}
                  {item.changePercent?.toFixed(2)}%
                </span>
              </div>
            )}
          </For>
        </div>
        
        {/* Gradient fade right */}
        <div class="absolute right-0 top-0 bottom-0 w-16 bg-gradient-to-l from-terminal-950 to-transparent z-10" />
      </div>

      {/* ===== BREAKING NEWS BANNER ===== */}
      <Show when={breakingNews() && showBreakingBanner()}>
        <div class="bg-gradient-to-r from-danger-900/40 via-danger-800/30 to-danger-900/40 border-b border-danger-700/50 px-4 py-2 flex items-center gap-3 animate-pulse-subtle">
          <div class="flex items-center gap-2 text-danger-400">
            <AlertTriangle class="w-4 h-4" />
            <span class="text-[10px] font-bold uppercase tracking-wider">Breaking</span>
          </div>
          <div class="flex-1 overflow-hidden">
            <p 
              class="text-sm text-white font-medium truncate cursor-pointer hover:text-danger-300 transition-colors"
              onClick={() => navigate(`/news/${breakingNews()!.id}`)}
            >
              {breakingNews()!.title}
            </p>
          </div>
          <span class="text-[10px] text-danger-400/70 font-mono flex-shrink-0">
            {timeAgo(breakingNews()!.published_at)}
          </span>
          <button 
            onClick={() => setShowBreakingBanner(false)}
            class="text-danger-400/50 hover:text-white p-1 rounded transition-colors"
          >
            <X class="w-3 h-3" />
          </button>
        </div>
      </Show>

      {/* ===== MAIN HEADER ===== */}
      <div class="bg-terminal-900/80 backdrop-blur border-b border-terminal-800 p-4">
        <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          {/* Title & Status */}
          <div class="flex items-center gap-4">
            <div class="w-12 h-12 bg-accent-500/10 rounded-lg flex items-center justify-center border border-accent-500/20">
              <Newspaper class="w-6 h-6 text-accent-500" />
            </div>
            <div>
              <h1 class="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                News Terminal
                <Show when={authStore.user()?.is_superuser}>
                  <span class="px-2 py-0.5 text-[9px] font-bold uppercase bg-gradient-to-r from-amber-500/20 to-accent-500/20 text-amber-400 rounded border border-amber-500/30 flex items-center gap-1">
                    <Crown class="w-3 h-3" />
                    Pro
                  </span>
                </Show>
                <Show when={!authStore.user()?.is_superuser}>
                  <span class="px-2 py-0.5 text-[9px] font-bold uppercase bg-gray-500/20 text-gray-400 rounded border border-gray-500/30">
                    Basic
                  </span>
                </Show>
              </h1>
              <div class="flex items-center gap-3 text-xs text-gray-500 mt-1">
                <span class="flex items-center gap-1.5">
                  <div class="w-1.5 h-1.5 rounded-full bg-success-500 animate-pulse" />
                  Real-time Feed
                </span>
                <span class="text-gray-600">•</span>
                <span>{articles().length} Articles</span>
                <span class="text-gray-600">•</span>
                <span class="flex items-center gap-1">
                  <Clock class="w-3 h-3" />
                  Updated {timeAgo(lastUpdate().toISOString())}
                </span>
              </div>
            </div>
          </div>

          {/* Search & Actions */}
          <div class="flex items-center gap-3 w-full lg:w-auto">
            {/* Search */}
            <div class="relative flex-1 lg:w-72">
              <Search class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search news, symbols, topics..."
                value={searchQuery()}
                onInput={(e) => setSearchQuery(e.currentTarget.value)}
                class="w-full bg-terminal-850 border border-terminal-700 rounded-lg pl-10 pr-4 py-2.5 text-sm text-white placeholder-gray-500 focus:border-accent-500 focus:ring-1 focus:ring-accent-500/20 focus:outline-none transition-all"
              />
            </div>
            
            {/* Sentiment Indicator */}
            <div class={`hidden md:flex items-center gap-2 px-3 py-2 rounded-lg border ${
              marketSentiment().color === 'success' ? 'bg-success-500/10 border-success-500/30 text-success-400' :
              marketSentiment().color === 'danger' ? 'bg-danger-500/10 border-danger-500/30 text-danger-400' :
              'bg-gray-500/10 border-gray-500/30 text-gray-400'
            }`}>
              {marketSentiment().color === 'success' ? <TrendingUp class="w-4 h-4" /> :
               marketSentiment().color === 'danger' ? <TrendingDown class="w-4 h-4" /> :
               <Activity class="w-4 h-4" />}
              <span class="text-xs font-bold">{marketSentiment().label}</span>
            </div>
            
            {/* Refresh */}
            <button
              onClick={() => loadData()}
              disabled={refreshing()}
              class="p-2.5 bg-terminal-850 hover:bg-terminal-800 border border-terminal-700 text-gray-400 hover:text-white rounded-lg transition-all disabled:opacity-50"
              title="Refresh Feed"
            >
              <RefreshCw class={`w-4 h-4 ${refreshing() ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* ===== CATEGORY TABS ===== */}
      <div class="bg-terminal-900/50 border-b border-terminal-800 px-4">
        <div class="flex items-center gap-1 overflow-x-auto no-scrollbar py-1">
          <For each={categories}>
            {(category) => (
              <button
                onClick={() => setSelectedCategory(category.id)}
                class={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all whitespace-nowrap ${
                  selectedCategory() === category.id
                    ? 'bg-accent-500/20 text-accent-400 border border-accent-500/30'
                    : 'text-gray-400 hover:text-white hover:bg-terminal-800/50 border border-transparent'
                }`}
              >
                <category.icon class="w-4 h-4" />
                {category.label}
              </button>
            )}
          </For>
        </div>
      </div>

      {/* ===== MAIN CONTENT ===== */}
      <div class="flex-1 flex overflow-hidden min-h-0">
        
        {/* ===== MAIN FEED (Left) ===== */}
        <div class="flex-1 overflow-y-auto min-h-0 p-4 space-y-4">
          
          {/* Featured Article */}
          <Show when={featuredArticle() && !searchQuery()}>
            <div 
              class="bg-gradient-to-br from-terminal-900 to-terminal-950 border border-terminal-700 rounded-xl overflow-hidden hover:border-accent-500/50 transition-all cursor-pointer group"
              onClick={() => navigate(`/news/${featuredArticle()!.id}`)}
            >
              <div class="flex flex-col lg:flex-row">
                <Show when={featuredArticle()!.image_url}>
                  <div class="lg:w-2/5 h-56 lg:h-auto relative overflow-hidden">
                    <img 
                      src={featuredArticle()!.image_url} 
                      class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                      alt=""
                    />
                    <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                    <div class="absolute bottom-4 left-4 flex items-center gap-2">
                      <span class={`px-2 py-1 rounded text-[10px] font-bold uppercase border backdrop-blur-sm ${getSentimentColor(featuredArticle()!.sentiment)}`}>
                        <Show when={featuredArticle()!.sentiment === 'positive'}>
                          <TrendingUp class="w-3 h-3 inline mr-1" />
                        </Show>
                        <Show when={featuredArticle()!.sentiment === 'negative'}>
                          <TrendingDown class="w-3 h-3 inline mr-1" />
                        </Show>
                        {featuredArticle()!.sentiment}
                      </span>
                    </div>
                  </div>
                </Show>
                <div class="p-6 flex-1 flex flex-col justify-center">
                  <div class="flex items-center gap-2 mb-3 flex-wrap">
                    <span class="px-2 py-0.5 bg-accent-500 text-black text-[10px] font-bold uppercase tracking-wider rounded">
                      Featured
                    </span>
                    <span class="text-xs font-bold text-accent-400">{featuredArticle()!.source}</span>
                    <span class="text-xs text-gray-500">{timeAgo(featuredArticle()!.published_at)}</span>
                  </div>
                  <h2 class="text-xl lg:text-2xl font-bold text-white mb-3 leading-tight group-hover:text-accent-400 transition-colors line-clamp-2">
                    {featuredArticle()!.title}
                  </h2>
                  <p class="text-gray-400 text-sm leading-relaxed line-clamp-3 mb-4">
                    {featuredArticle()!.summary}
                  </p>
                  <div class="flex items-center gap-2 flex-wrap">
                    <For each={featuredArticle()!.symbols.slice(0, 4)}>
                      {(symbol) => (
                        <button 
                          class="px-2 py-1 bg-terminal-800 hover:bg-accent-500/20 hover:text-accent-400 rounded text-xs font-mono text-gray-300 transition-colors flex items-center gap-1"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/symbol/${symbol}`);
                          }}
                        >
                          <span class="text-accent-400">$</span>{symbol}
                        </button>
                      )}
                    </For>
                    <Show when={featuredArticle()!.symbols.length > 4}>
                      <span class="text-xs text-gray-500">
                        +{featuredArticle()!.symbols.length - 4} more
                      </span>
                    </Show>
                  </div>
                </div>
              </div>
            </div>
          </Show>

          {/* Section Header */}
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
              <Flame class="w-4 h-4 text-accent-500" />
              {searchQuery() ? `Search Results for "${searchQuery()}"` : 'Latest Headlines'}
            </h3>
            <span class="text-xs text-gray-600">
              {filteredArticles().length} articles
            </span>
          </div>

          {/* Loading State */}
          <Show when={loading() && articles().length === 0}>
            <div class="space-y-4">
              <For each={[1, 2, 3, 4, 5]}>
                {() => (
                  <div class="bg-terminal-900 border border-terminal-800 p-4 rounded-lg animate-pulse">
                    <div class="flex gap-4">
                      <div class="flex-1 space-y-3">
                        <div class="h-4 bg-terminal-800 rounded w-1/4" />
                        <div class="h-5 bg-terminal-800 rounded w-3/4" />
                        <div class="h-4 bg-terminal-800 rounded w-full" />
                        <div class="h-4 bg-terminal-800 rounded w-2/3" />
                      </div>
                      <div class="w-24 h-24 bg-terminal-800 rounded hidden sm:block" />
                    </div>
                  </div>
                )}
              </For>
            </div>
          </Show>

          {/* Empty State */}
          <Show when={filteredArticles().length === 0 && !loading()}>
            <div class="text-center py-16 border border-dashed border-terminal-700 rounded-xl bg-terminal-900/50">
              <Search class="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p class="text-gray-400 font-medium mb-2">No articles found</p>
              <p class="text-gray-600 text-sm">
                {searchQuery() ? 'Try a different search term' : 'Check back later for new articles'}
              </p>
            </div>
          </Show>

          {/* News Articles Grid */}
          <div class="space-y-3">
            <For each={searchQuery() ? filteredArticles() : filteredArticles().slice(1)}>
              {(article) => (
                <article 
                  class="bg-terminal-900/80 border border-terminal-800 p-4 rounded-xl hover:border-accent-500/30 hover:bg-terminal-900 transition-all cursor-pointer group"
                  onClick={(e) => {
                    // Don't navigate if clicking on a button
                    if ((e.target as HTMLElement).closest('button')) return;
                    navigate(`/news/${article.id}`);
                  }}
                  role="link"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      navigate(`/news/${article.id}`);
                    }
                  }}
                >
                  <div class="flex gap-4">
                    <div class="flex-1 min-w-0">
                      {/* Meta Row */}
                      <div class="flex items-center gap-2 mb-2 flex-wrap">
                        <span class="text-[10px] font-mono text-gray-500 bg-terminal-800 px-1.5 py-0.5 rounded">
                          {timeAgo(article.published_at)}
                        </span>
                        <span class="text-xs font-bold text-accent-400">{article.source}</span>
                        <Show when={article.sentiment !== 'neutral'}>
                          <span class={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded border ${getSentimentColor(article.sentiment)}`}>
                            {article.sentiment}
                          </span>
                        </Show>
                      </div>
                      
                      {/* Title */}
                      <h3 class="text-base font-bold text-white mb-2 leading-snug group-hover:text-accent-400 transition-colors line-clamp-2">
                        {article.title}
                      </h3>
                      
                      {/* Summary */}
                      <p class="text-sm text-gray-400 line-clamp-2 mb-3 leading-relaxed">
                        {article.summary}
                      </p>
                      
                      {/* Symbols */}
                      <div class="flex items-center gap-2 flex-wrap">
                        <For each={article.symbols.slice(0, 5)}>
                          {(symbol) => (
                            <button 
                              class="text-[10px] font-mono bg-terminal-800 hover:bg-accent-500/20 text-gray-300 hover:text-accent-400 px-2 py-1 rounded transition-colors"
                              onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/symbol/${symbol}`);
                              }}
                            >
                              ${symbol}
                            </button>
                          )}
                        </For>
                      </div>
                    </div>
                    
                    {/* Thumbnail */}
                    <Show when={article.image_url}>
                      <div class="w-28 h-28 bg-terminal-800 rounded-lg overflow-hidden flex-shrink-0 hidden sm:block">
                        <img 
                          src={article.image_url} 
                          class="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500" 
                          alt=""
                        />
                      </div>
                    </Show>
                  </div>
                </article>
              )}
            </For>
          </div>
          
          {/* Load More Indicator */}
          <Show when={filteredArticles().length > 0}>
            <div class="text-center py-8 text-gray-600 text-sm">
              Showing {filteredArticles().length} of {articles().length} articles
            </div>
          </Show>
        </div>

        {/* ===== RIGHT SIDEBAR ===== */}
        <div class="w-80 border-l border-terminal-800 bg-terminal-950/50 overflow-y-auto min-h-0 hidden lg:flex flex-col">
          <div class="p-4 space-y-6 flex-1">
            
            {/* Interactive Globe Widget */}
            <Suspense fallback={
              <div class="bg-terminal-900 border border-terminal-800 rounded-xl h-64 flex items-center justify-center">
                <div class="text-xs text-gray-500">Loading globe...</div>
              </div>
            }>
              <NewsGlobeWidget />
            </Suspense>
            
            {/* Market Sentiment Widget */}
            <div class="bg-terminal-900 border border-terminal-800 rounded-xl p-4">
              <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                <Activity class="w-3.5 h-3.5" />
                Market Sentiment
              </h3>
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-gray-300">News Bias</span>
                  <span class={`text-sm font-bold ${
                    marketSentiment().color === 'success' ? 'text-success-400' :
                    marketSentiment().color === 'danger' ? 'text-danger-400' :
                    'text-gray-400'
                  }`}>
                    {marketSentiment().label}
                  </span>
                </div>
                <div class="relative h-2 bg-terminal-800 rounded-full overflow-hidden">
                  <div class="absolute top-0 bottom-0 left-1/2 w-0.5 bg-gray-600 z-10" />
                  <div 
                    class={`absolute top-0 bottom-0 transition-all duration-1000 rounded-full ${
                      marketSentiment().score >= 0 ? 'left-1/2 bg-success-500' : 'right-1/2 bg-danger-500'
                    }`}
                    style={{ width: `${Math.min(Math.abs(marketSentiment().score), 50)}%` }}
                  />
                </div>
                <div class="flex justify-between text-[10px] font-mono text-gray-500">
                  <span>Bearish</span>
                  <span>Bullish</span>
                </div>
              </div>
            </div>

            {/* Top Gainers */}
            <div class="bg-terminal-900 border border-terminal-800 rounded-xl overflow-hidden">
              <div class="p-3 border-b border-terminal-800 flex items-center justify-between">
                <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                  <TrendingUp class="w-3.5 h-3.5 text-success-400" />
                  Top Gainers
                </h3>
              </div>
              <div class="divide-y divide-terminal-800">
                <For each={gainers()}>
                  {(mover) => (
                    <div 
                      class="flex items-center justify-between p-3 hover:bg-terminal-800/50 cursor-pointer transition-colors"
                      onClick={() => navigate(`/symbol/${mover.symbol}`)}
                    >
                      <div>
                        <span class="text-sm font-bold text-white">{mover.symbol}</span>
                        <span class="text-[10px] text-gray-500 ml-2">{formatCurrency(mover.price)}</span>
                      </div>
                      <span class="text-sm font-bold text-success-400 flex items-center gap-1">
                        <ArrowUpRight class="w-3 h-3" />
                        +{formatPercentage(mover.change_percent)}
                      </span>
                    </div>
                  )}
                </For>
                <Show when={gainers().length === 0}>
                  <div class="p-4 text-center text-xs text-gray-600">No data available</div>
                </Show>
              </div>
            </div>

            {/* Top Losers */}
            <div class="bg-terminal-900 border border-terminal-800 rounded-xl overflow-hidden">
              <div class="p-3 border-b border-terminal-800 flex items-center justify-between">
                <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                  <TrendingDown class="w-3.5 h-3.5 text-danger-400" />
                  Top Losers
                </h3>
              </div>
              <div class="divide-y divide-terminal-800">
                <For each={losers()}>
                  {(mover) => (
                    <div 
                      class="flex items-center justify-between p-3 hover:bg-terminal-800/50 cursor-pointer transition-colors"
                      onClick={() => navigate(`/symbol/${mover.symbol}`)}
                    >
                      <div>
                        <span class="text-sm font-bold text-white">{mover.symbol}</span>
                        <span class="text-[10px] text-gray-500 ml-2">{formatCurrency(mover.price)}</span>
                      </div>
                      <span class="text-sm font-bold text-danger-400 flex items-center gap-1">
                        <ArrowDownRight class="w-3 h-3" />
                        {formatPercentage(mover.change_percent)}
                      </span>
                    </div>
                  )}
                </For>
                <Show when={losers().length === 0}>
                  <div class="p-4 text-center text-xs text-gray-600">No data available</div>
                </Show>
              </div>
            </div>

            {/* Economic Calendar */}
            <div class="bg-terminal-900 border border-terminal-800 rounded-xl overflow-hidden">
              <div class="p-3 border-b border-terminal-800 flex items-center justify-between">
                <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                  <Calendar class="w-3.5 h-3.5" />
                  Economic Events
                </h3>
              </div>
              <div class="divide-y divide-terminal-800 max-h-64 overflow-y-auto">
                <For each={economicEvents()}>
                  {(event) => (
                    <div class="p-3 hover:bg-terminal-800/50 transition-colors">
                      <div class="flex items-start justify-between gap-2 mb-1">
                        <span class="text-xs font-bold text-gray-200 line-clamp-2 flex-1">{event.title}</span>
                        <span class={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase flex-shrink-0 ${getImpactColor(event.impact)}`}>
                          {event.impact}
                        </span>
                      </div>
                      <div class="flex items-center gap-2 text-[10px] text-gray-500 font-mono">
                        <span>{new Date(event.date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                        <span>•</span>
                        <span>{event.country}</span>
                      </div>
                    </div>
                  )}
                </For>
                <Show when={economicEvents().length === 0}>
                  <div class="p-4 text-center text-xs text-gray-600">No upcoming events</div>
                </Show>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
