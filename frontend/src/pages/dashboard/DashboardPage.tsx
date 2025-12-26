/**
 * Professional Trading Platform Dashboard - v2.0
 * 
 * Bloomberg/TradingView-inspired dashboard with:
 * - High information density grid layout
 * - Real-time portfolio metrics with sparklines
 * - Portfolio allocation donut chart
 * - Market ticker & movers
 * - Active positions table with mini charts
 * - Recent activity feed with icons
 * - Quick actions and keyboard shortcuts
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 * 
 * Business Value:
 * - Instant portfolio health check
 * - Real-time market awareness
 * - Quick trade execution
 * - Performance tracking at a glance
 */

import { createSignal, createEffect, For, Show, onMount, onCleanup } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { 
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Plus,
  Bell,
  Briefcase,
  BarChart2,
  DollarSign,
  Zap,
  RefreshCw,
  ChevronRight,
  Clock,
  TrendingUp,
  TrendingDown,
  Target,
  AlertCircle,
  ExternalLink,
  Wallet,
  PieChart,
  LineChart,
  Newspaper,
  Calendar,
  List,
  Search,
  MoreHorizontal
} from 'lucide-solid';
import { Table, Column } from '~/components/ui/Table';
import { apiClient, Position, PortfolioSummary, NewsArticle, EconomicEvent, Watchlist } from '~/lib/api/client';
import { formatCurrency, formatPercent } from '~/lib/utils/format';

// Import visualization components
import { MiniDonutChart } from '~/components/ui/DonutChart';
import { MarketTicker, MarketMovers } from '~/components/ui/MarketTicker';
import { MiniEquityCurve } from '~/components/ui/EquityCurve';
import { NoPositionsState, LoadingState } from '~/components/ui/EmptyState';

// Types for market data from backend
interface TickerItem {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
}

interface EquityCurvePoint {
  timestamp: string;
  value: number;
}

interface TodayStats {
  trades_count: number;
  volume: number;
  win_rate: number | null;
  avg_pnl: number | null;
  total_pnl: number;
  wins: number;
  losses: number;
}

export default function DashboardPage() {
  const navigate = useNavigate();

  // -- State Management --
  const [portfolio, setPortfolio] = createSignal<PortfolioSummary | null>(null);
  const [positions, setPositions] = createSignal<Position[]>([]);
  const [activities, setActivities] = createSignal<any[]>([]);
  const [tickerData, setTickerData] = createSignal<TickerItem[]>([]);
  const [moversData, setMoversData] = createSignal<{ gainers: TickerItem[]; losers: TickerItem[] }>({ gainers: [], losers: [] });
  const [equityCurve, setEquityCurve] = createSignal<EquityCurvePoint[]>([]);
  const [todayStats, setTodayStats] = createSignal<TodayStats>({
    trades_count: 0, volume: 0, win_rate: null, avg_pnl: null, total_pnl: 0, wins: 0, losses: 0
  });
  const [news, setNews] = createSignal<NewsArticle[]>([]);
  const [calendar, setCalendar] = createSignal<EconomicEvent[]>([]);
  const [watchlists, setWatchlists] = createSignal<Watchlist[]>([]);
  
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal('');
  const [lastRefresh, setLastRefresh] = createSignal<Date>(new Date());
  const [isRefreshing, setIsRefreshing] = createSignal(false);
  
  // Quick Trade State
  const [tradeSymbol, setTradeSymbol] = createSignal('');
  const [tradeSide, setTradeSide] = createSignal<'buy' | 'sell'>('buy');
  const [tradeQty, setTradeQty] = createSignal(1);

  // Right Sidebar Tab State
  const [rightTab, setRightTab] = createSignal<'news' | 'calendar'>('news');

  // Auto-refresh interval
  let refreshInterval: number;

  const fetchDashboardData = async (showLoadingState = true) => {
    try {
      if (showLoadingState) setLoading(true);
      setIsRefreshing(true);
      setError('');

      // Fetch all dashboard data in parallel
      const results = await Promise.allSettled([
        apiClient.getPortfolio(),
        apiClient.getPositions(),
        apiClient.getActivity(10),
        apiClient.getMarketTicker(),
        apiClient.getDashboardMovers(5),
        apiClient.getEquityCurveData(30),
        apiClient.getTodayStats(),
        apiClient.getNews({ limit: 5 }),
        apiClient.getEconomicCalendar({ start_date: new Date().toISOString().split('T')[0] }),
        apiClient.getWatchlists()
      ]);

      // Handle results
      if (results[0].status === 'fulfilled') setPortfolio(results[0].value);
      if (results[1].status === 'fulfilled') setPositions(results[1].value);
      if (results[2].status === 'fulfilled') setActivities(results[2].value);
      if (results[3].status === 'fulfilled') setTickerData(results[3].value);
      if (results[4].status === 'fulfilled') setMoversData(results[4].value);
      if (results[5].status === 'fulfilled') setEquityCurve(results[5].value);
      if (results[6].status === 'fulfilled') setTodayStats(results[6].value);
      if (results[7].status === 'fulfilled') setNews(results[7].value.articles);
      if (results[8].status === 'fulfilled') setCalendar(results[8].value);
      if (results[9].status === 'fulfilled') setWatchlists(results[9].value);

      setLastRefresh(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  };

  // Initial fetch & Auto-refresh
  createEffect(() => {
    fetchDashboardData();
  });

  onMount(() => {
    refreshInterval = window.setInterval(() => {
      fetchDashboardData(false);
    }, 30000);
  });

  onCleanup(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });

  const handleRefresh = () => fetchDashboardData(false);

  // Quick Trade State for Execution
  const [tradeLoading, setTradeLoading] = createSignal(false);
  const [tradeError, setTradeError] = createSignal('');
  const [tradeSuccess, setTradeSuccess] = createSignal('');

  // Quick Trade Handler - Execute order directly
  const handleQuickTrade = async (e: Event) => {
    e.preventDefault();
    if (!tradeSymbol() || tradeLoading()) return;
    
    setTradeLoading(true);
    setTradeError('');
    setTradeSuccess('');
    
    try {
      const order = await apiClient.submitOrder({
        symbol: tradeSymbol(),
        side: tradeSide(),
        order_type: 'market',
        quantity: tradeQty(),
        time_in_force: 'day',
      });
      
      console.log(`✅ Quick ${tradeSide()} order placed:`, order);
      setTradeSuccess(`✓ ${tradeSide().toUpperCase()} ${tradeQty()} ${tradeSymbol()} executed`);
      setTradeSymbol(''); // Reset form
      setTradeQty(1);
      
      // Refresh dashboard data to show new position/activity
      setTimeout(() => {
        fetchDashboardData(false);
        setTradeSuccess('');
      }, 2000);
    } catch (err: any) {
      console.error('Quick trade failed:', err);
      setTradeError(err.message || 'Order failed');
      setTimeout(() => setTradeError(''), 5000);
    } finally {
      setTradeLoading(false);
    }
  };

  // Activity icon mapper
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'fill':
      case 'buy': return <ArrowUpRight class="w-3.5 h-3.5 text-success-400" />;
      case 'sell': return <ArrowDownRight class="w-3.5 h-3.5 text-danger-400" />;
      case 'alert': return <Bell class="w-3.5 h-3.5 text-warning-400" />;
      case 'deposit':
      case 'withdrawal': return <DollarSign class="w-3.5 h-3.5 text-accent-400" />;
      default: return <Activity class="w-3.5 h-3.5 text-gray-400" />;
    }
  };

  const positionColumns: Column<Position>[] = [
    {
      key: 'symbol',
      label: 'SYMBOL',
      sortable: true,
      align: 'left',
      render: (pos) => (
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded bg-terminal-800 flex items-center justify-center text-[10px] font-bold text-gray-400">
            {pos.symbol.substring(0, 2)}
          </div>
          <div>
            <div class="font-bold text-white text-sm">{pos.symbol}</div>
            <div class={`text-[10px] font-mono ${pos.side === 'long' ? 'text-success-400' : 'text-danger-400'}`}>
              {pos.side.toUpperCase()}
            </div>
          </div>
        </div>
      ),
    },
    {
      key: 'quantity',
      label: 'SIZE',
      sortable: true,
      align: 'right',
      render: (pos) => <span class="font-mono tabular-nums text-gray-300">{pos.quantity.toLocaleString()}</span>,
    },
    {
      key: 'current_price',
      label: 'PRICE',
      sortable: true,
      align: 'right',
      render: (pos) => (
        <div class="flex flex-col items-end">
          <span class="font-mono tabular-nums text-white">{formatCurrency(pos.current_price)}</span>
          <span class="text-[10px] text-gray-500">Avg: {formatCurrency(pos.avg_cost)}</span>
        </div>
      ),
    },
    {
      key: 'market_value',
      label: 'VALUE',
      sortable: true,
      align: 'right',
      render: (pos) => <span class="font-mono tabular-nums text-white">{formatCurrency(pos.market_value)}</span>,
    },
    {
      key: 'unrealized_pnl',
      label: 'P&L',
      sortable: true,
      align: 'right',
      render: (pos) => (
        <div class="text-right">
          <div class={`font-mono tabular-nums font-semibold ${pos.unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
            {pos.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealized_pnl)}
          </div>
          <div class={`text-[10px] font-mono ${pos.unrealized_pnl_pct >= 0 ? 'text-success-400/70' : 'text-danger-400/70'}`}>
            {pos.unrealized_pnl_pct >= 0 ? '+' : ''}{formatPercent(pos.unrealized_pnl_pct)}
          </div>
        </div>
      ),
    },
    {
      key: 'actions',
      label: '',
      align: 'right',
      render: (pos) => (
        <div class="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => { e.stopPropagation(); navigate(`/charts?symbol=${pos.symbol}`); }}
            class="p-1.5 hover:bg-terminal-800 rounded text-gray-500 hover:text-white"
            title="Chart"
          >
            <LineChart class="w-3.5 h-3.5" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); navigate(`/trading?symbol=${pos.symbol}&side=sell`); }}
            class="p-1.5 hover:bg-terminal-800 rounded text-gray-500 hover:text-danger-400"
            title="Close Position"
          >
            <DollarSign class="w-3.5 h-3.5" />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div class="flex flex-col gap-2 min-h-0 bg-terminal-950 md:h-full">
      {/* 1. Market Ticker */}
      <Show when={tickerData().length > 0}>
        <div class="hidden md:block">
          <MarketTicker items={tickerData()} speed={30} />
        </div>
      </Show>

      {/* 2. Top Section: Portfolio & Quick Trade */}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-2 flex-shrink-0">
        {/* Portfolio Card */}
        <div class="lg:col-span-2 bg-terminal-900 border border-terminal-800 p-4 rounded-sm relative overflow-hidden">
          <div class="absolute top-0 right-0 p-2">
             <button onClick={handleRefresh} class={`text-gray-600 hover:text-white transition-colors ${isRefreshing() ? 'animate-spin' : ''}`}>
               <RefreshCw class="w-3.5 h-3.5" />
             </button>
          </div>
          
          <Show when={portfolio()} fallback={<LoadingState message="Loading portfolio..." />}>
            <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
              <div>
                <div class="text-xs font-mono text-gray-500 uppercase mb-1 flex items-center gap-2">
                  <Wallet class="w-3.5 h-3.5" /> Total Equity
                </div>
                <div class="text-3xl font-bold text-white font-mono tabular-nums tracking-tight">
                  {formatCurrency(portfolio()!.total_value)}
                </div>
                <div class={`flex items-center gap-2 mt-1 text-sm font-mono ${portfolio()!.day_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                  <span class="flex items-center">
                    {portfolio()!.day_pnl >= 0 ? <TrendingUp class="w-3.5 h-3.5 mr-1" /> : <TrendingDown class="w-3.5 h-3.5 mr-1" />}
                    {portfolio()!.day_pnl >= 0 ? '+' : ''}{formatCurrency(portfolio()!.day_pnl)}
                  </span>
                  <span class="opacity-75">({formatPercent(portfolio()!.day_pnl_pct)})</span>
                  <span class="text-gray-600 text-xs ml-2">Today</span>
                </div>
              </div>

              {/* Mini Stats Grid */}
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-4 border-t sm:border-t-0 sm:border-l border-terminal-800 pt-4 sm:pt-0 sm:pl-8 w-full sm:w-auto">
                <div>
                  <div class="text-[10px] text-gray-500 uppercase font-mono">Buying Power</div>
                  <div class="text-sm text-white font-mono tabular-nums">{formatCurrency(portfolio()!.buying_power)}</div>
                </div>
                <div>
                  <div class="text-[10px] text-gray-500 uppercase font-mono">Cash</div>
                  <div class="text-sm text-white font-mono tabular-nums">{formatCurrency(portfolio()!.cash)}</div>
                </div>
                <div>
                  <div class="text-[10px] text-gray-500 uppercase font-mono">Invested</div>
                  <div class="text-sm text-white font-mono tabular-nums">{formatCurrency(portfolio()!.positions_value)}</div>
                </div>
                <div>
                  <div class="text-[10px] text-gray-500 uppercase font-mono">Leverage</div>
                  <div class="text-sm text-warning-400 font-mono tabular-nums">{portfolio()!.leverage.toFixed(2)}x</div>
                </div>
              </div>
            </div>
          </Show>
        </div>

        {/* Quick Trade Ticket */}
        <div class="bg-terminal-900 border border-terminal-800 p-4 rounded-sm flex flex-col justify-center">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-xs font-bold text-gray-400 uppercase flex items-center gap-2">
              <Zap class="w-3.5 h-3.5 text-accent-500" /> Quick Trade
            </h3>
            <div class="flex bg-terminal-950 rounded p-0.5">
              <button 
                onClick={() => setTradeSide('buy')}
                class={`px-3 py-0.5 text-[10px] font-bold rounded-sm transition-colors ${tradeSide() === 'buy' ? 'bg-success-500/20 text-success-400' : 'text-gray-500 hover:text-gray-300'}`}
              >
                BUY
              </button>
              <button 
                onClick={() => setTradeSide('sell')}
                class={`px-3 py-0.5 text-[10px] font-bold rounded-sm transition-colors ${tradeSide() === 'sell' ? 'bg-danger-500/20 text-danger-400' : 'text-gray-500 hover:text-gray-300'}`}
              >
                SELL
              </button>
            </div>
          </div>
          <form onSubmit={handleQuickTrade} class="flex gap-2">
            <div class="relative flex-1">
              <Search class="absolute left-2.5 top-2.5 w-3.5 h-3.5 text-gray-500" />
              <input 
                type="text" 
                placeholder="Symbol (e.g. AAPL)" 
                class="w-full bg-terminal-950 border border-terminal-700 rounded px-3 py-2 pl-8 text-sm font-mono text-white focus:border-accent-500 focus:outline-none uppercase"
                value={tradeSymbol()}
                onInput={(e) => setTradeSymbol(e.currentTarget.value.toUpperCase())}
              />
            </div>
            <input 
              type="number" 
              min="1"
              class="w-20 bg-terminal-950 border border-terminal-700 rounded px-3 py-2 text-sm font-mono text-white focus:border-accent-500 focus:outline-none text-center"
              value={tradeQty()}
              onInput={(e) => setTradeQty(parseInt(e.currentTarget.value) || 1)}
            />
            <button 
              type="submit"
              disabled={!tradeSymbol() || tradeLoading()}
              class={`px-4 rounded font-bold text-xs transition-colors ${
                !tradeSymbol() || tradeLoading() ? 'bg-terminal-800 text-gray-500 cursor-not-allowed' :
                tradeSide() === 'buy' ? 'bg-success-500 hover:bg-success-600 text-black' : 'bg-danger-500 hover:bg-danger-600 text-white'
              }`}
            >
              {tradeLoading() ? '...' : 'GO'}
            </button>
          </form>
          <Show when={tradeSuccess()}>
            <div class="mt-2 text-xs text-success-400 font-mono">{tradeSuccess()}</div>
          </Show>
          <Show when={tradeError()}>
            <div class="mt-2 text-xs text-danger-400 font-mono">{tradeError()}</div>
          </Show>
        </div>
      </div>

      {/* 3. Main Content Grid */}
      <div class="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-2 md:flex-1 md:min-h-0 md:overflow-hidden">
        
        {/* Left Column: Positions & Activity */}
        <div class="flex flex-col gap-2 md:min-h-0 md:overflow-hidden">
          
          {/* Active Positions Table */}
          <div class="bg-terminal-900 border border-terminal-800 rounded-sm flex flex-col h-[400px] md:h-auto md:flex-1 md:min-h-0">
            <div class="px-4 py-3 border-b border-terminal-800 flex items-center justify-between flex-shrink-0">
              <div class="flex items-center gap-2">
                <Briefcase class="w-4 h-4 text-accent-500" />
                <h2 class="text-sm font-bold text-white">Positions</h2>
                <span class="bg-terminal-800 text-gray-400 text-[10px] px-1.5 py-0.5 rounded-full">{positions().length}</span>
              </div>
              <button onClick={() => navigate('/portfolio')} class="text-xs text-gray-500 hover:text-white flex items-center gap-1">
                Full Portfolio <ChevronRight class="w-3 h-3" />
              </button>
            </div>
            
            <div class="flex-1 overflow-auto min-h-0">
              <Show when={!loading()} fallback={<div class="p-8"><LoadingState /></div>}>
                <Show when={positions().length > 0} fallback={<NoPositionsState onTrade={() => navigate('/trading')} />}>
                  {/* Desktop Table View */}
                  <div class="hidden md:block h-full">
                    <Table
                      data={positions()}
                      columns={positionColumns}
                      loading={loading()}
                      emptyMessage="No positions"
                      onRowClick={(pos) => navigate(`/trading?symbol=${pos.symbol}`)}
                      compact
                      hoverable
                    />
                  </div>

                  {/* Mobile Card View */}
                  <div class="md:hidden space-y-2 p-2">
                    <For each={positions()}>
                      {(pos) => (
                        <div 
                          class="bg-terminal-950 border border-terminal-800 rounded p-3 flex justify-between items-center active:bg-terminal-800 transition-colors"
                          onClick={() => navigate(`/trading?symbol=${pos.symbol}`)}
                        >
                          <div>
                            <div class="flex items-center gap-2 mb-1">
                              <span class="font-bold text-white">{pos.symbol}</span>
                              <span class={`text-[10px] px-1.5 rounded ${pos.side === 'long' ? 'bg-success-500/20 text-success-400' : 'bg-danger-500/20 text-danger-400'}`}>
                                {pos.side.toUpperCase()}
                              </span>
                            </div>
                            <div class="text-xs text-gray-500 font-mono">
                              {pos.quantity} @ {formatCurrency(pos.avg_cost)}
                            </div>
                          </div>
                          <div class="text-right">
                            <div class="font-mono text-white font-bold">{formatCurrency(pos.current_price)}</div>
                            <div class={`text-xs font-mono ${pos.unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                              {pos.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealized_pnl)}
                            </div>
                          </div>
                        </div>
                      )}
                    </For>
                  </div>
                </Show>
              </Show>
            </div>
          </div>

          {/* Recent Activity (Collapsible/Small) */}
          <div class="h-48 bg-terminal-900 border border-terminal-800 rounded-sm flex flex-col flex-shrink-0">
            <div class="px-4 py-2 border-b border-terminal-800 flex items-center justify-between bg-terminal-900/50">
              <h3 class="text-xs font-bold text-gray-400 uppercase flex items-center gap-2">
                <Clock class="w-3.5 h-3.5" /> Recent Activity
              </h3>
              <button onClick={() => navigate('/transactions')} class="text-[10px] text-accent-500 hover:text-accent-400">View All</button>
            </div>
            <div class="flex-1 overflow-y-auto">
              <Show when={activities().length > 0} fallback={
                <div class="flex items-center justify-center h-full text-xs text-gray-500">
                  <span>No recent activity</span>
                </div>
              }>
                <For each={activities()}>
                  {(activity) => (
                    <div class="flex items-center justify-between px-4 py-2 border-b border-terminal-800/50 hover:bg-terminal-800/50 transition-colors text-xs">
                      <div class="flex items-center gap-3">
                        {getActivityIcon(activity.type)}
                        <span class="text-gray-300">
                          {activity.description || (
                            // Fallback if description is missing
                            activity.type === 'order' ? `${activity.side?.toUpperCase()} ${activity.quantity} ${activity.symbol}` :
                            activity.type === 'fill' ? `FILLED ${activity.quantity} ${activity.symbol}` :
                            activity.type === 'transfer' ? `${activity.transfer_type?.toUpperCase()} ${formatCurrency(activity.amount)}` :
                            'Activity'
                          )}
                        </span>
                      </div>
                      <div class="flex items-center gap-3">
                        <span class="font-mono text-gray-500">{new Date(activity.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                        <Show when={activity.amount}>
                          <span class={`font-mono w-16 text-right ${activity.amount >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                            {formatCurrency(activity.amount)}
                          </span>
                        </Show>
                      </div>
                    </div>
                  )}
                </For>
              </Show>
            </div>
          </div>
        </div>

        {/* Right Column: Widgets */}
        <div class="flex flex-col gap-2 md:min-h-0 md:overflow-y-auto md:pr-1">
          
          {/* Today's Performance Card */}
          <div class="bg-terminal-900 border border-terminal-800 p-4 rounded-sm flex-shrink-0">
            <h3 class="text-[10px] font-mono text-gray-500 uppercase mb-3 flex items-center gap-2">
              <Target class="w-3.5 h-3.5" /> Today's Performance
            </h3>
            <div class="space-y-3">
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-400">Net P&L</span>
                <span class={`text-sm font-mono font-bold ${todayStats().total_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                  {todayStats().total_pnl >= 0 ? '+' : ''}{formatCurrency(todayStats().total_pnl)}
                </span>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-400">Win Rate</span>
                <div class="flex items-center gap-2">
                  <div class="w-16 h-1.5 bg-terminal-800 rounded-full overflow-hidden">
                    <div 
                      class="h-full bg-success-500" 
                      style={{ width: `${todayStats().win_rate || 0}%` }}
                    />
                  </div>
                  <span class="text-xs font-mono text-white">{todayStats().win_rate || 0}%</span>
                </div>
              </div>
              <div class="flex justify-between items-center text-xs text-gray-500">
                <span>Trades: <span class="text-white">{todayStats().trades_count}</span></span>
                <span>Vol: <span class="text-white">{formatCurrency(todayStats().volume)}</span></span>
              </div>
            </div>
          </div>

          {/* Market Movers */}
          <div class="bg-terminal-900 border border-terminal-800 rounded-sm flex-shrink-0">
            <MarketMovers
              gainers={moversData().gainers}
              losers={moversData().losers}
              onSymbolClick={(symbol) => navigate(`/trading?symbol=${symbol}`)}
            />
          </div>

          {/* News & Calendar Tabs */}
          <div class="flex-1 bg-terminal-900 border border-terminal-800 rounded-sm flex flex-col min-h-[300px]">
            <div class="flex border-b border-terminal-800">
              <button 
                onClick={() => setRightTab('news')}
                class={`flex-1 py-2 text-xs font-bold uppercase flex items-center justify-center gap-2 transition-colors ${
                  rightTab() === 'news' ? 'bg-terminal-800 text-white border-b-2 border-accent-500' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                <Newspaper class="w-3.5 h-3.5" /> News
              </button>
              <button 
                onClick={() => setRightTab('calendar')}
                class={`flex-1 py-2 text-xs font-bold uppercase flex items-center justify-center gap-2 transition-colors ${
                  rightTab() === 'calendar' ? 'bg-terminal-800 text-white border-b-2 border-accent-500' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                <Calendar class="w-3.5 h-3.5" /> Calendar
              </button>
            </div>

            <div class="flex-1 overflow-y-auto p-0">
              <Show when={rightTab() === 'news'}>
                <Show when={news().length > 0} fallback={<div class="p-4 text-xs text-gray-500 text-center">No recent news</div>}>
                  <div class="divide-y divide-terminal-800">
                    <For each={news()}>
                      {(article) => (
                        <a href={article.url} target="_blank" rel="noopener noreferrer" class="block p-3 hover:bg-terminal-800 transition-colors group">
                          <div class="flex justify-between items-start gap-2 mb-1">
                            <span class="text-[10px] font-bold text-accent-500 uppercase">{article.source}</span>
                            <span class="text-[10px] text-gray-600">{new Date(article.published_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                          </div>
                          <h4 class="text-xs text-gray-300 group-hover:text-white leading-snug line-clamp-2 mb-1">
                            {article.title}
                          </h4>
                          <div class="flex gap-1">
                            <For each={article.symbols.slice(0, 3)}>
                              {(sym) => <span class="text-[9px] bg-terminal-950 text-gray-500 px-1 rounded">{sym}</span>}
                            </For>
                          </div>
                        </a>
                      )}
                    </For>
                  </div>
                </Show>
              </Show>

              <Show when={rightTab() === 'calendar'}>
                <Show when={calendar().length > 0} fallback={<div class="p-4 text-xs text-gray-500 text-center">No upcoming events</div>}>
                  <div class="divide-y divide-terminal-800">
                    <For each={calendar()}>
                      {(event) => (
                        <div class="p-3 hover:bg-terminal-800 transition-colors">
                          <div class="flex justify-between items-center mb-1">
                            <div class="flex items-center gap-2">
                              <span class={`w-1.5 h-1.5 rounded-full ${
                                event.impact === 'high' ? 'bg-danger-500' : event.impact === 'medium' ? 'bg-warning-500' : 'bg-success-500'
                              }`} />
                              <span class="text-[10px] font-bold text-gray-400">{event.country}</span>
                            </div>
                            <span class="text-[10px] text-gray-500">{new Date(event.date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                          </div>
                          <div class="text-xs text-gray-200 mb-1">{event.title}</div>
                          <div class="flex justify-between text-[10px] font-mono">
                            <span class="text-gray-600">Fcst: {event.forecast || '-'}</span>
                            <span class="text-white">Act: {event.actual || '-'}</span>
                          </div>
                        </div>
                      )}
                    </For>
                  </div>
                </Show>
              </Show>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
