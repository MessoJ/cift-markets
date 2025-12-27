/**
 * Professional Watchlists Page
 * 
 * Symbol list management with:
 * - Create/manage watchlists
 * - Add/remove symbols with sparklines
 * - Real-time quotes
 * - Quick trade actions
 * - Drag-drop reordering (visual)
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 */

import { createSignal, createEffect, Show, For } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { Table, Column } from '~/components/ui/Table';
import { apiClient } from '~/lib/api/client';
import { formatCurrency, formatPercent, formatLargeNumber } from '~/lib/utils/format';
import { Plus, X, TrendingUp, Bell, MoreHorizontal, Star, Edit2, Trash2, GripVertical, ArrowUpDown, Search, ChevronDown, Check, LayoutGrid, List, RefreshCw, Filter, TrendingDown, BarChart3, Clock } from 'lucide-solid';
import { Sparkline } from '~/components/ui/Sparkline';
import { WatchlistAnalyzer } from '~/components/analysis/WatchlistAnalyzer';

export default function WatchlistsPage() {
  const navigate = useNavigate();
  
  const [watchlists, setWatchlists] = createSignal<any[]>([]);
  const [activeList, setActiveList] = createSignal<any | null>(null);
  const [symbols, setSymbols] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [newSymbol, setNewSymbol] = createSignal('');
  const [newListName, setNewListName] = createSignal('');
  const [showNewList, setShowNewList] = createSignal(false);
  const [searchQuery, setSearchQuery] = createSignal('');
  const [sortBy, setSortBy] = createSignal<'symbol' | 'change' | 'volume' | 'price'>('symbol');
  const [sortDir, setSortDir] = createSignal<'asc' | 'desc'>('asc');
  const [showListMenu, setShowListMenu] = createSignal(false);
  const [viewMode, setViewMode] = createSignal<'table' | 'compact'>('table');
  const [isRenaming, setIsRenaming] = createSignal(false);
  const [renameValue, setRenameValue] = createSignal('');
  const [showDeleteConfirm, setShowDeleteConfirm] = createSignal(false);
  const [quickFilter, setQuickFilter] = createSignal<'all' | 'gainers' | 'losers' | 'highvol'>('all');
  const [lastUpdated, setLastUpdated] = createSignal<Date | null>(null);
  const [isRefreshing, setIsRefreshing] = createSignal(false);

  // Generate sparkline data based on available real data
  // Uses open, high, low, close to create a simple price path representation
  const getSparklineData = (item: any) => {
    const price = item.price || 0;
    const open = item.open || price;
    const high = item.high || price;
    const low = item.low || price;
    const change = item.change || 0;
    
    // If no real data available, return empty array (sparkline won't render)
    if (!price || price === 0) return [];
    
    // Create a simplified 5-point path: open -> low -> high -> price
    // This represents the day's price action using real OHLC data
    const data = [
      open,
      change >= 0 ? low : high,  // Dip first if up day, peak first if down day
      change >= 0 ? high : low,  // Peak then if up day, dip then if down day  
      price
    ];
    
    return data;
  };

  // Filter and sort symbols
  const filteredSymbols = () => {
    let result = [...symbols()];
    
    // Quick filter
    const filter = quickFilter();
    if (filter === 'gainers') {
      result = result.filter(s => (s.change_pct || 0) > 0);
    } else if (filter === 'losers') {
      result = result.filter(s => (s.change_pct || 0) < 0);
    } else if (filter === 'highvol') {
      // Top 50% by volume
      const sorted = [...result].sort((a, b) => (b.volume || 0) - (a.volume || 0));
      const cutoff = sorted[Math.floor(sorted.length / 2)]?.volume || 0;
      result = result.filter(s => (s.volume || 0) >= cutoff);
    }
    
    // Search filter
    if (searchQuery()) {
      const query = searchQuery().toLowerCase();
      result = result.filter(s => 
        s.symbol?.toLowerCase().includes(query) ||
        s.name?.toLowerCase().includes(query)
      );
    }
    
    // Sort
    result.sort((a, b) => {
      let aVal = a[sortBy()] || 0;
      let bVal = b[sortBy()] || 0;
      if (sortBy() === 'symbol') {
        aVal = a.symbol || '';
        bVal = b.symbol || '';
        return sortDir() === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDir() === 'asc' ? aVal - bVal : bVal - aVal;
    });
    
    return result;
  };

  // Format last updated time
  const formatLastUpdated = () => {
    const date = lastUpdated();
    if (!date) return '--:--:--';
    return date.toLocaleTimeString('en-US', { hour12: false });
  };

  // Manual refresh
  const handleRefresh = async () => {
    if (isRefreshing()) return;
    setIsRefreshing(true);
    await fetchSymbols();
    setIsRefreshing(false);
  };

  const fetchWatchlists = async () => {
    try {
      setError(null);
      const data = await apiClient.getWatchlists();
      setWatchlists(data);
      
      // If no watchlists exist, create a default one with popular symbols
      if (data.length === 0) {
        try {
          const defaultList = await apiClient.createWatchlist({
            name: 'My Watchlist',
            description: 'Default watchlist with popular symbols',
            symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY'],
            is_default: true
          });
          setWatchlists([defaultList]);
          setActiveList(defaultList);
        } catch (createErr) {
          console.error('Failed to create default watchlist:', createErr);
        }
      } else if (!activeList()) {
        setActiveList(data[0]);
      }
    } catch (err: any) {
      console.error('Failed to load watchlists:', err);
      if (err?.response?.status === 401) {
        setError('Please log in to view your watchlists');
      } else {
        setError('Failed to load watchlists. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchSymbols = async () => {
    if (!activeList()) return;
    try {
      setLoading(true);
      const data = await apiClient.getWatchlistSymbols(activeList()!);
      setSymbols(data);
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Failed to load symbols:', err);
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    fetchWatchlists();
  });

  createEffect(() => {
    if (activeList()) {
      fetchSymbols();
    }
  });

  const createList = async () => {
    const name = newListName().trim();
    if (!name) return;
    
    if (name.length < 3) {
      alert('Watchlist name must be at least 3 characters long');
      return;
    }
    
    if (name.length > 50) {
      alert('Watchlist name must be less than 50 characters');
      return;
    }

    try {
      await apiClient.createWatchlist({ name });
      setNewListName('');
      setShowNewList(false);
      await fetchWatchlists();
    } catch (err: any) {
      alert(`Failed to create list: ${err.message}`);
    }
  };

  const addSymbol = async () => {
    const symbol = newSymbol().trim().toUpperCase();
    if (!symbol || !activeList()) return;

    if (!/^[A-Z0-9]{1,10}$/.test(symbol)) {
      alert('Invalid symbol format. Use alphanumeric characters only (max 10).');
      return;
    }

    // Check for duplicates
    if (symbols().some(s => s.symbol === symbol)) {
      alert(`${symbol} is already in this watchlist`);
      return;
    }

    try {
      await apiClient.addSymbolToWatchlist(activeList()!, symbol);
      setNewSymbol('');
      await fetchSymbols();
    } catch (err: any) {
      alert(`Failed to add symbol: ${err.message}`);
    }
  };

  const removeSymbol = async (symbol: string) => {
    if (!activeList()) return;
    try {
      await apiClient.removeSymbolFromWatchlist(activeList()!, symbol);
      await fetchSymbols();
    } catch (err: any) {
      alert(`Failed to remove symbol: ${err.message}`);
    }
  };

  const renameList = async () => {
    if (!activeList() || !renameValue().trim()) return;
    try {
      await apiClient.updateWatchlist(activeList()!.id, { name: renameValue() });
      setIsRenaming(false);
      await fetchWatchlists();
      // Update active list name locally to avoid full reload flicker
      setActiveList({ ...activeList(), name: renameValue() });
    } catch (err: any) {
      alert(`Failed to rename list: ${err.message}`);
    }
  };

  const deleteList = async () => {
    if (!activeList()) return;
    try {
      await apiClient.deleteWatchlist(activeList()!.id);
      setShowDeleteConfirm(false);
      await fetchWatchlists();
      setActiveList(null);
    } catch (err: any) {
      alert(`Failed to delete list: ${err.message}`);
    }
  };

  const symbolColumns: Column<any>[] = [
    {
      key: 'symbol',
      label: 'SYMBOL',
      sortable: true,
      align: 'left',
      render: (item) => (
        <div class="flex items-center gap-3 relative">
          <GripVertical class="w-3 h-3 text-gray-600 cursor-grab opacity-0 group-hover:opacity-100 transition-opacity" />
          <div>
            <div class="flex items-center gap-2">
              <button
                onClick={() => navigate(`/symbol/${item.symbol}`)}
                class="font-mono font-bold text-white hover:text-accent-500 transition-colors text-sm"
              >
                {item.symbol}
              </button>
              <div onClick={(e) => e.stopPropagation()}>
                <WatchlistAnalyzer symbol={item.symbol} />
              </div>
            </div>
            <div class="text-[10px] text-gray-500 truncate max-w-[120px]">{item.name || '-'}</div>
          </div>
        </div>
      ),
    },
    {
      key: 'price',
      label: 'LAST',
      sortable: true,
      align: 'right',
      render: (item) => (
        <div class="flex flex-col items-end">
          <span class="font-mono tabular-nums text-sm font-medium text-white">{formatCurrency(item.price || 0)}</span>
          <div class="flex items-center gap-1 text-[10px] font-mono text-gray-500">
            <span>B: {item.bid ? item.bid.toFixed(2) : '-'}</span>
            <span>A: {item.ask ? item.ask.toFixed(2) : '-'}</span>
          </div>
        </div>
      ),
    },
    {
      key: 'change',
      label: 'CHG / %',
      sortable: true,
      align: 'right',
      render: (item) => (
        <div class="flex flex-col items-end">
          <span class={`font-mono tabular-nums text-sm font-bold ${(item.change || 0) >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
            {(item.change || 0) >= 0 ? '+' : ''}{item.change?.toFixed(2)}
          </span>
          <span class={`font-mono tabular-nums text-[10px] ${(item.change_pct || 0) >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
            {(item.change_pct || 0) >= 0 ? '+' : ''}{formatPercent(item.change_pct || 0)}
          </span>
        </div>
      ),
    },
    {
      key: 'sparkline',
      label: 'TREND (24H)',
      align: 'center',
      render: (item) => (
        <div class="w-24 h-8 mx-auto">
          <Sparkline 
            data={getSparklineData(item)} 
            height={32} 
            color={(item.change || 0) >= 0 ? '#22c55e' : '#ef4444'} 
            strokeWidth={1.5}
            showArea
          />
        </div>
      ),
    },
    {
      key: 'volume',
      label: 'VOL',
      sortable: true,
      align: 'right',
      render: (item) => <span class="font-mono tabular-nums text-xs text-gray-400">{formatLargeNumber(item.volume || 0)}</span>,
    },
    {
      key: 'range',
      label: 'DAY RANGE',
      align: 'center',
      render: (item) => {
        const low = item.low || item.price * 0.98;
        const high = item.high || item.price * 1.02;
        const current = item.price || 0;
        const range = high - low || 1;
        const percent = Math.max(0, Math.min(100, ((current - low) / range) * 100));
        
        return (
          <div class="w-24 flex flex-col gap-1">
            <div class="flex justify-between text-[9px] font-mono text-gray-500">
              <span>{low.toFixed(2)}</span>
              <span>{high.toFixed(2)}</span>
            </div>
            <div class="h-1 bg-terminal-800 rounded-full overflow-hidden relative">
              <div 
                class="absolute top-0 bottom-0 w-1.5 h-full bg-white rounded-full -ml-0.5"
                style={{ left: `${percent}%` }}
              />
            </div>
          </div>
        );
      },
    },
    {
      key: 'actions',
      label: '',
      align: 'right',
      render: (item) => (
        <div class="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation();
              navigate('/trading', { state: { symbol: item.symbol } });
            }}
            class="p-1.5 text-success-500 hover:text-white hover:bg-success-600 rounded transition-colors"
            title="Trade"
          >
            <TrendingUp class="w-3.5 h-3.5" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              navigate('/alerts', { state: { symbol: item.symbol } });
            }}
            class="p-1.5 text-warning-500 hover:text-white hover:bg-warning-600 rounded transition-colors"
            title="Set Alert"
          >
            <Bell class="w-3.5 h-3.5" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              removeSymbol(item.symbol);
            }}
            class="p-1.5 text-gray-500 hover:text-white hover:bg-danger-600 rounded transition-colors"
            title="Remove"
          >
            <X class="w-3.5 h-3.5" />
          </button>
        </div>
      ),
    },
  ];

  return (
    <div class="h-full flex flex-col gap-2 overflow-hidden">
      {/* Top Bar - List Selector & Tools */}
      <div class="bg-terminal-900 border border-terminal-750 p-2 flex items-center justify-between flex-shrink-0">
        <div class="flex items-center gap-2">
          {/* Watchlist Selector Dropdown */}
          <div class="relative">
            <button
              onClick={() => setShowListMenu(!showListMenu())}
              class="flex items-center gap-2 px-3 py-1.5 bg-terminal-850 border border-terminal-750 text-white font-mono text-xs hover:border-primary-500 transition-colors"
            >
              <Star class="w-3 h-3 text-warning-500" />
              <span>{activeList()?.name || 'Select List'}</span>
              <span class="text-gray-500">({symbols().length})</span>
              <ChevronDown class="w-3 h-3 text-gray-500" />
            </button>
            <Show when={showListMenu()}>
              <div class="absolute top-full left-0 mt-1 w-48 bg-terminal-900 border border-terminal-750 shadow-lg z-50">
                <div class="p-2 border-b border-terminal-750">
                  <div class="text-[10px] font-mono text-gray-500 uppercase">My Watchlists</div>
                </div>
                <div class="max-h-48 overflow-y-auto">
                  <For each={watchlists()}>
                    {(list) => (
                      <button
                        onClick={() => {
                          setActiveList(list);
                          setShowListMenu(false);
                        }}
                        class={`w-full flex items-center justify-between px-3 py-2 text-xs font-mono hover:bg-terminal-850 transition-colors ${
                          activeList()?.id === list.id ? 'bg-primary-500/10 text-primary-400' : 'text-gray-400'
                        }`}
                      >
                        <span>{list.name}</span>
                        <span class="text-gray-600">{list.symbol_count || 0}</span>
                      </button>
                    )}
                  </For>
                </div>
                <div class="p-2 border-t border-terminal-750">
                  <button
                    onClick={() => {
                      setShowNewList(true);
                      setShowListMenu(false);
                    }}
                    class="w-full flex items-center gap-2 px-2 py-1.5 text-xs font-mono text-primary-400 hover:bg-primary-500/10 transition-colors"
                  >
                    <Plus class="w-3 h-3" />
                    <span>New Watchlist</span>
                  </button>
                </div>
              </div>
            </Show>
          </div>

          {/* List Actions */}
          <Show when={activeList()}>
            <button
              onClick={() => {
                setRenameValue(activeList().name);
                setIsRenaming(true);
              }}
              class="p-1.5 text-gray-500 hover:text-white hover:bg-terminal-800 transition-colors"
              title="Edit List"
            >
              <Edit2 class="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setShowDeleteConfirm(true)}
              class="p-1.5 text-gray-500 hover:text-danger-400 hover:bg-danger-500/10 transition-colors"
              title="Delete List"
            >
              <Trash2 class="w-3.5 h-3.5" />
            </button>
          </Show>
        </div>

        {/* Right Tools */}
        <div class="flex items-center gap-2">
          {/* Last Updated */}
          <div class="hidden sm:flex items-center gap-1 text-[10px] font-mono text-gray-500 mr-2">
            <Clock class="w-3 h-3" />
            <span>{formatLastUpdated()}</span>
          </div>

          {/* Refresh */}
          <button
            onClick={handleRefresh}
            disabled={isRefreshing()}
            class="p-1.5 text-gray-500 hover:text-white hover:bg-terminal-800 transition-colors disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw class={`w-3.5 h-3.5 ${isRefreshing() ? 'animate-spin' : ''}`} />
          </button>

          {/* View Mode Toggle */}
          <div class="flex items-center border border-terminal-750 rounded overflow-hidden">
            <button
              onClick={() => setViewMode('table')}
              class={`p-1.5 transition-colors ${viewMode() === 'table' ? 'bg-primary-500/20 text-primary-400' : 'text-gray-500 hover:text-white hover:bg-terminal-800'}`}
              title="Table View"
            >
              <List class="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setViewMode('compact')}
              class={`p-1.5 transition-colors ${viewMode() === 'compact' ? 'bg-primary-500/20 text-primary-400' : 'text-gray-500 hover:text-white hover:bg-terminal-800'}`}
              title="Compact View"
            >
              <LayoutGrid class="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Search */}
          <div class="relative">
            <Search class="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-500" />
            <input
              type="text"
              value={searchQuery()}
              onInput={(e) => setSearchQuery(e.currentTarget.value)}
              placeholder="Filter..."
              class="pl-7 pr-2 py-1.5 w-28 bg-terminal-850 border border-terminal-750 text-white font-mono text-xs focus:outline-none focus:border-primary-500"
            />
          </div>

          {/* Add Symbol */}
          <div class="flex items-center gap-1">
            <input
              type="text"
              value={newSymbol()}
              onInput={(e) => setNewSymbol(e.currentTarget.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
              placeholder="+ Add"
              class="w-20 bg-terminal-850 border border-terminal-750 text-white font-mono text-xs px-2 py-1.5 uppercase focus:outline-none focus:border-success-500"
            />
            <button
              onClick={addSymbol}
              disabled={!newSymbol().trim() || !activeList()}
              class="px-2 py-1.5 bg-success-500 hover:bg-success-600 disabled:bg-terminal-800 disabled:text-gray-600 text-black text-xs font-bold font-mono transition-colors"
            >
              <Plus class="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Quick Filters Bar */}
      <Show when={activeList()}>
        <div class="flex items-center gap-2 px-1 flex-shrink-0">
          <span class="text-[10px] font-mono text-gray-600 uppercase">Filter:</span>
          <div class="flex items-center gap-1">
            <button
              onClick={() => setQuickFilter('all')}
              class={`px-2 py-1 text-[10px] font-mono rounded transition-colors ${
                quickFilter() === 'all' 
                  ? 'bg-primary-500/20 text-primary-400 border border-primary-500/50' 
                  : 'text-gray-500 hover:text-white hover:bg-terminal-800 border border-transparent'
              }`}
            >
              All ({symbols().length})
            </button>
            <button
              onClick={() => setQuickFilter('gainers')}
              class={`px-2 py-1 text-[10px] font-mono rounded flex items-center gap-1 transition-colors ${
                quickFilter() === 'gainers' 
                  ? 'bg-success-500/20 text-success-400 border border-success-500/50' 
                  : 'text-gray-500 hover:text-success-400 hover:bg-success-500/10 border border-transparent'
              }`}
            >
              <TrendingUp class="w-3 h-3" />
              Gainers ({symbols().filter(s => (s.change_pct || 0) > 0).length})
            </button>
            <button
              onClick={() => setQuickFilter('losers')}
              class={`px-2 py-1 text-[10px] font-mono rounded flex items-center gap-1 transition-colors ${
                quickFilter() === 'losers' 
                  ? 'bg-danger-500/20 text-danger-400 border border-danger-500/50' 
                  : 'text-gray-500 hover:text-danger-400 hover:bg-danger-500/10 border border-transparent'
              }`}
            >
              <TrendingDown class="w-3 h-3" />
              Losers ({symbols().filter(s => (s.change_pct || 0) < 0).length})
            </button>
            <button
              onClick={() => setQuickFilter('highvol')}
              class={`px-2 py-1 text-[10px] font-mono rounded flex items-center gap-1 transition-colors ${
                quickFilter() === 'highvol' 
                  ? 'bg-accent-500/20 text-accent-400 border border-accent-500/50' 
                  : 'text-gray-500 hover:text-accent-400 hover:bg-accent-500/10 border border-transparent'
              }`}
            >
              <BarChart3 class="w-3 h-3" />
              High Vol
            </button>
          </div>
          <div class="flex-1" />
          <span class="text-[10px] font-mono text-gray-600">
            Showing {filteredSymbols().length} of {symbols().length}
          </span>
        </div>
      </Show>

      {/* New List Modal */}
      <Show when={showNewList()}>
        <div class="bg-terminal-900 border border-primary-500 p-3 flex items-center gap-2 flex-shrink-0">
          <span class="text-xs font-mono text-gray-400">New watchlist name:</span>
          <input
            type="text"
            value={newListName()}
            onInput={(e) => setNewListName(e.currentTarget.value)}
            onKeyPress={(e) => e.key === 'Enter' && createList()}
            placeholder="My Watchlist"
            class="flex-1 bg-terminal-850 border border-terminal-750 text-white font-mono text-xs px-2 py-1.5 focus:outline-none focus:border-primary-500"
            autofocus
          />
          <button
            onClick={createList}
            class="px-3 py-1.5 bg-primary-500 hover:bg-primary-600 text-white text-xs font-bold font-mono transition-colors"
          >
            Create
          </button>
          <button
            onClick={() => {
              setShowNewList(false);
              setNewListName('');
            }}
            class="px-2 py-1.5 text-gray-400 hover:text-white text-xs font-mono transition-colors"
          >
            Cancel
          </button>
        </div>
      </Show>

      {/* Rename List Modal */}
      <Show when={isRenaming()}>
        <div class="bg-terminal-900 border border-primary-500 p-3 flex items-center gap-2 flex-shrink-0">
          <span class="text-xs font-mono text-gray-400">Rename watchlist:</span>
          <input
            type="text"
            value={renameValue()}
            onInput={(e) => setRenameValue(e.currentTarget.value)}
            onKeyPress={(e) => e.key === 'Enter' && renameList()}
            class="flex-1 bg-terminal-850 border border-terminal-750 text-white font-mono text-xs px-2 py-1.5 focus:outline-none focus:border-primary-500"
            autofocus
          />
          <button
            onClick={renameList}
            class="px-3 py-1.5 bg-primary-500 hover:bg-primary-600 text-white text-xs font-bold font-mono transition-colors"
          >
            Save
          </button>
          <button
            onClick={() => setIsRenaming(false)}
            class="px-2 py-1.5 text-gray-400 hover:text-white text-xs font-mono transition-colors"
          >
            Cancel
          </button>
        </div>
      </Show>

      {/* Delete Confirmation Modal */}
      <Show when={showDeleteConfirm()}>
        <div class="bg-terminal-900 border border-danger-500 p-3 flex items-center gap-2 flex-shrink-0">
          <span class="text-xs font-mono text-danger-400 font-bold">Delete "{activeList()?.name}"?</span>
          <div class="flex-1"></div>
          <button
            onClick={deleteList}
            class="px-3 py-1.5 bg-danger-500 hover:bg-danger-600 text-white text-xs font-bold font-mono transition-colors"
          >
            Confirm Delete
          </button>
          <button
            onClick={() => setShowDeleteConfirm(false)}
            class="px-2 py-1.5 text-gray-400 hover:text-white text-xs font-mono transition-colors"
          >
            Cancel
          </button>
        </div>
      </Show>

      {/* Compact View */}
      <Show when={viewMode() === 'compact' && activeList() && !error()}>
        <div class="flex-1 overflow-y-auto bg-terminal-900 border border-terminal-750">
          <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-px bg-terminal-750">
            <For each={filteredSymbols()}>
              {(item) => (
                <button
                  onClick={() => navigate(`/symbol/${item.symbol}`)}
                  class="bg-terminal-900 p-3 hover:bg-terminal-850 transition-colors text-left"
                >
                  <div class="flex items-center justify-between mb-1">
                    <span class="text-sm font-mono font-bold text-white">{item.symbol}</span>
                    <span class={`text-xs font-mono font-bold ${(item.change || 0) >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                      {(item.change_pct || 0) >= 0 ? '+' : ''}{formatPercent(item.change_pct || 0)}
                    </span>
                  </div>
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-mono text-gray-400">{formatCurrency(item.price || 0)}</span>
                    <div class="flex items-center gap-2">
                      <div class="w-12 h-4">
                        <Sparkline 
                          data={getSparklineData(item)} 
                          height={16} 
                          color={(item.change || 0) >= 0 ? '#22c55e' : '#ef4444'} 
                        />
                      </div>
                      <div onClick={(e) => e.stopPropagation()}>
                        <WatchlistAnalyzer symbol={item.symbol} />
                      </div>
                    </div>
                  </div>
                </button>
              )}
            </For>
          </div>
        </div>
      </Show>

      {/* Error State */}
      <Show when={error()}>
        <div class="flex-1 flex items-center justify-center bg-terminal-900 border border-terminal-750">
          <div class="text-center">
            <Star class="w-12 h-12 text-danger-500 mx-auto mb-3" />
            <p class="text-sm font-mono text-danger-400 mb-1">{error()}</p>
            <p class="text-xs font-mono text-gray-600 mb-4">You need to be logged in to manage watchlists</p>
            <button
              onClick={() => navigate('/login')}
              class="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white text-xs font-bold font-mono transition-colors"
            >
              Go to Login
            </button>
          </div>
        </div>
      </Show>

      {/* Table View */}
      <Show when={viewMode() === 'table' && !error()}>
        <div class="flex-1 min-h-0 overflow-auto">
          <Show when={activeList()} fallback={
            <div class="flex items-center justify-center h-full bg-terminal-900 border border-terminal-750">
              <div class="text-center">
                <Star class="w-12 h-12 text-gray-700 mx-auto mb-3" />
                <p class="text-sm font-mono text-gray-400 mb-1">No watchlists yet</p>
                <p class="text-xs font-mono text-gray-600 mb-4">Create a watchlist to track your favorite symbols</p>
                <button
                  onClick={() => setShowNewList(true)}
                  class="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white text-xs font-bold font-mono transition-colors"
                >
                  Create First Watchlist
                </button>
              </div>
            </div>
          }>
            {/* Desktop Table */}
            <div class="hidden md:block h-full">
              <Table
                data={filteredSymbols()}
                columns={symbolColumns}
                loading={loading()}
                emptyMessage="No symbols in this watchlist. Add symbols above."
                onRowClick={(item) => navigate(`/symbol/${item.symbol}`)}
                compact
                hoverable
              />
            </div>

            {/* Mobile List View */}
            <div class="md:hidden h-full overflow-y-auto bg-terminal-950">
              <Show when={filteredSymbols().length > 0} fallback={
                <div class="text-center py-12 text-gray-500 text-sm font-mono">
                  No symbols in this watchlist.
                </div>
              }>
                <div class="divide-y divide-terminal-800">
                  <For each={filteredSymbols()}>
                    {(item) => (
                      <button
                        onClick={() => navigate(`/symbol/${item.symbol}`)}
                        class="w-full bg-terminal-900 p-4 hover:bg-terminal-800 transition-colors text-left"
                      >
                        <div class="flex items-center justify-between mb-2">
                          <div class="flex flex-col items-start">
                            <span class="text-base font-mono font-bold text-white">{item.symbol}</span>
                            <span class="text-[10px] text-gray-500 uppercase truncate max-w-[150px]">{item.name}</span>
                          </div>
                          <div class="flex flex-col items-end">
                            <span class="text-base font-mono font-bold text-white">{formatCurrency(item.price || 0)}</span>
                            <span class={`text-xs font-mono font-bold ${(item.change || 0) >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                              {(item.change_pct || 0) >= 0 ? '+' : ''}{formatPercent(item.change_pct || 0)}
                            </span>
                          </div>
                        </div>
                        <div class="flex items-center justify-between gap-4">
                          <div class="flex items-center gap-3 text-[10px] text-gray-500 font-mono">
                            <span>VOL: {formatLargeNumber(item.volume || 0)}</span>
                            <span class="w-1 h-1 rounded-full bg-terminal-700"></span>
                            <span>MCAP: {formatLargeNumber(item.market_cap || 0)}</span>
                          </div>
                          <div class="flex items-center gap-3">
                            <div class="w-24 h-8 opacity-80">
                              <Sparkline 
                                data={getSparklineData(item)} 
                                height={32} 
                                color={(item.change || 0) >= 0 ? '#22c55e' : '#ef4444'} 
                              />
                            </div>
                            <div onClick={(e) => e.stopPropagation()}>
                              <WatchlistAnalyzer symbol={item.symbol} />
                            </div>
                          </div>
                        </div>
                      </button>
                    )}
                  </For>
                </div>
              </Show>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}
