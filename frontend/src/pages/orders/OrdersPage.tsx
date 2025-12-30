/**
 * Professional Orders Page
 * 
 * Order management with:
 * - Tabbed filtering (Open/Filled/Cancelled/All)
 * - Search and filters
 * - Order actions (Cancel, Modify, View Details)
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 */

import { createSignal, createEffect, Show, onCleanup, onMount, For, createMemo } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { Table, Column } from '~/components/ui/Table';
import { apiClient } from '~/lib/api/client';
import { formatCurrency, formatPercent } from '~/lib/utils/format';
import { X, RefreshCw, Trash2, Filter, Calendar, TrendingUp, Clock, CheckCircle, XCircle } from 'lucide-solid';
import { DateRangePicker, DateRange, defaultPresets } from '~/components/ui/DateRangePicker';

type OrderStatus = 'all' | 'open' | 'filled' | 'cancelled';


const ExpandedOrderRow = (props: { order: any }) => {
  const [details, setDetails] = createSignal<any>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);

  onMount(async () => {
    try {
      const data = await apiClient.getOrder(props.order.id);
      // Handle both direct response and nested order response
      setDetails(data.order || data);
    } catch (err: any) {
      console.error('Failed to load order details:', err);
      setError(err.message || 'Failed to load details');
    } finally {
      setLoading(false);
    }
  });

  return (
    <div class="p-4 text-sm bg-terminal-900/50">
      <Show when={!loading()} fallback={<div class="text-gray-500 flex items-center gap-2"><RefreshCw class="w-3 h-3 animate-spin"/> Loading details...</div>}>
        <Show when={error()}>
          <div class="text-danger-400 text-xs">{error()}</div>
        </Show>
        <Show when={details() && !error()}>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 class="text-[10px] font-bold text-accent-500 uppercase mb-3 tracking-wider">Execution Analysis</h4>
              <div class="space-y-2 font-mono text-xs">
                <div class="flex justify-between border-b border-terminal-800 pb-1">
                  <span class="text-gray-500">Order ID</span>
                  <span class="text-gray-400 select-all">{details().id || props.order.id}</span>
                </div>
                <div class="flex justify-between border-b border-terminal-800 pb-1">
                  <span class="text-gray-500">Strategy</span>
                  <span class="text-white">{details().strategy_id || 'Manual Trade'}</span>
                </div>
                <div class="flex justify-between border-b border-terminal-800 pb-1">
                  <span class="text-gray-500">Avg Fill Price</span>
                  <span class="text-white font-bold">{formatCurrency(details().execution_quality?.avg_fill_price || 0)}</span>
                </div>
                <div class="flex justify-between border-b border-terminal-800 pb-1">
                  <span class="text-gray-500">Slippage</span>
                  <span class={`${(details().execution_quality?.slippage_bps || 0) > 0 ? 'text-danger-400' : 'text-success-400'}`}>
                    {details().execution_quality?.slippage_bps || 0} bps
                  </span>
                </div>
                <div class="flex justify-between border-b border-terminal-800 pb-1">
                  <span class="text-gray-500">Total Fees</span>
                  <span class="text-gray-300">{formatCurrency(details().execution_quality?.total_commission || 0)}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 class="text-[10px] font-bold text-accent-500 uppercase mb-3 tracking-wider">Fills & Events</h4>
              <Show when={details().fills && details().fills.length > 0} fallback={<div class="text-gray-500 italic text-xs">No fills recorded</div>}>
                <div class="overflow-x-auto">
                  <table class="w-full text-xs font-mono">
                    <thead>
                      <tr class="text-gray-500 text-left border-b border-terminal-800">
                        <th class="pb-2 font-normal">Time</th>
                        <th class="pb-2 text-right font-normal">Qty</th>
                        <th class="pb-2 text-right font-normal">Price</th>
                        <th class="pb-2 text-right font-normal">Venue</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-terminal-800/50">
                      <For each={details().fills}>
                        {(fill: any) => (
                          <tr>
                            <td class="py-2 text-gray-400">{new Date(fill.timestamp).toLocaleTimeString()}</td>
                            <td class="py-2 text-right text-white">{fill.quantity}</td>
                            <td class="py-2 text-right text-white">{formatCurrency(fill.price)}</td>
                            <td class="py-2 text-right text-gray-500">{fill.venue || 'EXCH'}</td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>
              </Show>
            </div>
          </div>
        </Show>
      </Show>
    </div>
  );
};

export default function OrdersPage() {
  const navigate = useNavigate();
  
  const [orders, setOrders] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [activeTab, setActiveTab] = createSignal<OrderStatus>('all');
  const [symbolFilter, setSymbolFilter] = createSignal('');
  const [expandedOrderId, setExpandedOrderId] = createSignal<string | null>(null);
  const [dateRange, setDateRange] = createSignal<DateRange>({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    end: new Date(),
    label: 'Last 30 Days'
  });
  
  const fetchOrders = async (forceSync = false) => {
    try {
      // Don't set loading on background refreshes if we already have data
      if (orders().length === 0 || forceSync) setLoading(true);
      
      const status = activeTab() === 'all' ? undefined : activeTab();
      const data = await apiClient.getOrders({ 
        status,
        sync: forceSync
      });
      setOrders(data);
    } catch (err) {
      console.error('Failed to load orders:', err);
    } finally {
      setLoading(false);
    }
  };

  // Track if this is the first mount
  let isFirstMount = true;
  
  // Single effect for fetching - tracks activeTab changes
  createEffect(() => {
    const tab = activeTab(); // Track dependency
    
    // Only show loading spinner if we don't have data or it's a tab change
    if (orders().length === 0 || !isFirstMount) {
      setLoading(true);
    }
    
    // Force sync on first mount, regular fetch on tab changes
    fetchOrders(isFirstMount);
    isFirstMount = false;
  });

  // Separate auto-refresh interval - doesn't depend on activeTab
  onMount(() => {
    const interval = setInterval(() => {
      // Background refresh - don't show loading
      fetchOrders(false);
    }, 30000);
    onCleanup(() => clearInterval(interval));
  });

  const filteredOrders = () => {
    let filtered = orders();
    
    // Symbol Filter
    if (symbolFilter()) {
      filtered = filtered.filter(o => 
        o.symbol.toLowerCase().includes(symbolFilter().toLowerCase())
      );
    }

    // Date Filter (Client-side filtering if API doesn't support it yet)
    const start = dateRange().start.getTime();
    const end = dateRange().end.getTime();
    filtered = filtered.filter(o => {
      const time = new Date(o.created_at).getTime();
      return time >= start && time <= end;
    });

    return filtered;
  };

  // Compute order statistics
  const orderStats = createMemo(() => {
    const all = filteredOrders();
    const open = all.filter(o => o.status === 'open');
    const filled = all.filter(o => o.status === 'filled');
    const cancelled = all.filter(o => o.status === 'cancelled');
    
    const filledValue = filled.reduce((sum, o) => 
      sum + (o.filled_quantity || 0) * (o.avg_fill_price || o.limit_price || 0), 0);
    
    const pendingValue = open.reduce((sum, o) => 
      sum + (o.quantity - (o.filled_quantity || 0)) * (o.limit_price || 0), 0);
    
    const fillRate = all.length > 0 
      ? (filled.length / (filled.length + cancelled.length)) * 100 
      : 0;
    
    return {
      total: all.length,
      open: open.length,
      filled: filled.length,
      cancelled: cancelled.length,
      filledValue,
      pendingValue,
      fillRate: isNaN(fillRate) ? 0 : fillRate,
    };
  });

  const cancelOrder = async (orderId: string) => {
    if (!confirm('Cancel this order?')) return;
    try {
      await apiClient.cancelOrder(orderId);
      await fetchOrders();
    } catch (err: any) {
      alert(`Failed to cancel order: ${err.message}`);
    }
  };

  const cancelAllOpen = async () => {
    const openCount = orders().filter(o => o.status === 'open').length;
    if (openCount === 0) return;
    
    if (!confirm(`Are you sure you want to CANCEL ALL ${openCount} open orders?`)) return;
    
    try {
      // Use bulk cancel API endpoint
      await apiClient.cancelAllOrders();
      await fetchOrders(true);
    } catch (err: any) {
      console.error('Bulk cancel failed, trying individual cancellations:', err);
      // Fallback to individual cancellations if bulk fails
      try {
        const openOrders = orders().filter(o => o.status === 'open');
        for (const order of openOrders) {
          await apiClient.cancelOrder(order.id);
        }
        await fetchOrders(true);
      } catch (fallbackErr: any) {
        alert(`Failed to cancel orders: ${fallbackErr.message}`);
      }
    }
  };

  const orderColumns: Column<any>[] = [
    {
      key: 'symbol',
      label: 'SYMBOL',
      sortable: true,
      align: 'left',
      render: (order) => (
        <div class="flex flex-col">
          <span class="font-mono font-bold text-white text-sm">{order.symbol}</span>
          <span class="text-[10px] text-gray-500 font-mono">{order.exchange || 'US'}</span>
        </div>
      ),
    },
    {
      key: 'side',
      label: 'SIDE',
      align: 'center',
      render: (order) => (
        <span class={`font-bold text-xs px-2 py-0.5 rounded-sm ${
          order.side === 'buy' 
            ? 'bg-success-900/20 text-success-400 border border-success-900/50' 
            : 'bg-danger-900/20 text-danger-400 border border-danger-900/50'
        }`}>
          {order.side.toUpperCase()}
        </span>
      ),
    },
    {
      key: 'order_type',
      label: 'TYPE',
      align: 'left',
      render: (order) => (
        <div class="flex flex-col">
          <span class="font-mono text-xs text-gray-300">{order.order_type.toUpperCase()}</span>
          <span class="text-[10px] text-gray-500">{order.time_in_force || 'GTC'}</span>
        </div>
      ),
    },
    {
      key: 'quantity',
      label: 'QTY / FILLED',
      sortable: true,
      align: 'right',
      render: (order) => {
        const pct = (order.filled_quantity / order.quantity) * 100;
        return (
          <div class="flex flex-col items-end w-24">
            <div class="flex items-baseline gap-1">
              <span class="font-mono text-white">{order.filled_quantity}</span>
              <span class="text-gray-500 text-[10px]">/ {order.quantity}</span>
            </div>
            {/* Progress Bar */}
            <div class="w-full h-1 bg-terminal-800 mt-1 rounded-full overflow-hidden">
              <div 
                class={`h-full ${order.side === 'buy' ? 'bg-success-500' : 'bg-danger-500'}`} 
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      },
    },
    {
      key: 'price',
      label: 'PRICE',
      sortable: true,
      align: 'right',
      render: (order) => (
        <div class="flex flex-col items-end">
          <span class="font-mono text-white">
            {order.limit_price ? formatCurrency(order.limit_price) : 'MKT'}
          </span>
          <Show when={order.avg_fill_price}>
            <span class="text-[10px] text-gray-500">
              Avg: {formatCurrency(order.avg_fill_price)}
            </span>
          </Show>
        </div>
      ),
    },
    {
      key: 'value',
      label: 'VALUE',
      align: 'right',
      render: (order) => (
        <span class="font-mono text-gray-300">
          {formatCurrency((order.filled_quantity || 0) * (order.avg_fill_price || order.limit_price || 0))}
        </span>
      ),
    },
    {
      key: 'status',
      label: 'STATUS',
      align: 'center',
      render: (order) => (
        <span 
          class={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full flex items-center justify-center gap-1 w-24 ${
            order.status === 'filled' ? 'bg-success-500/10 text-success-400 border border-success-500/20' :
            order.status === 'open' ? 'bg-accent-500/10 text-accent-400 border border-accent-500/20' :
            order.status === 'cancelled' ? 'bg-gray-700/30 text-gray-400 border border-gray-600/30' :
            'bg-gray-800 text-gray-400'
          }`}
        >
          <Show when={order.status === 'open'}>
            <div class="w-1.5 h-1.5 rounded-full bg-accent-400" />
          </Show>
          {order.status.toUpperCase()}
        </span>
      ),
    },
    {
      key: 'created_at',
      label: 'TIME',
      sortable: true,
      align: 'right',
      render: (order) => (
        <div class="flex flex-col items-end">
          <span class="font-mono text-xs text-gray-300">
            {new Date(order.created_at).toLocaleTimeString()}
          </span>
          <span class="text-[10px] text-gray-500">
            {new Date(order.created_at).toLocaleDateString()}
          </span>
        </div>
      ),
    },
    {
      key: 'actions',
      label: '',
      align: 'right',
      render: (order) => (
        <Show when={order.status === 'open'}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              cancelOrder(order.id);
            }}
            class="p-1.5 text-gray-400 hover:text-danger-400 hover:bg-danger-900/20 rounded transition-colors"
            title="Cancel Order"
          >
            <X class="w-4 h-4" />
          </button>
        </Show>
      ),
    },
  ];

  return (
    <div class="h-full flex flex-col gap-4 p-4 max-w-[1600px] mx-auto w-full">
      {/* Header */}
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 class="text-2xl font-bold text-white flex items-center gap-2">
            <RefreshCw class={`w-6 h-6 text-primary-400 ${loading() ? 'animate-spin' : ''}`} />
            Orders
          </h1>
          <p class="text-gray-400 text-sm mt-1">
            Manage your open orders and view execution history
          </p>
        </div>
        
        <div class="flex items-center gap-3">
          <button
            onClick={() => navigate('/trading')}
            class="flex items-center gap-2 px-3 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors text-sm font-medium shadow-sm shadow-primary-900/20"
          >
            <span class="text-lg leading-none">+</span>
            <span class="hidden sm:inline">New Order</span>
          </button>

          <button
            onClick={() => fetchOrders(true)}
            class="flex items-center gap-2 px-3 py-2 bg-terminal-800 text-gray-300 border border-terminal-700 rounded-md hover:bg-terminal-700 transition-colors text-sm font-medium"
            title="Sync with Broker"
          >
            <RefreshCw class={`w-4 h-4 ${loading() ? 'animate-spin' : ''}`} />
            <span class="hidden sm:inline">Sync</span>
          </button>
          
          <DateRangePicker 
            value={dateRange()} 
            onChange={setDateRange}
            presets={defaultPresets}
          />
          <button
            onClick={cancelAllOpen}
            disabled={!orders().some(o => o.status === 'open')}
            class={`flex items-center gap-2 px-3 py-2 border rounded-md transition-colors text-sm font-medium ${
              orders().some(o => o.status === 'open')
                ? 'bg-danger-900/20 text-danger-400 border-danger-900/50 hover:bg-danger-900/40 cursor-pointer'
                : 'bg-terminal-800 text-gray-600 border-terminal-700 cursor-not-allowed opacity-50'
            }`}
          >
            <Trash2 class="w-4 h-4" />
            <span class="hidden sm:inline">Cancel All Open</span>
          </button>
        </div>
      </div>

      {/* Summary Stats Cards */}
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-3">
          <div class="flex items-center gap-2 mb-1">
            <Clock class="w-4 h-4 text-accent-400" />
            <span class="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Open Orders</span>
          </div>
          <div class="flex items-baseline gap-2">
            <span class="text-2xl font-bold text-white font-mono">{orderStats().open}</span>
            <Show when={orderStats().pendingValue > 0}>
              <span class="text-xs text-gray-500">{formatCurrency(orderStats().pendingValue)}</span>
            </Show>
          </div>
        </div>
        
        <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-3">
          <div class="flex items-center gap-2 mb-1">
            <CheckCircle class="w-4 h-4 text-success-400" />
            <span class="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Filled</span>
          </div>
          <div class="flex items-baseline gap-2">
            <span class="text-2xl font-bold text-success-400 font-mono">{orderStats().filled}</span>
            <Show when={orderStats().filledValue > 0}>
              <span class="text-xs text-gray-500">{formatCurrency(orderStats().filledValue)}</span>
            </Show>
          </div>
        </div>
        
        <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-3">
          <div class="flex items-center gap-2 mb-1">
            <XCircle class="w-4 h-4 text-gray-500" />
            <span class="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Cancelled</span>
          </div>
          <div class="text-2xl font-bold text-gray-500 font-mono">{orderStats().cancelled}</div>
        </div>
        
        <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-3">
          <div class="flex items-center gap-2 mb-1">
            <TrendingUp class="w-4 h-4 text-primary-400" />
            <span class="text-[10px] font-medium text-gray-500 uppercase tracking-wider">Fill Rate</span>
          </div>
          <div class="flex items-baseline gap-1">
            <span class={`text-2xl font-bold font-mono ${orderStats().fillRate >= 80 ? 'text-success-400' : orderStats().fillRate >= 50 ? 'text-accent-400' : 'text-gray-400'}`}>
              {orderStats().fillRate.toFixed(0)}
            </span>
            <span class="text-sm text-gray-500">%</span>
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-1 flex flex-col sm:flex-row gap-2 justify-between items-center">
        {/* Tabs */}
        <div class="flex p-1 bg-terminal-950 rounded-md">
          {(['all', 'open', 'filled', 'cancelled'] as OrderStatus[]).map(tab => (
            <button
              onClick={() => setActiveTab(tab)}
              class={`px-4 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab() === tab
                  ? 'bg-terminal-800 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-terminal-900'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Filters */}
        <div class="flex items-center gap-3 w-full sm:w-auto px-2">
          <div class="relative flex-1 sm:flex-none">
            <Filter class="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
            <input
              type="text"
              value={symbolFilter()}
              onInput={(e) => setSymbolFilter(e.currentTarget.value)}
              placeholder="Filter by symbol..."
              class="w-full sm:w-48 bg-terminal-950 border border-terminal-800 text-white text-xs pl-8 pr-3 py-1.5 rounded-md focus:outline-none focus:border-primary-500 transition-colors"
            />
          </div>
          
          {/* Stats Pills */}
          <div class="hidden lg:flex items-center gap-2 text-[10px] font-mono border-l border-terminal-800 pl-3">
            <div class="px-2 py-1 rounded bg-terminal-950 border border-terminal-800 text-gray-400">
              TOTAL: <span class="text-white font-bold">{filteredOrders().length}</span>
            </div>
            <div class="px-2 py-1 rounded bg-terminal-950 border border-terminal-800 text-gray-400">
              OPEN: <span class="text-accent-400 font-bold">{filteredOrders().filter(o => o.status === 'open').length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Orders Table - Desktop */}
      <div class="hidden md:block flex-1 min-h-0 bg-terminal-900 border border-terminal-800 rounded-lg overflow-hidden shadow-sm">
        <Table
          data={filteredOrders()}
          columns={orderColumns}
          loading={loading()}
          emptyMessage={
            <div class="flex flex-col items-center justify-center py-12 text-gray-500">
              <div class="w-12 h-12 rounded-full bg-terminal-800 flex items-center justify-center mb-3">
                <Filter class="w-6 h-6 opacity-50" />
              </div>
              <p class="text-sm font-medium">No orders found</p>
              <p class="text-xs opacity-70 mt-1 max-w-xs text-center">
                Try adjusting your filters. If you have orders on Alpaca, ensure your API keys are configured.
              </p>
              <button 
                onClick={() => navigate('/trading')}
                class="mt-4 text-primary-400 hover:text-primary-300 text-xs font-medium"
              >
                Place your first trade &rarr;
              </button>
            </div>
          }
          expandedRowId={expandedOrderId()}
          onExpandChange={setExpandedOrderId}
          getRowId={(order) => order.id || order.order_id}
          renderExpandedRow={(order) => <ExpandedOrderRow order={order} />}
          compact
          hoverable
          rowClass={(order) => `
            border-b border-terminal-800/50 last:border-0 transition-colors
            ${order.status === 'open' ? 'bg-accent-900/5 hover:bg-accent-900/10' : ''}
          `}
        />
      </div>

      {/* Orders List - Mobile */}
      <div class="md:hidden flex-1 overflow-y-auto space-y-3 pb-20">
        <Show when={!loading()} fallback={
          <div class="flex justify-center py-8">
            <RefreshCw class="w-6 h-6 text-primary-400 animate-spin" />
          </div>
        }>
          <Show when={filteredOrders().length > 0} fallback={
            <div class="flex flex-col items-center justify-center py-12 text-gray-500">
              <p class="text-sm font-medium">No orders found</p>
              <button 
                onClick={() => navigate('/trading')}
                class="mt-4 text-primary-400 hover:text-primary-300 text-xs font-medium"
              >
                Place your first trade &rarr;
              </button>
            </div>
          }>
            <For each={filteredOrders()}>
              {(order) => (
                <div 
                  class="bg-terminal-900 border border-terminal-800 rounded-lg p-3 space-y-3 active:bg-terminal-800 transition-colors"
                  onClick={() => navigate(`/order/${order.id}`)}
                >
                  {/* Header: Symbol, Side, Status */}
                  <div class="flex justify-between items-start">
                    <div class="flex items-center gap-2">
                      <div class="font-bold text-white text-lg">{order.symbol}</div>
                      <span class={`text-[10px] font-bold px-2 py-0.5 rounded-sm ${
                        order.side === 'buy' 
                          ? 'bg-success-900/20 text-success-400 border border-success-900/50' 
                          : 'bg-danger-900/20 text-danger-400 border border-danger-900/50'
                      }`}>
                        {order.side.toUpperCase()}
                      </span>
                    </div>
                    <span class={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full flex items-center gap-1 ${
                      order.status === 'filled' ? 'bg-success-500/10 text-success-400 border border-success-500/20' :
                      order.status === 'open' ? 'bg-accent-500/10 text-accent-400 border border-accent-500/20' :
                      order.status === 'cancelled' ? 'bg-gray-700/30 text-gray-400 border border-gray-600/30' :
                      'bg-gray-800 text-gray-400'
                    }`}>
                      <Show when={order.status === 'open'}>
                        <div class="w-1.5 h-1.5 rounded-full bg-accent-400 animate-ping" />
                      </Show>
                      {order.status.toUpperCase()}
                    </span>
                  </div>
                  
                  {/* Progress Bar for Fills */}
                  <div class="w-full h-1 bg-terminal-800 rounded-full overflow-hidden">
                    <div 
                      class={`h-full ${order.side === 'buy' ? 'bg-success-500' : 'bg-danger-500'}`} 
                      style={{ width: `${(order.filled_quantity / order.quantity) * 100}%` }}
                    />
                  </div>

                  {/* Details Grid */}
                  <div class="grid grid-cols-2 gap-y-2 gap-x-4 text-xs">
                    <div>
                      <div class="text-gray-500 text-[10px] uppercase">Filled / Qty</div>
                      <div class="font-mono text-gray-300 text-sm">
                        <span class="text-white">{order.filled_quantity}</span>
                        <span class="text-gray-500"> / {order.quantity}</span>
                      </div>
                    </div>
                    <div class="text-right">
                      <div class="text-gray-500 text-[10px] uppercase">Price</div>
                      <div class="font-mono text-gray-300 text-sm">
                        {order.limit_price ? formatCurrency(order.limit_price) : 'MKT'}
                      </div>
                    </div>
                    <div>
                      <div class="text-gray-500 text-[10px] uppercase">Type</div>
                      <div class="font-mono text-gray-300">{order.order_type.toUpperCase()}</div>
                    </div>
                    <div class="text-right">
                      <div class="text-gray-500 text-[10px] uppercase">Value</div>
                      <div class="font-mono text-gray-300">
                        {formatCurrency((order.filled_quantity || 0) * (order.avg_fill_price || order.limit_price || 0))}
                      </div>
                    </div>
                  </div>

                  {/* Footer: Time & Actions */}
                  <div class="flex justify-between items-center pt-2 border-t border-terminal-800">
                    <div class="text-[10px] text-gray-500 font-mono">
                      {new Date(order.created_at).toLocaleString()}
                    </div>
                    <Show when={order.status === 'open'}>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          cancelOrder(order.id);
                        }}
                        class="px-3 py-1.5 bg-danger-900/20 text-danger-400 border border-danger-900/50 rounded text-xs font-medium hover:bg-danger-900/40 flex items-center gap-1"
                      >
                        <X class="w-3 h-3" /> Cancel
                      </button>
                    </Show>
                  </div>
                </div>
              )}
            </For>
          </Show>
        </Show>
      </div>
    </div>
  );
}
