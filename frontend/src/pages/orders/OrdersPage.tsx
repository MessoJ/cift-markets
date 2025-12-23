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

import { createSignal, createEffect, Show, onCleanup } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { Table, Column } from '~/components/ui/Table';
import { apiClient } from '~/lib/api/client';
import { formatCurrency, formatPercent } from '~/lib/utils/format';
import { X, RefreshCw, Trash2, Filter, Calendar } from 'lucide-solid';
import { DateRangePicker, DateRange, defaultPresets } from '~/components/ui/DateRangePicker';

type OrderStatus = 'all' | 'open' | 'filled' | 'cancelled';

export default function OrdersPage() {
  const navigate = useNavigate();
  
  const [orders, setOrders] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [activeTab, setActiveTab] = createSignal<OrderStatus>('all');
  const [symbolFilter, setSymbolFilter] = createSignal('');
  const [dateRange, setDateRange] = createSignal<DateRange>({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    endDate: new Date(),
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

  // Initial fetch and auto-refresh
  createEffect(() => {
    // Initial fetch with sync to ensure we have latest broker data
    fetchOrders(true);
    const interval = setInterval(() => fetchOrders(false), 5000); // Poll every 5s
    onCleanup(() => clearInterval(interval));
  });

  // Re-fetch when tab changes
  createEffect(() => {
    activeTab(); // Dependency
    setLoading(true);
    fetchOrders();
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
    const start = dateRange().startDate.getTime();
    const end = dateRange().endDate.getTime();
    filtered = filtered.filter(o => {
      const time = new Date(o.created_at).getTime();
      return time >= start && time <= end;
    });

    return filtered;
  };

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
      // Assuming API has a bulk cancel or we loop
      // Ideally: await apiClient.cancelAllOrders();
      // Fallback: Loop
      const openOrders = orders().filter(o => o.status === 'open');
      await Promise.all(openOrders.map(o => apiClient.cancelOrder(o.id)));
      await fetchOrders();
    } catch (err: any) {
      alert(`Failed to cancel all orders: ${err.message}`);
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
            order.status === 'open' ? 'bg-accent-500/10 text-accent-400 border border-accent-500/20 animate-pulse-slow' :
            order.status === 'cancelled' ? 'bg-gray-700/30 text-gray-400 border border-gray-600/30' :
            'bg-gray-800 text-gray-400'
          }`}
        >
          <Show when={order.status === 'open'}>
            <div class="w-1.5 h-1.5 rounded-full bg-accent-400 animate-ping" />
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
          onRowClick={(order) => navigate(`/order/${order.id}`)}
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
