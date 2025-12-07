/**
 * Professional Position Detail Page
 * 
 * Deep dive into a single position with:
 * - Position overview and metrics
 * - P&L breakdown
 * - Related orders
 * - Transaction history
 * - Quick actions (Add, Close, Set Alert)
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 */
import { createSignal, createEffect, Show } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import { Table, Column } from '~/components/ui/Table';
import { apiClient } from '~/lib/api/client';
import { formatCurrency, formatPercent } from '~/lib/utils/format';
import { Plus, X, Bell } from 'lucide-solid';

type Tab = 'overview' | 'orders' | 'transactions';

export default function PositionDetailPage() {
  const params = useParams();
  const navigate = useNavigate();
  
  const [position, setPosition] = createSignal<any>(null);
  const [orders, setOrders] = createSignal<any[]>([]);
  const [transactions, setTransactions] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [activeTab, setActiveTab] = createSignal<Tab>('overview');
  
  // Alias for position getter
  const pos = position;

  const fetchPosition = async () => {
    try {
      setLoading(true);
      const data = await apiClient.getPosition(params.symbol);
      setPosition(data);
    } catch (err) {
      console.error('Failed to load position:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchOrders = async () => {
    try {
      const data = await apiClient.getOrders({ symbol: params.symbol });
      setOrders(data);
    } catch (err) {
      console.error('Failed to load orders:', err);
    }
  };

  const fetchTransactions = async () => {
    try {
      // Note: Transactions endpoint doesn't filter by symbol currently
      // Getting all transactions - could be enhanced to filter client-side
      const data = await apiClient.getTransactions({ limit: 100 });
      // Filter by symbol client-side for now
      const filtered = data.filter((t: any) => t.symbol === params.symbol);
      setTransactions(filtered);
    } catch (err) {
      console.error('Failed to load transactions:', err);
    }
  };

  createEffect(() => {
    fetchPosition();
    fetchOrders();
    fetchTransactions();
  });

  const closePosition = async () => {
    if (!confirm(`Close entire ${params.symbol} position?`)) return;
    try {
      const position = pos();
      if (!position) return;
      
      // Close position by submitting market order in opposite direction
      await apiClient.submitOrder({
        symbol: params.symbol,
        side: position.quantity > 0 ? 'sell' : 'buy',
        order_type: 'market',
        quantity: Math.abs(position.quantity),
        time_in_force: 'day'
      });
      
      navigate('/portfolio');
    } catch (err: any) {
      alert(`Failed to close position: ${err.message}`);
    }
  };

  const orderColumns: Column<any>[] = [
    {
      key: 'created_at',
      label: 'TIME',
      sortable: true,
      align: 'left',
      render: (order) => (
        <span class="font-mono text-xs text-gray-400">
          {new Date(order.created_at).toLocaleString()}
        </span>
      ),
    },
    {
      key: 'side',
      label: 'SIDE',
      align: 'center',
      render: (order) => (
        <span class={order.side === 'buy' ? 'text-success-400' : 'text-danger-400'}>
          {order.side.toUpperCase()}
        </span>
      ),
    },
    {
      key: 'order_type',
      label: 'TYPE',
      align: 'center',
      render: (order) => <span class="font-mono text-xs">{order.order_type.toUpperCase()}</span>,
    },
    {
      key: 'quantity',
      label: 'QTY',
      sortable: true,
      align: 'right',
      render: (order) => <span class="font-mono tabular-nums">{order.quantity}</span>,
    },
    {
      key: 'price',
      label: 'PRICE',
      sortable: true,
      align: 'right',
      render: (order) => (
        <span class="font-mono tabular-nums">
          {order.limit_price ? formatCurrency(order.limit_price) : 'MKT'}
        </span>
      ),
    },
    {
      key: 'status',
      label: 'STATUS',
      align: 'center',
      render: (order) => (
        <span class={`text-xs font-mono font-bold px-2 py-0.5 ${
          order.status === 'filled' ? 'bg-success-900/30 text-success-400 border border-success-700' :
          order.status === 'open' ? 'bg-accent-900/30 text-accent-400 border border-accent-700' :
          'bg-gray-800 text-gray-500 border border-gray-700'
        }`}>
          {order.status.toUpperCase()}
        </span>
      ),
    },
  ];

  const txnColumns: Column<any>[] = [
    {
      key: 'created_at',
      label: 'DATE/TIME',
      sortable: true,
      align: 'left',
      render: (txn) => (
        <span class="font-mono text-xs text-gray-400">
          {new Date(txn.created_at).toLocaleString()}
        </span>
      ),
    },
    {
      key: 'type',
      label: 'TYPE',
      align: 'center',
      render: (txn) => <span class="font-mono text-xs">{txn.type.toUpperCase()}</span>,
    },
    {
      key: 'quantity',
      label: 'QUANTITY',
      sortable: true,
      align: 'right',
      render: (txn) => <span class="font-mono tabular-nums">{txn.quantity || '-'}</span>,
    },
    {
      key: 'price',
      label: 'PRICE',
      sortable: true,
      align: 'right',
      render: (txn) => (
        <span class="font-mono tabular-nums">
          {txn.price ? formatCurrency(txn.price) : '-'}
        </span>
      ),
    },
    {
      key: 'amount',
      label: 'AMOUNT',
      sortable: true,
      align: 'right',
      render: (txn) => (
        <span class={`font-mono tabular-nums ${txn.amount >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
          {txn.amount >= 0 ? '+' : ''}{formatCurrency(txn.amount)}
        </span>
      ),
    },
  ];

  return (
    <div class="h-full flex flex-col gap-2">
      {/* Top Bar - Position Summary */}
      <div class="bg-terminal-900 border border-terminal-750 p-2 sm:p-3">
        <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-2 lg:gap-0">
          <div class="flex flex-wrap items-center gap-2 sm:gap-4">
            <button
              onClick={() => navigate('/portfolio')}
              class="text-gray-500 hover:text-white text-xs font-mono"
            >
              ← BACK
            </button>
            <h2 class="text-base sm:text-lg font-mono font-bold text-white">{params.symbol}</h2>
            <Show when={position()}>
              <div class="flex flex-wrap items-center gap-3 sm:gap-6 text-xs font-mono">
                <div>
                  <span class="text-gray-500 uppercase">Quantity</span>
                  <span class="ml-2 text-white font-bold">{position()!.quantity}</span>
                </div>
                <div class="h-4 w-px bg-terminal-750" />
                <div>
                  <span class="text-gray-500 uppercase">Avg Cost</span>
                  <span class="ml-2 text-white">{formatCurrency(position()!.avg_cost || 0)}</span>
                </div>
                <div class="h-4 w-px bg-terminal-750" />
                <div>
                  <span class="text-gray-500 uppercase">Current</span>
                  <span class="ml-2 text-white">{formatCurrency(position()!.current_price)}</span>
                </div>
                <div class="h-4 w-px bg-terminal-750" />
                <div>
                  <span class="text-gray-500 uppercase">P&L</span>
                  <span class={`ml-2 font-bold ${position()!.unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                    {position()!.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(position()!.unrealized_pnl)}
                  </span>
                  <span class={`ml-1 ${position()!.unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                    ({position()!.unrealized_pnl >= 0 ? '+' : ''}{formatPercent(position()!.unrealized_pnl_pct)})
                  </span>
                </div>
              </div>
            </Show>
          </div>

          {/* Actions */}
          <div class="flex flex-wrap items-center gap-2 w-full lg:w-auto">
            <button
              onClick={() => navigate('/trading', { state: { symbol: params.symbol, side: 'buy' } })}
              class="px-3 py-1.5 bg-success-500 hover:bg-success-600 text-black text-xs font-bold font-mono transition-colors flex items-center gap-1.5"
            >
              <Plus class="w-3.5 h-3.5" />
              ADD
            </button>
            <button
              onClick={closePosition}
              class="px-3 py-1.5 bg-danger-500 hover:bg-danger-600 text-black text-xs font-bold font-mono transition-colors flex items-center gap-1.5"
            >
              <X class="w-3.5 h-3.5" />
              CLOSE
            </button>
            <button
              class="px-3 py-1.5 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white text-xs font-mono transition-colors flex items-center gap-1.5"
            >
              <Bell class="w-3.5 h-3.5" />
              ALERT
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div class="bg-terminal-900 border border-terminal-750 p-2">
        <div class="flex items-center gap-1">
          {(['overview', 'orders', 'transactions'] as Tab[]).map(tab => (
            <button
              onClick={() => setActiveTab(tab)}
              class={`px-3 sm:px-4 py-1.5 text-[10px] sm:text-xs font-mono font-bold transition-colors ${
                activeTab() === tab
                  ? 'bg-accent-500 text-black'
                  : 'bg-terminal-850 text-gray-400 hover:bg-terminal-800 border border-terminal-750'
              }`}
            >
              {tab.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div class="flex-1 overflow-y-auto p-2 sm:p-3">
        <Show when={!loading() && position()} fallback={
          <div class="flex items-center justify-center h-full">
            <span class="text-xs font-mono text-gray-600">Loading position...</span>
          </div>
        }>
          {/* Overview Tab */}
          <Show when={activeTab() === 'overview'}>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-3">
              <div class="bg-terminal-900 border border-terminal-750 p-3">
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Market Value</div>
                <div class="text-2xl font-mono font-bold text-white tabular-nums">
                  {formatCurrency(position()!.market_value)}
                </div>
              </div>
              <div class="bg-terminal-900 border border-terminal-750 p-3">
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Total Cost</div>
                <div class="text-2xl font-mono font-bold text-white tabular-nums">
                  {formatCurrency(position()!.total_cost || 0)}
                </div>
              </div>
              <div class="bg-terminal-900 border border-terminal-750 p-3">
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Unrealized P&L</div>
                <div class={`text-2xl font-mono font-bold tabular-nums ${
                  position()!.unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'
                }`}>
                  {position()!.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(position()!.unrealized_pnl)}
                </div>
              </div>
              <div class="bg-terminal-900 border border-terminal-750 p-3">
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Realized P&L</div>
                <div class={`text-2xl font-mono font-bold tabular-nums ${
                  (position()!.realized_pnl || 0) >= 0 ? 'text-success-400' : 'text-danger-400'
                }`}>
                  {(position()!.realized_pnl || 0) >= 0 ? '+' : ''}{formatCurrency(position()!.realized_pnl || 0)}
                </div>
              </div>
              <div class="bg-terminal-900 border border-terminal-750 p-3">
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Day P&L</div>
                <div class={`text-2xl font-mono font-bold tabular-nums ${
                  (position()!.day_pnl || 0) >= 0 ? 'text-success-400' : 'text-danger-400'
                }`}>
                  {(position()!.day_pnl || 0) >= 0 ? '+' : ''}{formatCurrency(position()!.day_pnl || 0)}
                </div>
              </div>
              <div class="bg-terminal-900 border border-terminal-750 p-3">
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Total Return</div>
                <div class={`text-2xl font-mono font-bold tabular-nums ${
                  position()!.unrealized_pnl >= 0 ? 'text-success-400' : 'text-danger-400'
                }`}>
                  {position()!.unrealized_pnl >= 0 ? '+' : ''}{formatPercent(position()!.unrealized_pnl_pct)}
                </div>
              </div>
            </div>
          </Show>

          {/* Orders Tab */}
          <Show when={activeTab() === 'orders'}>
            <Table
              data={orders()}
              columns={orderColumns}
              loading={false}
              emptyMessage="No orders for this position."
              onRowClick={(order) => navigate(`/order/${order.id}`)}
              compact
              hoverable
            />
          </Show>

          {/* Transactions Tab */}
          <Show when={activeTab() === 'transactions'}>
            <Table
              data={transactions()}
              columns={txnColumns}
              loading={false}
              emptyMessage="No transactions for this position."
              compact
              hoverable
            />
          </Show>
        </Show>
      </div>
    </div>
  );
}
