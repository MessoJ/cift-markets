/**
 * Professional Transactions Page
 * 
 * Complete transaction history with:
 * - All account activity (trades, deposits, withdrawals, fees, dividends)
 * - Date range filtering
 * - Type filtering
 * - Symbol filtering
 * - Export functionality
 * - Summary Dashboard
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 */

import { createSignal, createEffect, Show, For } from 'solid-js';
import { Table, Column } from '~/components/ui/Table';
import { apiClient } from '~/lib/api/client';
import { formatCurrency } from '~/lib/utils/format';
import { 
  Download, 
  Search, 
  ArrowUpRight, 
  ArrowDownLeft, 
  Activity,
  X,
  Copy,
  ExternalLink,
  Calendar
} from 'lucide-solid';

// Types
type TransactionType = 'all' | 'trade' | 'deposit' | 'withdrawal' | 'fee' | 'dividend' | 'interest' | 'adjustment';

interface TransactionSummary {
  period: {
    start_date: string;
    end_date: string;
    days: number;
  };
  stats: {
    total_inflow: number;
    total_outflow: number;
    net_flow: number;
  };
}

// Components

const SummaryCard = (props: { title: string; value: number; type: 'inflow' | 'outflow' | 'net' }) => (
  <div class="bg-terminal-900 border border-terminal-750 p-4 rounded-sm flex-1 min-w-[200px]">
    <div class="flex items-center justify-between mb-2">
      <span class="text-xs font-mono text-gray-400 uppercase">{props.title}</span>
      <div class={`p-1.5 rounded-full ${
        props.type === 'inflow' ? 'bg-success-900/20 text-success-400' :
        props.type === 'outflow' ? 'bg-danger-900/20 text-danger-400' :
        'bg-accent-900/20 text-accent-400'
      }`}>
        {props.type === 'inflow' && <ArrowDownLeft class="w-4 h-4" />}
        {props.type === 'outflow' && <ArrowUpRight class="w-4 h-4" />}
        {props.type === 'net' && <Activity class="w-4 h-4" />}
      </div>
    </div>
    <div class={`text-xl font-mono font-bold ${
      props.type === 'inflow' ? 'text-success-400' :
      props.type === 'outflow' ? 'text-danger-400' :
      props.value >= 0 ? 'text-success-400' : 'text-danger-400'
    }`}>
      {props.value >= 0 && props.type !== 'outflow' ? '+' : ''}
      {formatCurrency(props.value)}
    </div>
    <div class="text-[10px] text-gray-500 mt-1 font-mono">
      Last 30 days
    </div>
  </div>
);

const TransactionDetailsModal = (props: { transaction: any; onClose: () => void }) => {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add toast here
  };

  return (
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={props.onClose}>
      <div class="bg-terminal-900 border border-terminal-700 w-full max-w-md m-4 shadow-2xl" onClick={e => e.stopPropagation()}>
        <div class="flex items-center justify-between p-4 border-b border-terminal-700">
          <h3 class="text-sm font-bold font-mono text-white">TRANSACTION DETAILS</h3>
          <button onClick={props.onClose} class="text-gray-400 hover:text-white">
            <X class="w-4 h-4" />
          </button>
        </div>
        
        <div class="p-6 space-y-4">
          <div class="flex flex-col items-center justify-center py-4 border-b border-terminal-800">
            <div class={`text-3xl font-mono font-bold mb-1 ${
              props.transaction.amount >= 0 ? 'text-success-400' : 'text-danger-400'
            }`}>
              {props.transaction.amount >= 0 ? '+' : ''}{formatCurrency(props.transaction.amount)}
            </div>
            <div class="text-xs font-mono text-gray-400 uppercase tracking-wider">
              {props.transaction.transaction_type}
            </div>
          </div>

          <div class="space-y-3">
            <div class="flex justify-between items-center">
              <span class="text-xs text-gray-500 font-mono">STATUS</span>
              <span class="text-xs text-success-400 font-mono font-bold bg-success-900/20 px-2 py-0.5 rounded border border-success-900/50">
                COMPLETED
              </span>
            </div>
            
            <div class="flex justify-between items-center">
              <span class="text-xs text-gray-500 font-mono">DATE</span>
              <span class="text-xs text-white font-mono">
                {new Date(props.transaction.transaction_date).toLocaleString()}
              </span>
            </div>

            <Show when={props.transaction.symbol}>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-500 font-mono">SYMBOL</span>
                <span class="text-xs text-white font-mono font-bold">{props.transaction.symbol}</span>
              </div>
            </Show>

            <Show when={props.transaction.quantity}>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-500 font-mono">QUANTITY</span>
                <span class="text-xs text-white font-mono">{props.transaction.quantity}</span>
              </div>
            </Show>

            <Show when={props.transaction.price}>
              <div class="flex justify-between items-center">
                <span class="text-xs text-gray-500 font-mono">PRICE</span>
                <span class="text-xs text-white font-mono">{formatCurrency(props.transaction.price)}</span>
              </div>
            </Show>

            <div class="flex justify-between items-center">
              <span class="text-xs text-gray-500 font-mono">BALANCE AFTER</span>
              <span class="text-xs text-white font-mono">{formatCurrency(props.transaction.balance_after)}</span>
            </div>

            <div class="pt-3 border-t border-terminal-800">
              <Show when={props.transaction.order_id}>
                <div class="flex justify-between items-center mb-2">
                  <span class="text-xs text-gray-500 font-mono">ORDER ID</span>
                  <button 
                    onClick={() => copyToClipboard(props.transaction.order_id)}
                    class="text-xs text-accent-400 hover:text-accent-300 font-mono flex items-center gap-1"
                  >
                    <Copy class="w-3 h-3" />
                    COPY
                  </button>
                </div>
                <div class="text-[10px] text-gray-600 font-mono break-all mb-3">
                  {props.transaction.order_id}
                </div>
              </Show>

              <div class="flex justify-between items-center mb-1">
                <span class="text-xs text-gray-500 font-mono">TRANSACTION ID</span>
                <button 
                  onClick={() => copyToClipboard(props.transaction.id)}
                  class="text-xs text-accent-400 hover:text-accent-300 font-mono flex items-center gap-1"
                >
                  <Copy class="w-3 h-3" />
                  COPY
                </button>
              </div>
              <div class="text-[10px] text-gray-600 font-mono break-all">
                {props.transaction.id}
              </div>
            </div>

            <Show when={props.transaction.external_ref}>
              <div class="pt-2">
                <div class="flex justify-between items-center mb-1">
                  <span class="text-xs text-gray-500 font-mono">EXTERNAL REF</span>
                </div>
                <div class="text-[10px] text-gray-600 font-mono break-all">
                  {props.transaction.external_ref}
                </div>
              </div>
            </Show>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function TransactionsPage() {
  // State
  const [transactions, setTransactions] = createSignal<any[]>([]);
  const [summary, setSummary] = createSignal<TransactionSummary | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [selectedTransaction, setSelectedTransaction] = createSignal<any | null>(null);
  
  // Filters
  const [typeFilter, setTypeFilter] = createSignal<TransactionType>('all');
  const [symbolFilter, setSymbolFilter] = createSignal('');
  const [startDate, setStartDate] = createSignal('');
  const [endDate, setEndDate] = createSignal('');
  
  // Pagination
  const [page, setPage] = createSignal(1);
  const [limit] = createSignal(50);
  const [total, setTotal] = createSignal(0);

  // Date Presets
  const setDateRange = (range: '1M' | '3M' | 'YTD' | 'ALL') => {
    const end = new Date();
    let start = new Date();
    
    switch (range) {
      case '1M':
        start.setMonth(end.getMonth() - 1);
        break;
      case '3M':
        start.setMonth(end.getMonth() - 3);
        break;
      case 'YTD':
        start = new Date(end.getFullYear(), 0, 1);
        break;
      case 'ALL':
        start = new Date(2020, 0, 1); // Far past
        break;
    }
    
    setStartDate(start.toISOString().split('T')[0]);
    setEndDate(end.toISOString().split('T')[0]);
  };

  // Fetch Data
  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Capture signals synchronously before await
      const currentType = typeFilter();
      const currentSymbol = symbolFilter();
      const currentStart = startDate();
      const currentEnd = endDate();
      const currentLimit = limit();
      const currentPage = page();

      // Fetch Summary (once or on date change)
      const summaryData = await apiClient.getTransactionSummary(30);
      setSummary(summaryData);

      // Fetch Transactions
      const params: any = {
        limit: currentLimit,
        offset: (currentPage - 1) * currentLimit
      };
      
      if (currentType !== 'all') params.transaction_type = currentType;
      if (currentSymbol) params.symbol = currentSymbol;
      if (currentStart) params.start_date = new Date(currentStart).toISOString();
      if (currentEnd) params.end_date = new Date(currentEnd).toISOString();
      
      const data = await apiClient.getTransactions(params);
      
      // Handle response structure
      if (data && data.transactions) {
        setTransactions(data.transactions);
        setTotal(data.pagination.total);
      } else if (Array.isArray(data)) {
        // Fallback for old API structure
        setTransactions(data);
        setTotal(data.length);
      }
      
    } catch (err) {
      console.error('Failed to load transactions:', err);
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    fetchData();
  });

  // Export
  const exportCSV = () => {
    const headers = ['Date', 'ID', 'Type', 'Symbol', 'Description', 'Quantity', 'Price', 'Amount', 'Balance', 'Ref'];
    const rows = transactions().map(t => [
      new Date(t.transaction_date).toISOString(),
      t.id,
      t.transaction_type,
      t.symbol || '',
      `"${t.description}"`, // Quote description to handle commas
      t.quantity || '',
      t.price || '',
      t.amount,
      t.balance_after,
      t.external_ref || ''
    ]);
    
    const csv = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transactions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  // Table Columns
  const columns: Column<any>[] = [
    {
      key: 'transaction_date',
      label: 'DATE',
      sortable: true,
      align: 'left',
      render: (txn) => (
        <div class="flex flex-col">
          <span class="font-mono text-xs text-white">
            {new Date(txn.transaction_date).toLocaleDateString()}
          </span>
          <span class="font-mono text-[10px] text-gray-500">
            {new Date(txn.transaction_date).toLocaleTimeString()}
          </span>
        </div>
      ),
    },
    {
      key: 'transaction_type',
      label: 'TYPE',
      sortable: true,
      align: 'center',
      render: (txn) => {
        const type = txn.transaction_type;
        const style = 
          type === 'trade' ? 'bg-primary-900/30 text-primary-400 border-primary-700' :
          type === 'deposit' ? 'bg-success-900/30 text-success-400 border-success-700' :
          type === 'withdrawal' ? 'bg-danger-900/30 text-danger-400 border-danger-700' :
          type === 'fee' ? 'bg-gray-800 text-gray-400 border-gray-700' :
          type === 'dividend' ? 'bg-accent-900/30 text-accent-400 border-accent-700' :
          'bg-gray-800 text-gray-400 border-gray-700';
          
        return (
          <span class={`text-[10px] font-mono font-bold px-2 py-0.5 border rounded-sm uppercase ${style}`}>
            {type}
          </span>
        );
      },
    },
    {
      key: 'symbol',
      label: 'ASSET',
      sortable: true,
      align: 'left',
      render: (txn) => (
        <span class={`font-mono font-bold ${txn.symbol ? 'text-white' : 'text-gray-600'}`}>
          {txn.symbol || '-'}
        </span>
      ),
    },
    {
      key: 'description',
      label: 'DESCRIPTION',
      align: 'left',
      render: (txn) => (
        <span class="text-xs text-gray-400 truncate max-w-[200px] block" title={txn.description}>
          {txn.description}
        </span>
      ),
    },
    {
      key: 'quantity',
      label: 'QTY',
      sortable: true,
      align: 'right',
      render: (txn) => (
        <span class="font-mono tabular-nums text-xs text-gray-300">
          {txn.quantity ? txn.quantity.toLocaleString() : '-'}
        </span>
      ),
    },
    {
      key: 'price',
      label: 'PRICE',
      sortable: true,
      align: 'right',
      render: (txn) => (
        <span class="font-mono tabular-nums text-xs text-gray-300">
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
        <span class={`font-mono tabular-nums font-bold text-sm ${
          txn.amount >= 0 ? 'text-success-400' : 'text-danger-400'
        }`}>
          {txn.amount >= 0 ? '+' : ''}{formatCurrency(txn.amount)}
        </span>
      ),
    },
    {
      key: 'balance_after',
      label: 'BALANCE',
      sortable: true,
      align: 'right',
      render: (txn) => (
        <span class="font-mono tabular-nums text-xs text-gray-400">
          {formatCurrency(txn.balance_after)}
        </span>
      ),
    },
    {
      key: 'actions',
      label: '',
      align: 'right',
      render: (txn) => (
        <button 
          onClick={(e) => { e.stopPropagation(); setSelectedTransaction(txn); }}
          class="p-1 hover:bg-terminal-800 rounded text-gray-500 hover:text-white transition-colors"
        >
          <ExternalLink class="w-3 h-3" />
        </button>
      ),
    },
  ];

  return (
    <div class="h-full flex flex-col gap-4 p-4 overflow-hidden">
      {/* Summary Dashboard */}
      <Show when={summary()}>
        <div class="flex flex-wrap gap-4">
          <SummaryCard title="Total Inflow" value={summary()!.stats.total_inflow} type="inflow" />
          <SummaryCard title="Total Outflow" value={summary()!.stats.total_outflow} type="outflow" />
          <SummaryCard title="Net Flow" value={summary()!.stats.net_flow} type="net" />
        </div>
      </Show>

      {/* Main Content Area */}
      <div class="flex-1 flex flex-col bg-terminal-900 border border-terminal-750 rounded-sm overflow-hidden">
        
        {/* Toolbar */}
        <div class="p-3 border-b border-terminal-750 flex flex-col lg:flex-row gap-3 justify-between items-start lg:items-center bg-terminal-850/50">
          
          {/* Left: Filters */}
          <div class="flex flex-wrap items-center gap-3 w-full lg:w-auto">
            {/* Search */}
            <div class="relative group">
              <Search class="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500 group-focus-within:text-accent-500" />
              <input
                type="text"
                value={symbolFilter()}
                onInput={(e) => setSymbolFilter(e.currentTarget.value.toUpperCase())}
                placeholder="Search Symbol..."
                class="bg-terminal-900 border border-terminal-700 text-white text-xs font-mono pl-8 pr-3 py-1.5 w-32 focus:w-48 transition-all focus:outline-none focus:border-accent-500 rounded-sm"
              />
            </div>

            {/* Type Filter */}
            <div class="flex items-center bg-terminal-900 border border-terminal-700 rounded-sm p-0.5">
              <For each={['all', 'trade', 'deposit', 'withdrawal', 'fee']}>
                {(type) => (
                  <button
                    onClick={() => setTypeFilter(type as TransactionType)}
                    class={`px-3 py-1 text-[10px] font-mono font-bold uppercase transition-colors rounded-sm ${
                      typeFilter() === type
                        ? 'bg-accent-500 text-black'
                        : 'text-gray-400 hover:text-white hover:bg-terminal-800'
                    }`}
                  >
                    {type}
                  </button>
                )}
              </For>
            </div>
          </div>

          {/* Right: Date & Export */}
          <div class="flex items-center gap-2 w-full lg:w-auto justify-end">
            {/* Date Presets */}
            <div class="flex items-center bg-terminal-900 border border-terminal-700 rounded-sm p-0.5">
              <For each={['1M', '3M', 'YTD', 'ALL']}>
                {(range) => (
                  <button
                    onClick={() => setDateRange(range as any)}
                    class="px-2 py-1 text-[10px] font-mono font-bold text-gray-400 hover:text-white hover:bg-terminal-800 rounded-sm transition-colors"
                  >
                    {range}
                  </button>
                )}
              </For>
            </div>

            <div class="flex items-center bg-terminal-900 border border-terminal-700 rounded-sm px-2 py-1 gap-2">
              <Calendar class="w-3.5 h-3.5 text-gray-500" />
              <input
                type="date"
                value={startDate()}
                onInput={(e) => setStartDate(e.currentTarget.value)}
                class="bg-transparent text-white font-mono text-xs focus:outline-none w-24"
              />
              <span class="text-gray-600 text-xs">—</span>
              <input
                type="date"
                value={endDate()}
                onInput={(e) => setEndDate(e.currentTarget.value)}
                class="bg-transparent text-white font-mono text-xs focus:outline-none w-24"
              />
            </div>

            <button
              onClick={exportCSV}
              disabled={transactions().length === 0}
              class="flex items-center gap-1.5 px-3 py-1.5 bg-terminal-850 hover:bg-terminal-700 border border-terminal-700 text-gray-300 hover:text-white text-xs font-mono transition-colors rounded-sm disabled:opacity-50"
            >
              <Download class="w-3.5 h-3.5" />
              EXPORT
            </button>
          </div>
        </div>

        {/* Table */}
        <div class="flex-1 overflow-hidden relative">
          <Table
            data={transactions()}
            columns={columns}
            loading={loading()}
            emptyMessage="No transactions found matching your criteria."
            compact
            hoverable
            onRowClick={(txn) => setSelectedTransaction(txn)}
          />
        </div>

        {/* Pagination Footer */}
        <div class="p-2 border-t border-terminal-750 bg-terminal-850/50 flex justify-between items-center text-xs font-mono text-gray-500">
          <div>
            Showing {transactions().length} of {total()} transactions
          </div>
          <div class="flex gap-2">
            <button 
              disabled={page() === 1}
              onClick={() => setPage(p => p - 1)}
              class="px-2 py-1 hover:text-white disabled:opacity-30"
            >
              PREV
            </button>
            <span class="text-white">Page {page()}</span>
            <button 
              disabled={(page() * limit()) >= total()}
              onClick={() => setPage(p => p + 1)}
              class="px-2 py-1 hover:text-white disabled:opacity-30"
            >
              NEXT
            </button>
          </div>
        </div>
      </div>

      {/* Details Modal */}
      <Show when={selectedTransaction()}>
        <TransactionDetailsModal 
          transaction={selectedTransaction()} 
          onClose={() => setSelectedTransaction(null)} 
        />
      </Show>
    </div>
  );
}
