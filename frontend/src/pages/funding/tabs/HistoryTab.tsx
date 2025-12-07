import { createSignal, For, Show } from 'solid-js';
import { 
  ArrowDownRight, 
  ArrowUpRight, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  AlertCircle, 
  ChevronRight, 
  BarChart3, 
  FileText, 
  Download, 
  Loader2 
} from 'lucide-solid';
import { FundingTransaction } from '../../../lib/api/client';
import { formatCurrency } from '../../../lib/utils';
import { Modal } from '../../../components/ui/Modal';

interface HistoryTabProps {
  transactions: FundingTransaction[] | undefined;
}

export function HistoryTab(props: HistoryTabProps) {
  const [filterType, setFilterType] = createSignal<'all' | 'deposit' | 'withdrawal'>('all');
  const [filterStatus, setFilterStatus] = createSignal<string>('all');
  
  // Modal State
  const [selectedTransaction, setSelectedTransaction] = createSignal<FundingTransaction | null>(null);
  const [showDownloadModal, setShowDownloadModal] = createSignal(false);
  const [downloadProgress, setDownloadProgress] = createSignal(0);

  const filteredTransactions = () => {
    let result = props.transactions || [];
    
    if (filterType() !== 'all') {
      result = result.filter((t) => t.type === filterType());
    }
    
    if (filterStatus() !== 'all') {
      result = result.filter((t) => t.status === filterStatus());
    }
    
    return result;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 size={14} class="text-success-500" />;
      case 'failed':
      case 'cancelled':
      case 'returned': return <XCircle size={14} class="text-danger-500" />;
      case 'processing': return <Clock size={14} class="text-warning-500 animate-pulse" />;
      default: return <AlertCircle size={14} class="text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-success-400 bg-success-900/20 border border-success-900/50';
      case 'failed':
      case 'cancelled':
      case 'returned': return 'text-danger-400 bg-danger-900/20 border border-danger-900/50';
      case 'processing': return 'text-warning-400 bg-warning-900/20 border border-warning-900/50';
      default: return 'text-gray-400 bg-terminal-800 border border-terminal-700';
    }
  };

  const handleDownload = () => {
    setShowDownloadModal(true);
    setDownloadProgress(0);
    
    const interval = setInterval(() => {
      setDownloadProgress(p => {
        if (p >= 100) {
          clearInterval(interval);
          setTimeout(() => setShowDownloadModal(false), 1000);
          return 100;
        }
        return p + 5;
      });
    }, 100);
  };

  return (
    <div class="bg-terminal-900 border border-terminal-750 h-full flex flex-col rounded-xl overflow-hidden">
      
      {/* Analytics & Documents Header */}
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 border-b border-terminal-750 bg-terminal-800/30">
        {/* Net Deposits Chart Placeholder */}
        <div class="md:col-span-2 bg-terminal-900/50 border border-terminal-750 p-4 rounded-xl relative overflow-hidden group">
          <div class="flex items-center justify-between mb-3">
            <h4 class="text-xs font-mono font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
              <BarChart3 size={14} />
              Net Deposits (YTD)
            </h4>
            <span class="text-xs font-mono text-success-400 font-bold bg-success-900/20 px-2 py-1 rounded-md border border-success-900/30">+12.5%</span>
          </div>
          <div class="h-16 flex items-end gap-1.5">
            {/* Fake Sparkline Bars */}
            <For each={[40, 65, 45, 80, 55, 90, 70, 85, 60, 75, 95, 100]}>
              {(height) => (
                <div 
                  class="flex-1 bg-accent-900/40 hover:bg-accent-500 transition-all duration-300 rounded-t-sm hover:h-full"
                  style={{ height: `${height}%` }}
                ></div>
              )}
            </For>
          </div>
        </div>

        {/* Tax Documents Link */}
        <button 
          onClick={handleDownload}
          class="bg-terminal-900/50 border border-terminal-750 p-4 rounded-xl flex flex-col justify-between group hover:border-accent-500/50 hover:bg-terminal-800/50 transition-all text-left"
        >
          <div class="flex items-start justify-between w-full">
            <h4 class="text-xs font-mono font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
              <FileText size={14} />
              Tax Documents
            </h4>
            <div class="p-1.5 rounded-lg bg-terminal-800 group-hover:bg-accent-500/20 group-hover:text-accent-400 transition-colors">
              <Download size={14} />
            </div>
          </div>
          <div>
            <div class="text-sm font-bold text-white mb-1 group-hover:text-accent-400 transition-colors">2023 Form 1099-B</div>
            <div class="text-[10px] text-gray-500 group-hover:text-gray-400">Ready to download</div>
          </div>
        </button>
      </div>

      {/* Filters */}
      <div class="p-3 sm:p-4 border-b border-terminal-750 bg-terminal-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div class="flex flex-wrap items-center gap-3 sm:gap-4">
          <div>
            <label class="text-[10px] font-mono text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Type</label>
            <div class="relative">
              <select
                value={filterType()}
                onChange={(e) => setFilterType(e.target.value as any)}
                class="bg-terminal-800 border border-terminal-700 text-white text-xs font-mono pl-3 pr-8 py-2 rounded-lg focus:outline-none focus:border-accent-500 appearance-none cursor-pointer hover:border-gray-600 transition-colors min-w-[140px]"
              >
                <option value="all">ALL TRANSACTIONS</option>
                <option value="deposit">DEPOSITS ONLY</option>
                <option value="withdrawal">WITHDRAWALS ONLY</option>
              </select>
              <div class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                <ChevronRight size={12} class="rotate-90" />
              </div>
            </div>
          </div>
          
          <div>
            <label class="text-[10px] font-mono text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Status</label>
            <div class="relative">
              <select
                value={filterStatus()}
                onChange={(e) => setFilterStatus(e.target.value)}
                class="bg-terminal-800 border border-terminal-700 text-white text-xs font-mono pl-3 pr-8 py-2 rounded-lg focus:outline-none focus:border-accent-500 appearance-none cursor-pointer hover:border-gray-600 transition-colors min-w-[140px]"
              >
                <option value="all">ALL STATUSES</option>
                <option value="pending">PENDING</option>
                <option value="processing">PROCESSING</option>
                <option value="completed">COMPLETED</option>
                <option value="failed">FAILED</option>
                <option value="cancelled">CANCELLED</option>
              </select>
              <div class="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                <ChevronRight size={12} class="rotate-90" />
              </div>
            </div>
          </div>

          <div class="ml-auto text-xs font-mono text-gray-500 bg-terminal-800 px-3 py-1.5 rounded-full">
            {filteredTransactions().length} RECORD{filteredTransactions().length !== 1 ? 'S' : ''}
          </div>
        </div>
      </div>

      {/* Transaction List */}
      <div class="flex-1 overflow-auto custom-scrollbar">
        <Show when={filteredTransactions().length === 0}>
          <div class="flex items-center justify-center h-full min-h-[200px]">
            <div class="text-center p-8 border border-dashed border-terminal-800 rounded-xl bg-terminal-900/50">
              <div class="text-gray-500 mb-2 font-mono text-sm font-bold">NO TRANSACTIONS FOUND</div>
              <div class="text-xs text-gray-600 font-mono">Your funding history will appear here</div>
            </div>
          </div>
        </Show>

        <div class="divide-y divide-terminal-800">
          <For each={filteredTransactions()}>
            {(transaction) => (
              <div class="p-4 hover:bg-terminal-800/50 transition-colors group">
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center gap-4">
                    <div class={`p-2.5 rounded-xl ${transaction.type === 'deposit' ? 'bg-success-900/20 text-success-400' : 'bg-terminal-800 text-gray-400'}`}>
                      {transaction.type === 'deposit' ? <ArrowDownRight size={18} /> : <ArrowUpRight size={18} />}
                    </div>
                    <div>
                      <div class="font-bold text-white text-sm font-mono mb-0.5">
                        {transaction.type === 'deposit' ? 'Deposit' : 'Withdrawal'}
                      </div>
                      <div class="text-xs text-gray-500 font-mono flex items-center gap-2">
                        <span>{new Date(transaction.created_at).toLocaleDateString()}</span>
                        <span class="w-1 h-1 rounded-full bg-terminal-700"></span>
                        <span>{new Date(transaction.created_at).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  </div>
                  <div class="text-right">
                    <div class={`font-mono font-bold text-sm ${transaction.type === 'deposit' ? 'text-success-400' : 'text-white'}`}>
                      {transaction.type === 'deposit' ? '+' : '-'}{formatCurrency(transaction.amount)}
                    </div>
                    <div class={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-mono uppercase tracking-wider mt-1.5 font-bold ${getStatusColor(transaction.status)}`}>
                      {getStatusIcon(transaction.status)}
                      {transaction.status}
                    </div>
                  </div>
                </div>
                
                <div class="flex items-center justify-between text-xs text-gray-600 font-mono pl-[52px]">
                  <div class="font-mono">ID: <span class="text-gray-500">{transaction.id.substring(0, 8)}...</span></div>
                  <button 
                    onClick={() => setSelectedTransaction(transaction)}
                    class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all text-accent-500 cursor-pointer hover:text-accent-400 font-bold tracking-wider text-[10px] uppercase"
                  >
                    View Details <ChevronRight size={12} />
                  </button>
                </div>
              </div>
            )}
          </For>
        </div>
      </div>

      {/* Transaction Details Modal */}
      <Modal
        open={!!selectedTransaction()}
        onClose={() => setSelectedTransaction(null)}
        title="Transaction Details"
        size="sm"
      >
        <Show when={selectedTransaction()}>
          {(t) => (
            <div class="space-y-6">
              <div class="text-center py-4 border-b border-terminal-800">
                <div class={`text-3xl font-mono font-bold mb-2 ${t().type === 'deposit' ? 'text-success-400' : 'text-white'}`}>
                  {t().type === 'deposit' ? '+' : '-'}{formatCurrency(t().amount)}
                </div>
                <div class={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono uppercase tracking-wider font-bold ${getStatusColor(t().status)}`}>
                  {getStatusIcon(t().status)}
                  {t().status}
                </div>
              </div>

              <div class="space-y-4">
                <div class="flex justify-between items-center">
                  <span class="text-xs text-gray-500 uppercase tracking-wider font-bold">Type</span>
                  <span class="text-sm text-white font-mono capitalize">{t().type}</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-xs text-gray-500 uppercase tracking-wider font-bold">Date</span>
                  <span class="text-sm text-white font-mono">{new Date(t().created_at).toLocaleString()}</span>
                </div>
                <div class="flex justify-between items-center">
                  <span class="text-xs text-gray-500 uppercase tracking-wider font-bold">Transaction ID</span>
                  <span class="text-sm text-white font-mono text-right break-all max-w-[200px]">{t().id}</span>
                </div>
                <Show when={t().payment_method_id}>
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500 uppercase tracking-wider font-bold">Payment Method</span>
                    <span class="text-sm text-white font-mono">Linked Account</span>
                  </div>
                </Show>
              </div>

              <div class="pt-4 border-t border-terminal-800">
                <button 
                  onClick={() => setSelectedTransaction(null)}
                  class="w-full py-3 bg-terminal-800 hover:bg-terminal-700 text-white rounded-lg text-xs font-bold uppercase tracking-wider transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          )}
        </Show>
      </Modal>

      {/* Download Modal */}
      <Modal
        open={showDownloadModal()}
        onClose={() => {}}
        title="Downloading Document"
        size="sm"
        showCloseButton={false}
        closeOnOverlayClick={false}
      >
        <div class="text-center py-6">
          <div class="mb-6 relative">
            <div class="w-16 h-16 bg-terminal-800 rounded-full flex items-center justify-center mx-auto text-accent-500">
              <FileText size={32} />
            </div>
            <Show when={downloadProgress() < 100}>
              <div class="absolute inset-0 flex items-center justify-center">
                <Loader2 size={64} class="text-accent-500 animate-spin opacity-50" />
              </div>
            </Show>
            <Show when={downloadProgress() === 100}>
              <div class="absolute bottom-0 right-1/2 translate-x-8 translate-y-2 bg-success-500 rounded-full p-1 border-2 border-terminal-900">
                <CheckCircle2 size={16} class="text-white" />
              </div>
            </Show>
          </div>
          
          <h4 class="text-lg font-bold text-white mb-2">2023 Form 1099-B</h4>
          <p class="text-sm text-gray-400 mb-6">
            {downloadProgress() < 100 ? 'Preparing your document...' : 'Download complete!'}
          </p>

          <div class="w-full bg-terminal-800 h-2 rounded-full overflow-hidden mb-2">
            <div 
              class="bg-accent-500 h-full transition-all duration-100 ease-out"
              style={{ width: `${downloadProgress()}%` }}
            ></div>
          </div>
          <div class="text-right text-xs font-mono text-gray-500">
            {downloadProgress()}%
          </div>
        </div>
      </Modal>
    </div>
  );
}
