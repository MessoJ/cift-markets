/**
 * FUNDING TRANSACTION DETAIL PAGE
 * Complete drill-down with timeline, status tracking, and actions
 */

import { createSignal, createEffect, Show } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import {
  ArrowLeft,
  ArrowDownRight,
  ArrowUpRight,
  Building2,
  TrendingUp,
  CreditCard,
  CheckCircle2,
  XCircle,
  Clock,
  AlertCircle,
  FileText,
  Download,
  RefreshCw,
} from 'lucide-solid';
import { apiClient, FundingTransaction } from '../../lib/api/client';
import { formatCurrency } from '../../lib/utils';

export default function FundingTransactionDetail() {
  const params = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = createSignal(true);
  const [transaction, setTransaction] = createSignal<FundingTransaction | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  createEffect(() => {
    loadTransaction();
  });

  const loadTransaction = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getFundingTransaction(params.id);
      setTransaction(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load transaction');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!confirm('Are you sure you want to cancel this transaction?')) return;
    
    try {
      await apiClient.cancelFundingTransaction(params.id);
      await loadTransaction();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleDownloadReceipt = async () => {
    try {
      // Download PDF receipt from backend using API client
      const blob = await apiClient.downloadReceipt(params.id);
      
      // Create blob URL and trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `receipt_${params.id}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to download receipt');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 size={24} class="text-success-500" />;
      case 'failed':
      case 'cancelled':
      case 'returned': return <XCircle size={24} class="text-danger-500" />;
      case 'processing': return <Clock size={24} class="text-warning-500" />;
      default: return <AlertCircle size={24} class="text-gray-500" />;
    }
  };

  const getMethodIcon = (method: string) => {
    switch (method) {
      case 'ach': return <Building2 size={20} />;
      case 'wire': return <TrendingUp size={20} />;
      case 'card': return <CreditCard size={20} />;
      default: return <FileText size={20} />;
    }
  };

  return (
    <div class="h-full flex flex-col gap-2 sm:gap-3 p-2 sm:p-4">
      {/* Header */}
      <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
        <div class="flex items-center gap-4">
          <button
            onClick={() => navigate('/funding')}
            class="w-8 h-8 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 rounded flex items-center justify-center transition-colors"
          >
            <ArrowLeft size={16} class="text-gray-400" />
          </button>
          
          <div class="flex-1">
            <div class="flex items-center gap-3">
              <h1 class="text-base sm:text-lg font-bold text-white">Transaction Details</h1>
              <Show when={transaction()}>
                <span class="text-sm text-gray-500">
                  ID: {transaction()!.id}
                </span>
              </Show>
            </div>
            <p class="text-xs text-gray-400 mt-1">
              Complete transaction information and timeline
            </p>
          </div>

          <Show when={transaction() && (transaction()!.status === 'pending' || transaction()!.status === 'processing')}>
            <button
              onClick={handleCancel}
              class="px-4 py-2 bg-danger-500/10 hover:bg-danger-500/20 border border-danger-500/30 text-danger-500 text-sm font-semibold rounded transition-colors"
            >
              Cancel Transaction
            </button>
          </Show>
        </div>
      </div>

      <Show when={loading()}>
        <div class="flex-1 flex items-center justify-center">
          <div class="text-center">
            <RefreshCw size={32} class="text-gray-600 animate-spin mx-auto mb-3" />
            <div class="text-gray-500">Loading transaction...</div>
          </div>
        </div>
      </Show>

      <Show when={error()}>
        <div class="bg-danger-500/10 border border-danger-500/20 p-4 rounded flex items-center gap-3">
          <AlertCircle size={20} class="text-danger-500" />
          <span class="text-sm text-danger-500">{error()}</span>
        </div>
      </Show>

      <Show when={!loading() && transaction()}>
        <div class="flex-1 overflow-auto grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-2 sm:gap-3">
          {/* Left: Main Details */}
          <div class="space-y-3">
            {/* Status Card */}
            <div class="bg-terminal-900 border border-terminal-750 p-4 sm:p-6">
              <div class="flex items-start gap-4">
                <div class={`w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  transaction()!.type === 'deposit' ? 'bg-success-500/10' : 'bg-danger-500/10'
                }`}>
                  {transaction()!.type === 'deposit' ? (
                    <ArrowDownRight size={32} class="text-success-500" />
                  ) : (
                    <ArrowUpRight size={32} class="text-danger-500" />
                  )}
                </div>

                <div class="flex-1">
                  <div class="flex items-center gap-3 mb-2">
                    <h2 class="text-2xl font-bold text-white">
                      {transaction()!.type === 'deposit' ? '+' : '-'}
                      {formatCurrency(transaction()!.amount)}
                    </h2>
                    <div class="flex items-center gap-2 px-3 py-1.5 bg-terminal-850 border border-terminal-750 rounded">
                      {getStatusIcon(transaction()!.status)}
                      <span class="text-sm font-semibold text-white capitalize">
                        {transaction()!.status}
                      </span>
                    </div>
                  </div>

                  <div class="text-sm text-gray-400">
                    {transaction()!.type === 'deposit' ? 'Deposit' : 'Withdrawal'} via {transaction()!.method.toUpperCase()}
                    {transaction()!.bank_account_last4 && ` • Account ending in ${transaction()!.bank_account_last4}`}
                  </div>
                </div>
              </div>

              <Show when={transaction()!.failed_reason}>
                <div class="mt-4 p-3 bg-danger-500/10 border border-danger-500/20 rounded">
                  <div class="flex items-start gap-2">
                    <XCircle size={16} class="text-danger-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <div class="text-sm font-semibold text-danger-500 mb-1">Transaction Failed</div>
                      <div class="text-xs text-gray-400">{transaction()!.failed_reason}</div>
                    </div>
                  </div>
                </div>
              </Show>
            </div>

            {/* Transaction Details */}
            <div class="bg-terminal-900 border border-terminal-750">
              <div class="p-3 sm:p-4 border-b border-terminal-750">
                <h3 class="text-sm font-bold text-white">Transaction Information</h3>
              </div>
              <div class="p-3 sm:p-4">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Transaction ID</div>
                    <div class="text-sm text-white font-mono">{transaction()!.id}</div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">External ID</div>
                    <div class="text-sm text-white font-mono">{transaction()!.external_id || 'N/A'}</div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Type</div>
                    <div class="text-sm text-white capitalize">{transaction()!.type}</div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Method</div>
                    <div class="flex items-center gap-2 text-sm text-white">
                      {getMethodIcon(transaction()!.method)}
                      <span class="uppercase">{transaction()!.method}</span>
                    </div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Amount</div>
                    <div class="text-sm text-white font-bold tabular-nums">
                      {formatCurrency(transaction()!.amount)}
                    </div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Fee</div>
                    <div class="text-sm text-white tabular-nums">
                      {transaction()!.fee > 0 ? formatCurrency(transaction()!.fee) : 'FREE'}
                    </div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Net Amount</div>
                    <div class="text-sm text-white font-bold tabular-nums">
                      {formatCurrency(transaction()!.net_amount)}
                    </div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Currency</div>
                    <div class="text-sm text-white">{transaction()!.currency}</div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Initiated</div>
                    <div class="text-sm text-white">
                      {new Date(transaction()!.created_at).toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-500 mb-1">Last Updated</div>
                    <div class="text-sm text-white">
                      {new Date(transaction()!.updated_at).toLocaleString()}
                    </div>
                  </div>
                  <Show when={transaction()!.estimated_completion}>
                    <div>
                      <div class="text-xs text-gray-500 mb-1">Est. Completion</div>
                      <div class="text-sm text-warning-500">
                        {new Date(transaction()!.estimated_completion!).toLocaleString()}
                      </div>
                    </div>
                  </Show>
                  <Show when={transaction()!.completed_at}>
                    <div>
                      <div class="text-xs text-gray-500 mb-1">Completed</div>
                      <div class="text-sm text-success-500">
                        {new Date(transaction()!.completed_at!).toLocaleString()}
                      </div>
                    </div>
                  </Show>
                </div>
              </div>
            </div>

            {/* Payment Method Details */}
            <div class="bg-terminal-900 border border-terminal-750">
              <div class="p-3 sm:p-4 border-b border-terminal-750">
                <h3 class="text-sm font-bold text-white">Payment Method</h3>
              </div>
              <div class="p-3 sm:p-4">
                <div class="flex items-center gap-3">
                  <div class={`w-12 h-12 rounded flex items-center justify-center ${
                    transaction()!.method === 'ach' ? 'bg-primary-500/10' : 'bg-accent-500/10'
                  }`}>
                    <div class={transaction()!.method === 'ach' ? 'text-primary-500' : 'text-accent-500'}>
                      {getMethodIcon(transaction()!.method)}
                    </div>
                  </div>
                  <div>
                    <div class="font-semibold text-white">
                      {transaction()!.method === 'ach' && 'Bank Account'}
                      {transaction()!.method === 'wire' && 'Wire Transfer'}
                      {transaction()!.method === 'card' && 'Debit Card'}
                    </div>
                    {transaction()!.bank_account_last4 && (
                      <div class="text-sm text-gray-400">
                        Account ending in {transaction()!.bank_account_last4}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Timeline & Actions */}
          <div class="space-y-3">
            {/* Actions */}
            <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
              <h3 class="text-sm font-bold text-white mb-3">Actions</h3>
              <div class="space-y-2">
                <button
                  onClick={handleDownloadReceipt}
                  class="w-full flex items-center gap-2 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-white text-sm rounded transition-colors"
                >
                  <Download size={16} />
                  <span>Download Receipt (PDF)</span>
                </button>
                
                <Show when={transaction()!.status === 'pending' || transaction()!.status === 'processing'}>
                  <button
                    onClick={handleCancel}
                    class="w-full flex items-center gap-2 px-4 py-2 bg-danger-500/10 hover:bg-danger-500/20 border border-danger-500/30 text-danger-500 text-sm rounded transition-colors"
                  >
                    <XCircle size={16} />
                    <span>Cancel Transaction</span>
                  </button>
                </Show>
              </div>
            </div>

            {/* Status Timeline */}
            <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
              <h3 class="text-sm font-bold text-white mb-4">Status Timeline</h3>
              <div class="space-y-4">
                {/* Initiated */}
                <div class="flex gap-3">
                  <div class="flex flex-col items-center">
                    <div class="w-8 h-8 rounded-full bg-success-500/10 border-2 border-success-500 flex items-center justify-center flex-shrink-0">
                      <CheckCircle2 size={16} class="text-success-500" />
                    </div>
                    <div class="w-0.5 h-full bg-terminal-750 mt-1"></div>
                  </div>
                  <div class="flex-1 pb-4">
                    <div class="font-semibold text-white text-sm">Transaction Initiated</div>
                    <div class="text-xs text-gray-400 mt-0.5">
                      {new Date(transaction()!.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>

                {/* Processing */}
                <div class="flex gap-3">
                  <div class="flex flex-col items-center">
                    <div class={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      transaction()!.status === 'processing' || transaction()!.status === 'completed'
                        ? 'bg-warning-500/10 border-2 border-warning-500'
                        : 'bg-terminal-850 border-2 border-terminal-750'
                    }`}>
                      <Clock size={16} class={
                        transaction()!.status === 'processing' || transaction()!.status === 'completed'
                          ? 'text-warning-500'
                          : 'text-gray-600'
                      } />
                    </div>
                    <div class="w-0.5 h-full bg-terminal-750 mt-1"></div>
                  </div>
                  <div class="flex-1 pb-4">
                    <div class={`font-semibold text-sm ${
                      transaction()!.status === 'processing' || transaction()!.status === 'completed'
                        ? 'text-white'
                        : 'text-gray-600'
                    }`}>
                      Processing
                    </div>
                    <div class="text-xs text-gray-400 mt-0.5">
                      {transaction()!.status === 'processing' || transaction()!.status === 'completed'
                        ? 'In progress...'
                        : 'Pending'}
                    </div>
                  </div>
                </div>

                {/* Completed */}
                <div class="flex gap-3">
                  <div class="flex flex-col items-center">
                    <div class={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      transaction()!.status === 'completed'
                        ? 'bg-success-500/10 border-2 border-success-500'
                        : 'bg-terminal-850 border-2 border-terminal-750'
                    }`}>
                      <CheckCircle2 size={16} class={
                        transaction()!.status === 'completed' ? 'text-success-500' : 'text-gray-600'
                      } />
                    </div>
                  </div>
                  <div class="flex-1">
                    <div class={`font-semibold text-sm ${
                      transaction()!.status === 'completed' ? 'text-white' : 'text-gray-600'
                    }`}>
                      Completed
                    </div>
                    <div class="text-xs text-gray-400 mt-0.5">
                      {transaction()!.completed_at
                        ? new Date(transaction()!.completed_at).toLocaleString()
                        : transaction()!.estimated_completion
                        ? `Est: ${new Date(transaction()!.estimated_completion).toLocaleDateString()}`
                        : 'Pending completion'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Help */}
            <div class="bg-primary-500/5 border border-primary-500/20 p-4 rounded">
              <div class="text-xs font-semibold text-primary-500 mb-2">Need Help?</div>
              <div class="text-xs text-gray-400 mb-3">
                If you have questions about this transaction, our support team is here to help.
              </div>
              <button
                onClick={() => navigate('/support')}
                class="text-xs text-primary-500 hover:underline font-semibold"
              >
                Contact Support →
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
